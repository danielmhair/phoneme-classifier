"""
ONNX export + latency check for the FUSED PAIR (the working configuration
decided 2026-07-13, see plans/channel-augmentation-experiment.md):

    audio -> Wav2Vec2 backbone -> [T,768] -> w2v2 CTC head (channel-aug @20ep)
    audio -> WavLM    backbone -> [T,768] -> WavLM CTC head (baseline)
    fused scores = mean of the two ctc_predict probability vectors

Exports four graphs to evaluation/full_models_fused_onnx/ (gitignored via the
evaluation/full_models*/ pattern):

    wav2vec2_backbone.onnx   raw audio [1,samples] -> embeddings [1,T,768]
                             (per-utterance zero-mean/unit-var normalization is
                             BAKED INTO the graph - facebook/wav2vec2-base's
                             processor has do_normalize=True)
    wavlm_backbone.onnx      raw audio [1,samples] -> embeddings [1,T,768]
                             (no normalization - microsoft/wavlm-base's
                             processor has do_normalize=False)
    wav2vec2_ctc_head.onnx   embeddings [1,T,768] -> log_probs [1,T,38]
                             (weights: evaluation/full_models_channel_e20/)
    wavlm_ctc_head.onnx      embeddings [1,T,768] -> log_probs [1,T,38]
                             (weights: evaluation/full_models/ baseline)

plus fused_pair_metadata.json (preprocessing contract, label order, fusion
rule) and fused_pair_onnx_report.json (parity, replay accuracy, latency).

Every export fails LOUDLY on a missing/0-byte file (the repo's historical
CTC ONNX failure mode) and must pass onnx.checker + numeric parity against
the PyTorch chain on real saved live-mic clips before it counts.

Run:  poetry run python evaluation/export_fused_onnx.py
      [--skip-replay]   only export + spot parity + latency (fast)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from workflows.shared.ctc_decode import ctc_predict  # noqa: E402

OUT_DIR = REPO_ROOT / "evaluation" / "full_models_fused_onnx"
W2V2_HEAD_CKPT = REPO_ROOT / "evaluation" / "full_models_channel_e20" / "wav2vec2_ctc.pt"
WAVLM_HEAD_CKPT = REPO_ROOT / "evaluation" / "full_models" / "wavlm_ctc.pt"
LABELS_PATH = REPO_ROOT / "evaluation" / "full_models" / "canonical_labels.json"
SESSIONS = [
    REPO_ROOT / "evaluation" / "live_mic_sessions" / "dan_goodmic_20260712_154201",
    REPO_ROOT / "evaluation" / "live_mic_sessions" / "dan_badmic_20260712_155952",
]
# PyTorch-chain fused numbers from the pairing eval (2026-07-13) that the
# ONNX chain must reproduce: good mic 78.38 top-1 / 93.69 top-3,
# bad mic 52.25 top-1.
EXPECTED_FUSED = {
    "dan_goodmic_20260712_154201": {"top1": 0.7838, "top3": 0.9369},
    "dan_badmic_20260712_155952": {"top1": 0.5225, "top3": 0.7027},
}
SAMPLE_RATE = 16000
MAX_EMBED_LENGTH = 1000  # frames - matches evaluation/harness/embeddings_cache.py
OPSET = 14
LATENCY_BUDGET_S = 0.150  # temporal brain's real-time budget (CLAUDE.md Epic 2)


def check_exported(path: Path) -> None:
    """The historical failure mode: a swallowed exception leaving a 0-byte
    .onnx nobody notices until inference much later. Fail here instead."""
    import onnx

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(
            f"ONNX export produced no usable file at {path} "
            f"(exists={path.exists()}, "
            f"size={path.stat().st_size if path.exists() else 0} bytes)"
        )
    onnx.checker.check_model(onnx.load(str(path)))
    print(f"  ok: {path.name} ({path.stat().st_size / 1e6:.1f} MB, checker passed)")


def export_backbones():
    import torch
    from transformers import Wav2Vec2Model, WavLMModel

    class W2V2Backbone(torch.nn.Module):
        """facebook/wav2vec2-base with the processor's per-utterance
        zero-mean/unit-var normalization (do_normalize=True, eps=1e-7,
        population variance) baked into the graph, so deployment feeds
        raw float32 audio."""

        def __init__(self):
            super().__init__()
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()

        def forward(self, input_values):
            x = input_values - input_values.mean(dim=-1, keepdim=True)
            x = x / torch.sqrt(input_values.var(dim=-1, keepdim=True, unbiased=False) + 1e-7)
            return self.model(x).last_hidden_state

    class WavLMBackbone(torch.nn.Module):
        """microsoft/wavlm-base takes raw audio (do_normalize=False)."""

        def __init__(self):
            super().__init__()
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base").eval()

        def forward(self, input_values):
            return self.model(input_values).last_hidden_state

    example = torch.randn(1, SAMPLE_RATE)  # 1s of audio
    for name, module in [("wav2vec2_backbone", W2V2Backbone()), ("wavlm_backbone", WavLMBackbone())]:
        out_path = OUT_DIR / f"{name}.onnx"
        print(f"Exporting {name} (opset {OPSET})...")
        with torch.no_grad():
            torch.onnx.export(
                module, example, out_path,
                export_params=True, opset_version=OPSET, do_constant_folding=True,
                input_names=["audio"], output_names=["embeddings"],
                dynamic_axes={"audio": {1: "samples"}, "embeddings": {1: "frames"}},
            )
        check_exported(out_path)


def export_ctc_heads(num_classes: int):
    import torch

    from workflows.ctc_w2v2_workflow.models.ctc_model import create_ctc_model

    class CTCHeadInference(torch.nn.Module):
        def __init__(self, ctc_model):
            super().__init__()
            self.ctc_model = ctc_model

        def forward(self, embeddings):
            log_probs, _ = self.ctc_model(embeddings)
            return log_probs

    example = torch.randn(1, 149, 768)  # ~3s of frames
    for name, ckpt_path in [("wav2vec2_ctc_head", W2V2_HEAD_CKPT), ("wavlm_ctc_head", WAVLM_HEAD_CKPT)]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = create_ctc_model(num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        out_path = OUT_DIR / f"{name}.onnx"
        print(f"Exporting {name} from {ckpt_path.relative_to(REPO_ROOT)}...")
        with torch.no_grad():
            torch.onnx.export(
                CTCHeadInference(model), example, out_path,
                export_params=True, opset_version=OPSET, do_constant_folding=True,
                input_names=["embeddings"], output_names=["log_probabilities"],
                dynamic_axes={"embeddings": {1: "frames"},
                              "log_probabilities": {1: "frames"}},
            )
        check_exported(out_path)


class OnnxFusedScorer:
    """The deployable chain: raw audio -> two ONNX backbones -> two ONNX CTC
    heads -> shared ctc_predict -> mean of the two probability vectors."""

    def __init__(self, out_dir: Path):
        import onnxruntime as ort

        self.sessions = {
            name: ort.InferenceSession(str(out_dir / f"{name}.onnx"),
                                       providers=["CPUExecutionProvider"])
            for name in ["wav2vec2_backbone", "wavlm_backbone",
                         "wav2vec2_ctc_head", "wavlm_ctc_head"]
        }

    def _branch(self, backbone: str, head: str, audio: np.ndarray,
                timings: dict = None) -> np.ndarray:
        t0 = time.perf_counter()
        emb = self.sessions[backbone].run(None, {"audio": audio})[0]
        t1 = time.perf_counter()
        emb = emb[:, :MAX_EMBED_LENGTH, :]
        log_probs = self.sessions[head].run(None, {"embeddings": emb})[0]
        t2 = time.perf_counter()
        _, p, _ = ctc_predict(log_probs[0])
        if timings is not None:
            timings[backbone] = timings.get(backbone, 0.0) + (t1 - t0)
            timings[head] = timings.get(head, 0.0) + (t2 - t1)
        return p

    BRANCHES = [("wav2vec2_backbone", "wav2vec2_ctc_head"),
                ("wavlm_backbone", "wavlm_ctc_head")]

    def score(self, audio: np.ndarray, timings: dict = None) -> np.ndarray:
        audio = audio.astype(np.float32)[None, :]  # [1, samples]
        probs = [self._branch(b, h, audio, timings) for b, h in self.BRANCHES]
        return (probs[0] + probs[1]) / 2.0

    def score_parallel(self, audio: np.ndarray, pool) -> np.ndarray:
        """The two branches are independent - running them on two threads
        (ORT releases the GIL) is the deployment-relevant latency shape."""
        audio = audio.astype(np.float32)[None, :]
        futures = [pool.submit(self._branch, b, h, audio) for b, h in self.BRANCHES]
        return (futures[0].result() + futures[1].result()) / 2.0


class TorchFusedScorer:
    """PyTorch reference chain, identical to the pairing eval / live_mic_smoke
    code path - the ground truth the ONNX chain must match."""

    def __init__(self, num_classes: int):
        import torch

        from evaluation.harness.embeddings_cache import _load_backbone
        from workflows.ctc_w2v2_workflow.models.ctc_model import create_ctc_model

        self.torch = torch
        self.w2v2_processor, self.w2v2_model, _ = _load_backbone("wav2vec2")
        self.wavlm_processor, self.wavlm_model, _ = _load_backbone("wavlm")
        self.heads = {}
        for base, ckpt_path in [("wav2vec2", W2V2_HEAD_CKPT), ("wavlm", WAVLM_HEAD_CKPT)]:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = create_ctc_model(num_classes=num_classes)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.heads[base] = model

    def score(self, audio: np.ndarray) -> np.ndarray:
        probs = []
        for base, processor, backbone in [
            ("wav2vec2", self.w2v2_processor, self.w2v2_model),
            ("wavlm", self.wavlm_processor, self.wavlm_model),
        ]:
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            with self.torch.no_grad():
                emb = backbone(**inputs).last_hidden_state
                emb = emb[:, :MAX_EMBED_LENGTH, :]
                log_probs, _ = self.heads[base](emb)
            _, p, _ = ctc_predict(log_probs.squeeze(0).numpy())
            probs.append(p)
        return (probs[0] + probs[1]) / 2.0


def session_wavs(session_dir: Path):
    return sorted(session_dir.glob("*/*.wav"))


def read_audio(wav_path: Path) -> np.ndarray:
    audio, sr = sf.read(wav_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{wav_path}: expected {SAMPLE_RATE}Hz, got {sr}Hz")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def run_parity_and_replay(onnx_scorer, canonical_labels, skip_replay: bool):
    """Numeric parity (ONNX vs PyTorch fused vectors) and, unless skipped,
    a full replay of both saved sessions through the ONNX chain."""
    torch_scorer = TorchFusedScorer(num_classes=len(canonical_labels))

    parity = {"clips_compared": 0, "max_abs_diff": 0.0, "top1_disagreements": 0}
    replay = {}
    for session_dir in SESSIONS:
        if not session_dir.exists():
            print(f"  WARNING: session {session_dir.name} not found - skipping")
            continue
        wavs = session_wavs(session_dir)
        # Parity on every clip is the point when validating a new export;
        # --skip-replay compares a 10-clip spot sample instead.
        parity_wavs = set(wavs if not skip_replay else wavs[::max(1, len(wavs) // 10)])
        loop_wavs = wavs if not skip_replay else sorted(parity_wavs)
        correct = top3 = 0
        print(f"  {session_dir.name}: {len(wavs)} clips "
              f"(parity on {len(parity_wavs)})")
        for i, wav_path in enumerate(loop_wavs):
            phoneme = wav_path.parent.name
            audio = read_audio(wav_path)
            fused_onnx = onnx_scorer.score(audio)
            if wav_path in parity_wavs:
                fused_torch = torch_scorer.score(audio)
                diff = float(np.abs(fused_onnx - fused_torch).max())
                parity["max_abs_diff"] = max(parity["max_abs_diff"], diff)
                parity["clips_compared"] += 1
                if int(fused_onnx.argmax()) != int(fused_torch.argmax()):
                    parity["top1_disagreements"] += 1
                    print(f"    PARITY TOP-1 MISMATCH on {wav_path.name} (diff={diff:.6f})")
            if skip_replay:
                continue
            ranking = np.argsort(fused_onnx)[::-1]
            true_idx = canonical_labels.index(phoneme)
            correct += int(ranking[0] == true_idx)
            top3 += int(true_idx in ranking[:3].tolist())
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(wavs)} scored")
        if not skip_replay and wavs:
            replay[session_dir.name] = {
                "num_clips": len(wavs),
                "fused_top1": correct / len(wavs),
                "fused_top3_known_target": top3 / len(wavs),
                "expected_from_torch_pairing_eval": EXPECTED_FUSED.get(session_dir.name),
            }
            print(f"    fused ONNX: top-1 {correct / len(wavs):.2%}, "
                  f"top-3 {top3 / len(wavs):.2%} "
                  f"(torch pairing eval: {EXPECTED_FUSED.get(session_dir.name)})")
    return parity, replay


def run_latency(onnx_scorer):
    """Time the full fused chain (both backbones + both heads + decode) on
    CPU at several clip durations, against the ~150ms temporal-brain budget.
    Measured two ways: sequential (worst case) and the two branches on
    parallel threads (the deployment-relevant shape)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    results = {"cpu_count": os.cpu_count(), "budget_s": LATENCY_BUDGET_S, "per_duration": {}}
    rng = np.random.default_rng(42)
    pool = ThreadPoolExecutor(max_workers=2)
    for duration in [0.5, 1.0, 3.0]:
        audio = rng.standard_normal(int(duration * SAMPLE_RATE)).astype(np.float32) * 0.05
        entry = {}
        for mode in ["sequential", "parallel"]:
            call = (onnx_scorer.score if mode == "sequential"
                    else lambda a: onnx_scorer.score_parallel(a, pool))
            for _ in range(3):  # warmup
                call(audio)
            times, components = [], {}
            for _ in range(15):
                if mode == "sequential":
                    t0 = time.perf_counter()
                    onnx_scorer.score(audio, timings=components)
                    times.append(time.perf_counter() - t0)
                else:
                    t0 = time.perf_counter()
                    call(audio)
                    times.append(time.perf_counter() - t0)
            times = np.array(times)
            entry[mode] = {
                "mean_s": float(times.mean()),
                "p50_s": float(np.percentile(times, 50)),
                "p95_s": float(np.percentile(times, 95)),
                "within_budget": bool(times.mean() <= LATENCY_BUDGET_S),
            }
            if mode == "sequential":
                entry[mode]["component_mean_s"] = {k: v / len(times) for k, v in components.items()}
            print(f"  {duration}s clip [{mode}]: mean {times.mean() * 1000:.0f}ms, "
                  f"p95 {np.percentile(times, 95) * 1000:.0f}ms "
                  f"({'WITHIN' if times.mean() <= LATENCY_BUDGET_S else 'OVER'} "
                  f"{LATENCY_BUDGET_S * 1000:.0f}ms budget)")
        results["per_duration"][f"{duration}s"] = entry
    pool.shutdown()
    return results


def main():
    parser = argparse.ArgumentParser(description="Export + validate the fused pair as ONNX")
    parser.add_argument("--skip-replay", action="store_true",
                        help="Skip the full 222-clip replay; spot-check parity only")
    args = parser.parse_args()

    canonical_labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    # Both source checkpoints must agree on the label order - the fused
    # average is meaningless otherwise.
    import torch
    for ckpt_path in [W2V2_HEAD_CKPT, WAVLM_HEAD_CKPT]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt["canonical_labels"] == canonical_labels, f"label mismatch: {ckpt_path}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1/4: exporting backbones ===")
    export_backbones()
    print("=== 2/4: exporting CTC heads ===")
    export_ctc_heads(num_classes=len(canonical_labels))

    metadata = {
        "configuration": "fused pair (plans/channel-augmentation-experiment.md, 2026-07-13)",
        "fusion_rule": "mean of the two ctc_predict probability vectors",
        "pipeline": [
            "audio 16kHz mono float32 -> wav2vec2_backbone.onnx -> [1,T,768]"
            " (normalization baked into graph)",
            "audio 16kHz mono float32 -> wavlm_backbone.onnx -> [1,T,768] (raw audio in)",
            f"embeddings truncated to {MAX_EMBED_LENGTH} frames",
            "embeddings -> *_ctc_head.onnx -> log_probs [1,T,38]",
            "workflows/shared/ctc_decode.ctc_predict per head, then average",
        ],
        "canonical_labels": canonical_labels,
        "blank_token_index": len(canonical_labels),
        "source_checkpoints": {
            "wav2vec2_ctc_head": W2V2_HEAD_CKPT.relative_to(REPO_ROOT).as_posix(),
            "wavlm_ctc_head": WAVLM_HEAD_CKPT.relative_to(REPO_ROOT).as_posix(),
        },
        "opset": OPSET,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT_DIR / "fused_pair_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    onnx_scorer = OnnxFusedScorer(OUT_DIR)

    print("=== 3/4: parity vs PyTorch" + (" + full session replay ===" if not args.skip_replay else " (spot check) ==="))
    parity, replay = run_parity_and_replay(onnx_scorer, canonical_labels, args.skip_replay)

    print("=== 4/4: latency benchmark ===")
    latency = run_latency(onnx_scorer)

    report = {"parity": parity, "replay": replay, "latency": latency,
              "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT_DIR / "fused_pair_onnx_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'fused_pair_onnx_report.json'}")

    if parity["top1_disagreements"] > 0:
        print(f"FAIL: {parity['top1_disagreements']} top-1 disagreements vs PyTorch")
        return 1
    print(f"Parity: {parity['clips_compared']} clips, "
          f"max abs prob diff {parity['max_abs_diff']:.2e}, 0 top-1 disagreements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
