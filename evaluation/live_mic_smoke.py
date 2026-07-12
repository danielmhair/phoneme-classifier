"""
Live-mic smoke test (PRD §5.4): record a small FRESH set of clips (new
session, new mic conditions - not from the existing recordings/ corpus) and
score them against all three full-corpus models. This is the tripwire
against a repeat of the historical 82%-test-to-18%-real-world collapse.

Prerequisite: train the models first (uses the harness embedding caches, so
it's fast if a LOSO run has already populated them):

    poetry run python -m evaluation.harness.full_train

Then record a session (defaults: 3 clips per phoneme, all 37 phonemes,
3.0s per clip - the same raw, untrimmed, unnormalized 16kHz format the
training corpus was recorded with):

    poetry run python evaluation/live_mic_smoke.py --speaker dan
    poetry run python evaluation/live_mic_smoke.py --speaker dan --phonemes b ch sh --clips 5

Clips are saved under evaluation/live_mic_sessions/<session>/<phoneme>/ and
every clip is scored from its saved wav file (sf.read), so the audio path is
byte-identical to how training data is read. A saved session can be
re-scored later (e.g. after retraining) without re-recording:

    poetry run python evaluation/live_mic_smoke.py --replay evaluation/live_mic_sessions/<session>

Outputs per session: predictions.csv (per-clip, per-model top-1 / known-target
rank / margin, same metrics as the LOSO harness) and summary.json.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Phoneme names/guides contain IPA characters; don't depend on PYTHONUTF8=1
# being set for an interactive tool (Windows consoles default to cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from evaluation.harness.embeddings_cache import _load_backbone  # noqa: E402
from workflows.shared.ctc_decode import ctc_predict  # noqa: E402

DEFAULT_MODELS_DIR = REPO_ROOT / "evaluation" / "full_models"
DEFAULT_SESSIONS_DIR = REPO_ROOT / "evaluation" / "live_mic_sessions"
SAMPLE_RATE = 16000
DEFAULT_DURATION = 3.0  # seconds - matches how the training corpus was recorded
MAX_EMBED_LENGTH = 1000  # frames - matches evaluation/harness/embeddings_cache.py

# LOSO no-augmentation baseline (Chloe fold), for context in the summary.
LOSO_CHLOE_BASELINE = {"mlp_control": 0.4507, "wav2vec2_ctc": 0.5869, "wavlm_ctc": 0.5790}


class LiveMicScorer:
    """All three full-corpus models + both embedding backbones, scoring
    saved wav files exactly the way the harness scores corpus files."""

    def __init__(self, models_dir: Path):
        import torch
        from workflows.ctc_w2v2_workflow.models.ctc_model import create_ctc_model

        self.torch = torch
        self.canonical_labels = json.loads(
            (models_dir / "canonical_labels.json").read_text(encoding="utf-8")
        )

        with open(models_dir / "mlp_control.pkl", "rb") as f:
            mlp_bundle = pickle.load(f)
        assert mlp_bundle["canonical_labels"] == self.canonical_labels
        self.mlp = mlp_bundle["clf"]

        self.ctc_models = {}
        for model_type in ["wav2vec2_ctc", "wavlm_ctc"]:
            ckpt = torch.load(models_dir / f"{model_type}.pt", map_location="cpu")
            assert ckpt["canonical_labels"] == self.canonical_labels
            model = create_ctc_model(num_classes=len(self.canonical_labels))
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.ctc_models[model_type] = model

        print("Loading Wav2Vec2 backbone...")
        self.w2v2_processor, self.w2v2_model, _ = _load_backbone("wav2vec2")
        print("Loading WavLM backbone...")
        self.wavlm_processor, self.wavlm_model, _ = _load_backbone("wavlm")

    def _temporal_embedding(self, audio: np.ndarray, base_model: str) -> np.ndarray:
        processor, model = {
            "wav2vec2": (self.w2v2_processor, self.w2v2_model),
            "wavlm": (self.wavlm_processor, self.wavlm_model),
        }[base_model]
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0).numpy()  # [T, 768]
        return emb[:MAX_EMBED_LENGTH]

    def score_file(self, wav_path: Path) -> dict:
        """Return {model_type: score_vector aligned to canonical_labels}."""
        audio, sr = sf.read(wav_path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{wav_path}: expected {SAMPLE_RATE}Hz, got {sr}Hz")
        if audio.ndim > 1:
            audio = audio[:, 0]

        w2v2_emb = self._temporal_embedding(audio, "wav2vec2")
        wavlm_emb = self._temporal_embedding(audio, "wavlm")

        scores = {"mlp_control": self.mlp.predict_proba(w2v2_emb.mean(axis=0, keepdims=True))[0]}
        for model_type, emb in [("wav2vec2_ctc", w2v2_emb), ("wavlm_ctc", wavlm_emb)]:
            x = self.torch.FloatTensor(emb).unsqueeze(0)
            with self.torch.no_grad():
                log_probs, _ = self.ctc_models[model_type](x)
            _, probabilities, _ = ctc_predict(log_probs.squeeze(0).numpy())
            scores[model_type] = probabilities
        return scores


def score_row(scores: np.ndarray, true_phoneme: str, canonical_labels: list) -> dict:
    true_idx = canonical_labels.index(true_phoneme)
    ranking = np.argsort(scores)[::-1]
    pred_idx = int(ranking[0])
    true_rank = int(np.where(ranking == true_idx)[0][0]) + 1  # 1-based
    rival_idx = pred_idx if pred_idx != true_idx else int(ranking[1])
    return {
        "predicted_phoneme": canonical_labels[pred_idx],
        "correct": pred_idx == true_idx,
        "true_target_rank": true_rank,
        "margin_vs_rival": float(scores[true_idx] - scores[rival_idx]),
    }


def record_clip(duration: float) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def pronunciation_guides() -> dict:
    from workflows.mlp_control_workflow.validations.record_phonemes_cli import char_to_phoneme
    return char_to_phoneme


def evaluate_clip(scorer, wav_path, phoneme, rows, session_name):
    all_scores = scorer.score_file(wav_path)
    try:
        file_ref = str(wav_path.relative_to(REPO_ROOT))
    except ValueError:  # replay session outside the repo
        file_ref = str(wav_path)
    parts = []
    for model_type, scores in all_scores.items():
        result = score_row(scores, phoneme, scorer.canonical_labels)
        rows.append({
            "session": session_name,
            "model": model_type,
            "file": file_ref,
            "true_phoneme": phoneme,
            **result,
        })
        mark = "+" if result["correct"] else " "
        parts.append(f"{model_type}: {result['predicted_phoneme']:>9} "
                     f"[{mark}] rank={result['true_target_rank']}")
    print("    " + " | ".join(parts))


def write_outputs(rows, session_dir, session_name):
    if not rows:
        print("No clips scored - nothing to write.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(session_dir / "predictions.csv", index=False)

    summary = {"session": session_name, "num_clips": int(df["file"].nunique()), "per_model": {}}
    print(f"\n=== Session summary: {summary['num_clips']} clips ===")
    for model_type, group in df.groupby("model"):
        top1 = float(group["correct"].mean())
        top3 = float((group["true_target_rank"] <= 3).mean())
        baseline = LOSO_CHLOE_BASELINE.get(model_type)
        summary["per_model"][model_type] = {
            "top1_accuracy": top1,
            "top3_known_target": top3,
            "mean_true_target_rank": float(group["true_target_rank"].mean()),
            "loso_chloe_baseline_top1": baseline,
        }
        print(f"{model_type}: top-1 {top1:.2%}, top-3 {top3:.2%} "
              f"(LOSO Chloe baseline top-1: {baseline:.2%})")
    (session_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {session_dir / 'predictions.csv'} and summary.json")
    print("Context: a live-mic top-1 far below the LOSO baseline on a known adult "
          "speaker suggests a recording-conditions / pipeline gap (the historical "
          "82%->18% collapse pattern), not a model-quality finding by itself.")


def run_replay(scorer, session_dir: Path):
    session_name = session_dir.name
    rows = []
    wavs = sorted(session_dir.glob("*/*.wav"))
    if not wavs:
        print(f"No wavs found under {session_dir}/<phoneme>/*.wav")
        return
    print(f"Re-scoring {len(wavs)} clips from {session_dir}")
    for wav_path in wavs:
        phoneme = wav_path.parent.name
        if phoneme not in scorer.canonical_labels:
            print(f"  skipping {wav_path} - directory {phoneme!r} is not a canonical phoneme")
            continue
        print(f"  {wav_path.name} (target: {phoneme})")
        evaluate_clip(scorer, wav_path, phoneme, rows, session_name)
    write_outputs(rows, session_dir, session_name)


def run_recording(scorer, speaker: str, phonemes, clips_per_phoneme: int, duration: float,
                  sessions_dir: Path):
    session_name = f"{speaker}_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir = sessions_dir / session_name
    guides = pronunciation_guides()
    rows = []

    print(f"\n=== Live-mic smoke test session: {session_name} ===")
    print(f"{len(phonemes)} phonemes x {clips_per_phoneme} clips, {duration}s each.")
    print("Per phoneme: Enter = record, 's' = skip phoneme, 'q' = quit and summarize.\n")

    try:
        for phoneme in phonemes:
            ipa, guide = guides.get(phoneme, (phoneme, "(no pronunciation guide)"))
            print(f"--- Phoneme {phoneme!r} ({ipa}): {guide}")
            for take in range(1, clips_per_phoneme + 1):
                cmd = input(f"  [{phoneme} {take}/{clips_per_phoneme}] Enter=record, s=skip, q=quit > ").strip().lower()
                if cmd == "q":
                    raise KeyboardInterrupt
                if cmd == "s":
                    break
                print(f"  Recording {duration}s - speak now...")
                audio = record_clip(duration)
                out_dir = session_dir / phoneme
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                wav_path = out_dir / f"{speaker}_{phoneme}_ep-{ipa}_{timestamp}_{take}.wav"
                sf.write(wav_path, audio, SAMPLE_RATE)
                evaluate_clip(scorer, wav_path, phoneme, rows, session_name)
    except KeyboardInterrupt:
        print("\nStopping early - summarizing what was recorded.")

    write_outputs(rows, session_dir, session_name)


def main():
    parser = argparse.ArgumentParser(description="Live-mic smoke test across all three models (PRD 5.4)")
    parser.add_argument("--speaker", type=str, help="Name for this recording session (e.g. dan)")
    parser.add_argument("--phonemes", nargs="+", default=None, help="Subset of phonemes (default: all 37)")
    parser.add_argument("--clips", type=int, default=3, help="Clips per phoneme (default 3)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Seconds per clip (default 3.0)")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--replay", type=Path, help="Re-score an existing session directory instead of recording")
    args = parser.parse_args()

    if not args.replay and not args.speaker:
        parser.error("--speaker is required for a recording session (or use --replay <dir>)")

    if not (args.models_dir / "canonical_labels.json").exists():
        print(f"No trained models found in {args.models_dir}.")
        print("Train them first:  poetry run python -m evaluation.harness.full_train")
        return 1

    print("Loading models...")
    scorer = LiveMicScorer(args.models_dir)

    if args.replay:
        run_replay(scorer, args.replay)
        return 0

    phonemes = args.phonemes or scorer.canonical_labels
    unknown = [p for p in phonemes if p not in scorer.canonical_labels]
    if unknown:
        parser.error(f"Unknown phonemes {unknown}; canonical: {scorer.canonical_labels}")

    run_recording(scorer, args.speaker, phonemes, args.clips, args.duration, DEFAULT_SESSIONS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
