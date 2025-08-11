"""
Streaming Wav2Vec2 -> CTC spike.
Goal: minimal end-to-end pipeline for Epic 1 (Live Phoneme CTCs).

Features:
- Chunked streaming from a wav file (mic optional placeholder).
- Wav2Vec2 backbone + randomly initialized linear CTC head (logits over phoneme vocab + blank).
- Greedy CTC decoding with repeat + blank collapse.
- Stabilization heuristic: emit phoneme only if same symbol appears in consecutive decoded tail over N chunks.
- Latency + real-time factor metrics.
- Offline (full file) pass for baseline comparison (sequence difference ratio).

NOT production ready; this is a spike for latency + architectural validation.
"""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import json  # added for JSON vocab support

# --------- Config Defaults ---------
DEFAULT_SAMPLE_RATE = 16_000
CHUNK_MS = 40          # frame window size in ms
CHUNK_STRIDE_MS = 20   # hop (overlap 50%)
STABILIZATION_MIN_REPEATS = 2  # how many consecutive chunks must agree before emitting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Vocab Handling ---------


def load_phoneme_vocab(vocab_path: Path) -> List[str]:
    """Load phoneme symbols.

    Supports two formats:
    1. Plain text: one symbol per non-empty line.
    2. JSON file: a JSON array of symbols (e.g. phoneme_labels.json).
    """
    if not vocab_path.exists():
        raise FileNotFoundError(f"Phoneme vocab not found at {vocab_path}")
    text = vocab_path.read_text(encoding="utf-8").strip()
    symbols: List[str] = []
    if vocab_path.suffix.lower() == ".json" or text.startswith("["):
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("JSON vocab must be a list of symbols")
            for item in data:
                if not isinstance(item, str):
                    raise ValueError("All JSON vocab entries must be strings")
                sym = item.strip()
                if sym:
                    symbols.append(sym)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON vocab {vocab_path}: {e}") from e
    else:
        for line in text.splitlines():
            t = line.strip()
            if not t:
                continue
            symbols.append(t)
    if not symbols:
        raise ValueError(f"No symbols loaded from vocab {vocab_path}")
    return symbols

# --------- Model Wrapper ---------


@dataclass
class StreamingCTCModel:
    processor: Wav2Vec2Processor
    backbone: Wav2Vec2Model
    ctc_head: torch.nn.Module
    blank_id: int
    id2sym: List[str]

    @torch.inference_mode()
    def forward_logits(self, audio_tensor: torch.Tensor, sr: int) -> torch.Tensor:
        """audio_tensor: (samples,) float32"""
        inputs = self.processor(audio_tensor, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        hidden = self.backbone(**inputs).last_hidden_state  # (B, T, H)
        logits = self.ctc_head(hidden)  # (B, T, V)
        return logits[0]  # (T, V)

    @torch.inference_mode()
    def greedy_decode_chunk(self, logits: torch.Tensor) -> List[int]:
        # logits: (T, V)
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        # CTC collapse within chunk only (we'll stabilize across chunks separately)
        collapsed = []
        prev = None
        for pid in pred_ids:
            if pid == self.blank_id:
                prev = None
                continue
            if pid != prev:
                collapsed.append(pid)
                prev = pid
        return collapsed

# --------- Streaming Logic ---------


def chunk_audio(audio: torch.Tensor, sr: int, chunk_ms: int, stride_ms: int):
    chunk_size = int(sr * chunk_ms / 1000)
    stride_size = int(sr * stride_ms / 1000)
    if stride_size <= 0:
        raise ValueError("stride_ms too small")
    start = 0
    total = audio.shape[0]
    while start < total:
        end = min(start + chunk_size, total)
        yield audio[start:end], start, end
        if end == total:
            break
        start += stride_size

# --------- Metrics Helpers ---------


@dataclass
class Metrics:
    chunk_times: List[float]

    def add(self, dt: float):
        self.chunk_times.append(dt)

    def summary(self, total_audio_s: float) -> str:
        if not self.chunk_times:
            return "No chunks processed"
        avg = sum(self.chunk_times) / len(self.chunk_times)
        mx = max(self.chunk_times)
        rtf = sum(self.chunk_times) / total_audio_s
        return f"Chunks={len(self.chunk_times)} avg={avg*1000:.1f}ms max={mx*1000:.1f}ms RTF={rtf:.3f}"

# --------- Offline Reference Decode (naive) ---------


def offline_decode(model: StreamingCTCModel, audio: torch.Tensor, sr: int) -> List[str]:
    logits = model.forward_logits(audio, sr)  # (T, V)
    ids = model.greedy_decode_chunk(logits)  # reuse collapse
    return [model.id2sym[i] for i in ids]

# --------- Spike Main ---------


def run_spike(args):
    vocab_symbols = load_phoneme_vocab(Path(args.phoneme_vocab) or "dist/phoneme_labels.json")
    # Insert blank at index 0
    id2sym = ["<blank>"] + vocab_symbols
    # sym2id could be used later for mapping back; kept for potential future use
    blank_id = 0

    print(f"Loaded {len(vocab_symbols)} phoneme symbols (+ blank) => V={len(id2sym)}")

    print("Loading wav2vec2 backbone…")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")  # type: ignore
    backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")  # type: ignore
    backbone = backbone.to(DEVICE)  # type: ignore
    backbone.eval()
    hidden_size = backbone.config.hidden_size
    vocab_size = len(id2sym)
    ctc_head = torch.nn.Linear(hidden_size, vocab_size).to(DEVICE)
    # No training in spike: randomly initialized

    model = StreamingCTCModel(processor, backbone, ctc_head, blank_id, id2sym)  # type: ignore

    wav_path = Path(args.wav)
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != DEFAULT_SAMPLE_RATE:
        print(f"Resampling not implemented in spike; expected {DEFAULT_SAMPLE_RATE}, got {sr}")
        return
    audio_t = torch.tensor(audio, dtype=torch.float32)

    total_audio_s = len(audio_t)/sr
    print(f"Audio {wav_path.name}: {total_audio_s:.2f}s @ {sr}Hz")

    # Warmup (excluded from metrics)
    if args.warmup:
        warmup_samples = int(sr * args.chunk_ms / 1000)
        dummy = torch.zeros(warmup_samples, dtype=torch.float32)
        t0_w = time.time()
        _ = model.forward_logits(dummy, sr)
        warmup_dt = time.time() - t0_w
        print(f"Warmup forward completed in {warmup_dt*1000:.1f} ms (excluded from metrics)")

    metrics = Metrics([])
    emitted: List[str] = []
    pending_symbol: Optional[str] = None
    pending_count = 0

    # Overall start time could be used for additional aggregate metrics later.
    for chunk, s_idx, e_idx in chunk_audio(audio_t, sr, args.chunk_ms, args.stride_ms):
        t0 = time.time()
        logits = model.forward_logits(chunk, sr)
        ids = model.greedy_decode_chunk(logits)
        # Simple stabilization: look at last symbol if present
        if ids:
            last_id = ids[-1]
            if last_id == blank_id:
                pass
            else:
                symbol = model.id2sym[last_id]
                if symbol == pending_symbol:
                    pending_count += 1
                else:
                    pending_symbol = symbol
                    pending_count = 1
                if pending_count >= args.stabilization_repeats:
                    emitted.append(symbol)
                    print(f"[EMIT] {symbol} (t={e_idx/sr:.2f}s)")
                    pending_symbol = None
                    pending_count = 0
        dt = time.time() - t0
        metrics.add(dt)
    # total_rt can be used later for overall RTF if desired
    # total runtime could be used for overall RTF vs full audio duration
    # (currently relying on per-chunk metrics summary)

    streamed_seq = ''.join(emitted)
    print("Streamed sequence (greedy, stabilized):", streamed_seq)
    print("Latency summary:", metrics.summary(total_audio_s))
    if args.warmup:
        print("(Warmup forward excluded from metrics above)")
    # Steady-state metrics (exclude first chunk timing) if more than one chunk
    if len(metrics.chunk_times) > 1:
        steady = metrics.chunk_times[1:]
        steady_avg = sum(steady)/len(steady)
        steady_rtf = sum(steady)/total_audio_s
        print(f"Steady-state (excl first chunk): avg={steady_avg*1000:.1f}ms RTF={steady_rtf:.3f}")

    print("\nOffline reference decode… (NOTE: random head => meaningless symbols, used only for shape/latency in spike)")
    offline_symbols = offline_decode(model, audio_t, sr)
    offline_seq = ''.join(offline_symbols)
    # Difference ratio: naive coverage of streamed in offline
    coverage = 0.0
    if offline_seq:
        coverage = sum(c in offline_seq for c in streamed_seq) / max(1, len(streamed_seq))
    print(f"Offline length={len(offline_seq)} Stream length={len(streamed_seq)} Coverage~{coverage:.2f}")

    print("\nNEXT STEPS TODO (annotated):")
    print("- Replace random CTC head with trained phoneme classifier or fine-tuned head.")
    print("- Implement proper resampling & mic input (sounddevice stream).")
    print("- Add beam search / LM integration.")
    print("- Add confidence scores & smoothing window across chunk boundaries.")
    print("- Timestamp assignment per phoneme (align center of stable region).")
    print("- Evaluate RTF vs. target budgets; profile GPU vs CPU performance.")


def parse_args():
    ap = argparse.ArgumentParser(description="Streaming wav2vec2 CTC spike")
    ap.add_argument("--wav", type=str, required=True, help="Path to wav file for streaming simulation")
    ap.add_argument("--phoneme_vocab", type=str, default="tokenizer/phoneme_vocab.txt")
    ap.add_argument("--chunk_ms", type=int, default=CHUNK_MS)
    ap.add_argument("--stride_ms", type=int, default=CHUNK_STRIDE_MS)
    ap.add_argument("--stabilization_repeats", type=int, default=STABILIZATION_MIN_REPEATS)
    ap.add_argument("--warmup", action="store_true", help="Run a dummy forward pass before timing (excluded from metrics)")
    return ap.parse_args()

 
if __name__ == "__main__":
    run_spike(parse_args())
