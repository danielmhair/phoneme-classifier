"""
Channel/mic-simulation augmentation (training-only), for the
channel-augmentation experiment (plans/channel-augmentation-experiment.md).

Motivation: the 2026-07-12 live-mic smoke test measured a ~21-24 top-1 point
cost from mic quality alone, and none of the existing augmentation transforms
(speed/pitch/noise overlay) simulate channel effects. These transforms do:

  tel      - telephone-band bandpass (300-3400 Hz) + dynamic range compression
  cheapmic - lowpass at a random 3.5-6 kHz cutoff + dynamic range compression
  clip     - gain boost (+6..12 dB) into soft clipping (tanh saturation)
  8k       - 16k->8k->16k resample round-trip + small gain jitter

Same contract as augmentation.py: variants are generated once per source
recording, cached under evaluation/harness_cache/channel_augmented_recordings/
<speaker>/<phoneme>/, seeded per-file (deterministic, resumable), and must
only ever be added to training sets - never to test data.
"""
import json
from pathlib import Path
from random import Random
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from evaluation.harness.embeddings_cache import atomic_write_text

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANNEL_AUG_DIR = REPO_ROOT / "evaluation" / "harness_cache" / "channel_augmented_recordings"
SAMPLE_RATE = 16000


def _bandpass(audio: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return sosfilt(sos, audio)


def _lowpass(audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
    sos = butter(4, cutoff_hz, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    return sosfilt(sos, audio)


def _compress(audio: np.ndarray, threshold: float = 0.1, ratio: float = 4.0) -> np.ndarray:
    """Simple static dynamic-range compression on the sample envelope."""
    mag = np.abs(audio)
    over = mag > threshold
    compressed = np.where(over, threshold + (mag - threshold) / ratio, mag)
    out = np.sign(audio) * compressed
    peak = np.max(np.abs(out))
    if peak > 0:  # make up gain back to the original peak level
        out = out * (np.max(mag) / peak)
    return out


def _soft_clip(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20)
    return np.tanh(audio * gain)


def _resample_roundtrip(audio: np.ndarray, intermediate_hz: int = 8000) -> np.ndarray:
    down = resample_poly(audio, intermediate_hz, SAMPLE_RATE)
    return resample_poly(down, SAMPLE_RATE, intermediate_hz)[: len(audio)]


def _variants_for(audio: np.ndarray, rng: Random) -> Dict[str, np.ndarray]:
    return {
        "tel": _compress(_bandpass(audio, 300.0, 3400.0)),
        "cheapmic": _compress(_lowpass(audio, rng.uniform(3500.0, 6000.0))),
        "clip": _soft_clip(audio, gain_db=rng.uniform(6.0, 12.0)),
        "8k": _resample_roundtrip(audio) * 10 ** (rng.uniform(-3.0, 3.0) / 20),
    }


def build_channel_augmented_manifest(
    manifest: pd.DataFrame,
    aug_dir: Path = DEFAULT_CHANNEL_AUG_DIR,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Generate (or reuse cached) channel-augmented variants for every row.

    Returns a DataFrame with the same columns as `manifest` covering only the
    variants; caller concatenates with (a subset of) the original manifest.
    """
    aug_dir.mkdir(parents=True, exist_ok=True)
    index_path = aug_dir / "index.json"
    index: Dict[str, List[str]] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())

    to_process = [row for row in manifest.itertuples() if row.file_id not in index]
    if to_process:
        print(f"[channel-augment] generating variants for {len(to_process)} files "
              f"({len(manifest) - len(to_process)} already cached)")

    for i, row in enumerate(to_process):
        if i % progress_every == 0:
            print(f"[channel-augment] {i}/{len(to_process)}")

        audio, sr = sf.read(row.filepath)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row.filepath}: expected {SAMPLE_RATE}Hz, got {sr}Hz")
        if audio.ndim > 1:
            audio = audio[:, 0]

        out_dir = aug_dir / row.speaker / row.phoneme
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(row.filepath).stem
        rng = Random(f"channel:{row.file_id}")  # deterministic per-file

        variant_paths = []
        for label, variant in _variants_for(audio, rng).items():
            p = out_dir / f"{base_name}_ch_{label}.wav"
            sf.write(p, np.clip(variant, -1.0, 1.0).astype(np.float32), SAMPLE_RATE)
            variant_paths.append(str(p))
        index[row.file_id] = variant_paths

        if i % progress_every == 0:
            atomic_write_text(index_path, json.dumps(index, indent=2))

    if to_process:
        atomic_write_text(index_path, json.dumps(index, indent=2))

    rows = []
    for row in manifest.itertuples():
        for variant_path_str in index[row.file_id]:
            variant_path = Path(variant_path_str)
            rows.append({
                "speaker": row.speaker,
                "phoneme": row.phoneme,
                "filepath": str(variant_path),
                "file_id": f"{row.speaker}/{row.phoneme}/{variant_path.name}",
            })
    return pd.DataFrame(rows)
