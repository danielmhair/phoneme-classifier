"""
Training-only audio augmentation for the LOSO harness.

Reuses the exact transforms from workflows/shared/s0b_augment_audio.py
(speed 0.9x/1.1x, pitch +-2 semitones, noise overlay) so the harness matches
what production training used, instead of reimplementing augmentation logic
separately.

Augmented variants are generated once per source recording - fold-independent,
like the raw embedding cache - and written to disk under
evaluation/harness_cache/augmented_recordings/<speaker>/<phoneme>/. Because
each variant is derived only from its own speaker's audio, excluding a
held-out speaker's rows from a LOSO fold's training set automatically
excludes their augmented variants too - no cross-speaker leakage risk.

Augmented rows must ONLY ever be added to a fold's training set, never its
test set - LOSO evaluation must always score real, unaugmented held-out
audio. Enforced by the caller (loso_runner.py), not by this module.

Noise-application decisions in production (s0b_augment_audio.py) are
stochastic (random.random() < pct). Here they're seeded per-file
(random.Random(file_id)) so extraction is deterministic and resumable -
re-running produces the same variant set instead of re-rolling the dice.
"""
import json
from pathlib import Path
from random import Random
from typing import Dict, List

import librosa
import pandas as pd
import soundfile as sf
from pydub import AudioSegment

from evaluation.harness.embeddings_cache import atomic_write_text
from workflows.shared.s0b_augment_audio import change_pitch, change_speed, overlay_noise

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUG_DIR = REPO_ROOT / "evaluation" / "harness_cache" / "augmented_recordings"
DEFAULT_NOISE_PATH = REPO_ROOT / "recordings" / "silence.wav"


def build_augmented_manifest(
    manifest: pd.DataFrame,
    aug_dir: Path = DEFAULT_AUG_DIR,
    noise_path: Path = DEFAULT_NOISE_PATH,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Generate (or reuse cached) augmented variants for every row in manifest.

    Returns a DataFrame with the same columns as `manifest` (speaker,
    phoneme, filepath, file_id) covering only the augmented variants -
    concatenate with the original manifest to get a fold's full training set
    (after filtering both to the fold's training speakers).
    """
    aug_dir.mkdir(parents=True, exist_ok=True)
    index_path = aug_dir / "index.json"
    index: Dict[str, List[str]] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())

    noise_audio = None
    to_process = [row for row in manifest.itertuples() if row.file_id not in index]
    if to_process:
        print(f"[augment] generating variants for {len(to_process)} files "
              f"({len(manifest) - len(to_process)} already cached)")
        noise_audio = AudioSegment.from_wav(noise_path)

    for i, row in enumerate(to_process):
        if i % progress_every == 0:
            print(f"[augment] {i}/{len(to_process)}")

        out_dir = aug_dir / row.speaker / row.phoneme
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(row.filepath).stem
        rng = Random(row.file_id)  # deterministic per-file, for resumability

        variant_paths = _generate_variants(row.filepath, out_dir, base_name, noise_audio, rng)
        index[row.file_id] = [str(p) for p in variant_paths]

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


def _generate_variants(filepath: str, out_dir: Path, base_name: str, noise_audio, rng: Random) -> List[Path]:
    sound = AudioSegment.from_wav(filepath)
    y, sr = librosa.load(filepath, sr=None)
    paths: List[Path] = []

    if rng.random() < 0.4:
        noisy = overlay_noise(sound, noise_audio, noise_reduction_db=rng.randint(10, 25))
        p = out_dir / f"{base_name}_orig_noisy.wav"
        noisy.export(p, format="wav")
        paths.append(p)

    for label, factor in [("speed_0.9x", 0.9), ("speed_1.1x", 1.1)]:
        altered = change_speed(sound, factor)
        p = out_dir / f"{base_name}_{label}.wav"
        altered.export(p, format="wav")
        paths.append(p)
        if rng.random() < 0.2:
            noisy = overlay_noise(altered, noise_audio, noise_reduction_db=rng.randint(10, 25))
            p2 = out_dir / f"{base_name}_{label}_noisy.wav"
            noisy.export(p2, format="wav")
            paths.append(p2)

    for label, steps in [("pitch_up", 2), ("pitch_down", -2)]:
        y_shifted = change_pitch(y, sr, n_steps=steps)
        p = out_dir / f"{base_name}_{label}.wav"
        sf.write(p, y_shifted, sr)
        paths.append(p)
        if rng.random() < 0.2:
            pitch_seg = AudioSegment.from_wav(p)
            noisy = overlay_noise(pitch_seg, noise_audio, noise_reduction_db=rng.randint(10, 25))
            p2 = out_dir / f"{base_name}_{label}_noisy.wav"
            noisy.export(p2, format="wav")
            paths.append(p2)

    return paths
