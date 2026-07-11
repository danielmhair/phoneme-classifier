"""
Speaker-aware dataset manifest, built directly from recordings/<speaker>/<phoneme>/*.wav.

Deliberately reads from recordings/ rather than dist/organized_recordings/ -
the organized/augmented directories are phoneme-first (see
workflows/shared/s1_prepare_wav_files.py), which loses the speaker directory
boundary needed for leave-one-speaker-out splitting. recordings/ is
speaker-first already, so folds can be built directly from it with no
filename-prefix filtering required.

This harness intentionally does not include the noise/speed/pitch
augmentation step (workflows/shared/s0b_augment_audio.py) - see PRD
"Explicit Non-Decisions" / implementation notes. Training here uses the
original recordings only, for this first pass.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDINGS_DIR = REPO_ROOT / "recordings"


@dataclass(frozen=True)
class Recording:
    speaker: str
    phoneme: str
    filepath: str  # absolute path, as a string (stable dict/JSON key)
    file_id: str  # f"{speaker}/{phoneme}/{filename}" - stable across machines


def build_manifest(recordings_dir: Path = DEFAULT_RECORDINGS_DIR) -> pd.DataFrame:
    """Scan recordings/<speaker>/<phoneme>/*.wav into a flat manifest.

    Returns a DataFrame with columns: speaker, phoneme, filepath, file_id.
    Only real directories are scanned - recordings/evaluation1.json,
    recordings/evaluation1.md, recordings/silence.wav (loose files at the
    top level) are skipped since they aren't speaker directories.
    """
    rows: List[Recording] = []
    for speaker_dir in sorted(recordings_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        for phoneme_dir in sorted(speaker_dir.iterdir()):
            if not phoneme_dir.is_dir():
                continue
            phoneme = phoneme_dir.name
            for wav_path in sorted(phoneme_dir.glob("*.wav")):
                file_id = f"{speaker}/{phoneme}/{wav_path.name}"
                rows.append(Recording(speaker, phoneme, str(wav_path), file_id))

    if not rows:
        raise FileNotFoundError(f"No recordings found under {recordings_dir}")

    return pd.DataFrame(rows)


def canonical_phoneme_labels(manifest: pd.DataFrame) -> List[str]:
    """Sorted union of every phoneme seen across all speakers.

    Used as a fixed, canonical class-index space for every fold - the model
    fit on speakers {A,B,C,D} and evaluated on speaker E must use the exact
    same label->index mapping, even if E happens to be missing a class (or
    vice versa). This sidesteps the label-index-mismatch risk class
    entirely: there is one label list, computed once, never re-derived
    per-fold from whatever classes happen to appear in a given split.
    """
    return sorted(manifest["phoneme"].unique().tolist())


def get_speakers(manifest: pd.DataFrame) -> List[str]:
    return sorted(manifest["speaker"].unique().tolist())


def leave_one_speaker_out_folds(manifest: pd.DataFrame):
    """Yield (held_out_speaker, train_df, test_df) for every speaker."""
    for speaker in get_speakers(manifest):
        test_df = manifest[manifest["speaker"] == speaker].reset_index(drop=True)
        train_df = manifest[manifest["speaker"] != speaker].reset_index(drop=True)
        yield speaker, train_df, test_df


def speaker_summary(manifest: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Per-speaker file count and phoneme-class coverage, for sanity logging."""
    summary = {}
    for speaker, group in manifest.groupby("speaker"):
        summary[speaker] = {
            "num_files": len(group),
            "num_phoneme_classes": group["phoneme"].nunique(),
        }
    return summary
