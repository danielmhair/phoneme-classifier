"""
D6 (data-collection-game PRD 8): holdout-children split alongside the
existing per-speaker LOSO rotation.

Collected children are exported into recordings/<child_code>/ by
game/export_tool/export_verified.py, each with a child_meta.json carrying
age band and the holdout flag assigned at signup (stratified per age band,
~20%). Speakers WITHOUT a child_meta.json are the original base corpus
(callie, cassie, chloe, dan, sky) and always train.

Holdout discipline: holdout children's clips exist in recordings/ but must
never enter any training fold - the split here is by the signup-time flag,
not by anything derived from the data.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from evaluation.harness.dataset import DEFAULT_RECORDINGS_DIR


def load_child_meta(recordings_dir: Path = DEFAULT_RECORDINGS_DIR) -> Dict[str, dict]:
    """speaker -> child_meta.json contents, for collected children only."""
    meta: Dict[str, dict] = {}
    for speaker_dir in sorted(recordings_dir.iterdir()):
        meta_path = speaker_dir / "child_meta.json"
        if speaker_dir.is_dir() and meta_path.exists():
            meta[speaker_dir.name] = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta


def split_by_holdout(
    manifest: pd.DataFrame,
    child_meta: Optional[Dict[str, dict]] = None,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(train_manifest, holdout_manifest).

    Train = base speakers (no metadata) + collected children with
    holdout=false. Holdout = collected children flagged holdout=true at
    signup.
    """
    if child_meta is None:
        child_meta = load_child_meta(recordings_dir)
    holdout_speakers = {s for s, m in child_meta.items() if m.get("holdout")}
    is_holdout = manifest["speaker"].isin(holdout_speakers)
    return (
        manifest[~is_holdout].reset_index(drop=True),
        manifest[is_holdout].reset_index(drop=True),
    )


def training_children_in_ingest_order(child_meta: Dict[str, dict]) -> List[str]:
    """Non-holdout collected children, ordered by first session date (then
    child_code) - the N axis of the learning curve."""
    train_children = [(s, m) for s, m in child_meta.items() if not m.get("holdout")]

    def sort_key(item):
        speaker, meta = item
        dates = meta.get("session_dates") or []
        return (min(dates) if dates else "9999-99-99", speaker)

    return [s for s, _ in sorted(train_children, key=sort_key)]


def age_band_of(speaker: str, child_meta: Dict[str, dict]) -> Optional[int]:
    meta = child_meta.get(speaker)
    return meta.get("age_band") if meta else None
