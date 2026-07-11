"""
Leave-one-speaker-out fold orchestration: trains each model on 4 speakers,
evaluates on the 5th, rotates through all 5, and persists everything needed
to trust (or debug) the result.

Outputs, per run, into evaluation/loso_results/<run_id>/:
  - split_manifest.json   : held-out file_ids per fold (not just a random_state -
                             per PRD, "reproducible without re-running training
                             code unchanged")
  - predictions.csv        : one row per (fold, model, sample) with the known-target
                             rank + confidence-margin-vs-rival metric
  - summary.json            : per-(model, speaker) top-1 accuracy, headline Chloe
                             number, bug-floor flags

See evaluation/harness/models.py for the per-model fit/predict interface and
workflows/shared/ctc_decode.py for the CTC decode fix this harness depends on.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.harness.augmentation import build_augmented_manifest
from evaluation.harness.dataset import (
    build_manifest,
    canonical_phoneme_labels,
    leave_one_speaker_out_folds,
)
from evaluation.harness.embeddings_cache import extract_embeddings
from evaluation.harness.models import MODEL_BASE_EMBEDDING, build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation" / "loso_results"

# Below this, treat the number as a measurement-bug signal, not a verdict.
# 37 classes -> ~2.7% random chance; PRD sets the bug floor at ~10%.
BUG_FLOOR_ACCURACY = 0.10

# PRD ship bar, evaluated on Chloe specifically (the only target-age-band speaker).
SHIP_BAR_ACCURACY = 0.85

HEADLINE_SPEAKER = "chloe"

ALL_MODEL_TYPES = ["mlp_control", "wav2vec2_ctc", "wavlm_ctc"]


def run_loso(
    model_types: List[str] = ALL_MODEL_TYPES,
    manifest: Optional[pd.DataFrame] = None,
    ctc_epochs: int = 20,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    run_id: Optional[str] = None,
    use_augmentation: bool = False,
    only_folds: Optional[List[str]] = None,
) -> Dict:
    """
    only_folds: if given, only run LOSO folds for these held-out speakers
        (e.g. ["chloe"] for a fast headline-only pilot) - the full manifest
        is still used as the training pool for whichever folds do run, only
        the set of *held-out* speakers evaluated is restricted.
    use_augmentation: if True, add noise/speed/pitch-augmented variants
        (see augmentation.py) of each fold's training speakers to that
        fold's training set. Never applied to test data.
    """
    if manifest is None:
        manifest = build_manifest()
    canonical_labels = canonical_phoneme_labels(manifest)

    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== LOSO run {run_id} ===")
    print(f"Speakers: {sorted(manifest['speaker'].unique())}")
    print(f"Canonical phoneme classes ({len(canonical_labels)}): {canonical_labels}")
    print(f"Models: {model_types}")
    print(f"Augmentation: {use_augmentation}")
    if only_folds:
        print(f"Restricted to folds: {only_folds}")

    base_models_needed = sorted({MODEL_BASE_EMBEDDING[m] for m in model_types})
    embedding_indices = {bm: extract_embeddings(manifest, bm) for bm in base_models_needed}

    augmented_manifest = None
    if use_augmentation:
        # Only augment speakers that will actually be used as training data
        # for a fold we're running - skip augmenting a held-out-only speaker
        # (e.g. skip augmenting chloe entirely when only_folds=["chloe"],
        # since her data is never in a training set in that case).
        held_out_set = set(only_folds) if only_folds else set(manifest["speaker"].unique())
        speakers_needing_aug = set(manifest["speaker"].unique()) - (
            held_out_set if len(held_out_set) < len(manifest["speaker"].unique()) else set()
        )
        aug_source = manifest[manifest["speaker"].isin(speakers_needing_aug)] if speakers_needing_aug else manifest
        augmented_manifest = build_augmented_manifest(aug_source)
        print(f"Augmented manifest: {len(augmented_manifest)} synthetic training rows "
              f"from {sorted(aug_source['speaker'].unique())}")
        for bm in base_models_needed:
            aug_index = extract_embeddings(augmented_manifest, bm)
            embedding_indices[bm].update(aug_index)

    split_manifest: Dict[str, List[str]] = {}
    prediction_rows: List[dict] = []
    fold_summaries: List[dict] = []

    for speaker, train_df, test_df in leave_one_speaker_out_folds(manifest):
        if only_folds and speaker not in only_folds:
            continue
        split_manifest[speaker] = test_df["file_id"].tolist()

        if use_augmentation:
            train_aug = augmented_manifest[augmented_manifest["speaker"] != speaker]
            train_df = pd.concat([train_df, train_aug], ignore_index=True)

        for model_type in model_types:
            base = MODEL_BASE_EMBEDDING[model_type]
            embedding_index = embedding_indices[base]

            print(f"\n=== Fold: held-out={speaker}, model={model_type} "
                  f"(train={len(train_df)}, test={len(test_df)}) ===")

            model = build_model(model_type, canonical_labels, ctc_epochs=ctc_epochs)
            model.fit(train_df, embedding_index)

            correct = 0
            for row in test_df.itertuples():
                scores = model.predict_scores(row.file_id, embedding_index)
                true_idx = canonical_labels.index(row.phoneme)
                ranking = np.argsort(scores)[::-1]
                pred_idx = int(ranking[0])
                true_rank = int(np.where(ranking == true_idx)[0][0]) + 1  # 1-based

                rival_idx = int(ranking[0]) if ranking[0] != true_idx else int(ranking[1])
                margin = float(scores[true_idx] - scores[rival_idx])
                is_correct = pred_idx == true_idx
                correct += int(is_correct)

                prediction_rows.append({
                    "fold_held_out_speaker": speaker,
                    "model": model_type,
                    "file_id": row.file_id,
                    "true_phoneme": row.phoneme,
                    "predicted_phoneme": canonical_labels[pred_idx],
                    "correct": is_correct,
                    "true_target_rank": true_rank,  # 1 = model's top pick
                    "margin_vs_rival": margin,  # probability-space; see models.py docstring
                })

            accuracy = correct / len(test_df) if len(test_df) else 0.0
            fold_ranks = [
                r["true_target_rank"] for r in prediction_rows
                if r["fold_held_out_speaker"] == speaker and r["model"] == model_type
            ]
            mean_rank = float(np.mean(fold_ranks))
            print(f"Fold top-1 accuracy: {accuracy:.2%} ({correct}/{len(test_df)}), mean true-target rank: {mean_rank:.2f}")

            fold_summaries.append({
                "speaker": speaker,
                "model": model_type,
                "accuracy": accuracy,
                "n": len(test_df),
                "mean_true_target_rank": mean_rank,
                "below_bug_floor": accuracy < BUG_FLOOR_ACCURACY,
            })

    (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)

    summary = _build_summary(fold_summaries, model_types)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Results written to {run_dir} ===")
    _print_summary(summary)

    return summary


def _build_summary(fold_summaries: List[dict], model_types: List[str]) -> Dict:
    df = pd.DataFrame(fold_summaries)
    summary: Dict = {"per_fold": fold_summaries, "per_model": {}}

    for model_type in model_types:
        sub = df[df["model"] == model_type]
        headline_row = sub[sub["speaker"] == HEADLINE_SPEAKER]
        headline_acc = float(headline_row["accuracy"].iloc[0]) if len(headline_row) else None
        other = sub[sub["speaker"] != HEADLINE_SPEAKER]

        summary["per_model"][model_type] = {
            "headline_speaker": HEADLINE_SPEAKER,
            "headline_accuracy": headline_acc,
            "other_speakers_mean_accuracy": float(other["accuracy"].mean()) if len(other) else None,
            "any_fold_below_bug_floor": bool(sub["below_bug_floor"].any()),
            "meets_ship_bar": (headline_acc is not None and headline_acc >= SHIP_BAR_ACCURACY),
            "per_speaker_accuracy": {row["speaker"]: row["accuracy"] for _, row in sub.iterrows()},
        }
    return summary


def _print_summary(summary: Dict) -> None:
    print("\n--- Summary ---")
    for model_type, info in summary["per_model"].items():
        print(f"{model_type}:")
        for speaker, acc in info["per_speaker_accuracy"].items():
            flag = " <-- HEADLINE" if speaker == HEADLINE_SPEAKER else ""
            print(f"    {speaker}: {acc:.2%}{flag}")
        if info["any_fold_below_bug_floor"]:
            print(f"    !! at least one fold below {BUG_FLOOR_ACCURACY:.0%} bug floor - investigate before trusting this model's numbers")
        ship = "YES" if info["meets_ship_bar"] else "no"
        print(f"    Meets {SHIP_BAR_ACCURACY:.0%} ship bar on {HEADLINE_SPEAKER}: {ship}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run leave-one-speaker-out evaluation")
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_TYPES, choices=ALL_MODEL_TYPES)
    parser.add_argument("--ctc-epochs", type=int, default=20)
    parser.add_argument("--augment", action="store_true", help="Add noise/speed/pitch-augmented training data")
    parser.add_argument("--only-folds", nargs="+", default=None, help="Restrict to these held-out speakers")
    args = parser.parse_args()

    run_loso(
        model_types=args.models,
        ctc_epochs=args.ctc_epochs,
        use_augmentation=args.augment,
        only_folds=args.only_folds,
    )
