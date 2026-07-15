"""
D6 (data-collection-game PRD 8): the learning-curve protocol - the empirical
stopping rule for "how many kids do we need?".

At N = 10, 20, 30, 40... ingested training children, train (base 5-speaker
corpus + first N collected non-holdout children, in ingest order) and
evaluate on the held-out children. Report held-out-child accuracy per age
band, never just an average - "78% on kids" must not be able to hide
"61% on 5-year-olds" (PRD 8).

Curve still climbing at 20 -> recruit more; flattening at 40 -> stop.

Usage:
    poetry run python -m evaluation.harness.learning_curve
    poetry run python -m evaluation.harness.learning_curve --steps 10 20 40 \
        --models wavlm_ctc --ctc-epochs 20
    # single point using ALL training children (a plain holdout eval):
    poetry run python -m evaluation.harness.learning_curve --all-children

Outputs into evaluation/learning_curve_results/<run_id>/:
    summary.json     - per (N, model): overall + per-age-band + per-child accuracy
    predictions.csv  - one row per (N, model, holdout sample)
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.harness.dataset import build_manifest, canonical_phoneme_labels
from evaluation.harness.embeddings_cache import extract_embeddings
from evaluation.harness.holdout import (
    age_band_of,
    load_child_meta,
    split_by_holdout,
    training_children_in_ingest_order,
)
from evaluation.harness.loso_runner import ALL_MODEL_TYPES, BUG_FLOOR_ACCURACY
from evaluation.harness.models import MODEL_BASE_EMBEDDING, build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation" / "learning_curve_results"


def run_learning_curve(
    model_types: List[str] = ALL_MODEL_TYPES,
    steps: Optional[List[int]] = None,
    ctc_epochs: int = 20,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    run_id: Optional[str] = None,
    all_children: bool = False,
) -> Dict:
    manifest = build_manifest()
    child_meta = load_child_meta()
    canonical_labels = canonical_phoneme_labels(manifest)

    train_pool, holdout_df = split_by_holdout(manifest, child_meta)
    if holdout_df.empty:
        raise SystemExit(
            "No holdout children found. Export collected children first "
            "(game/export_tool/export_verified.py); holdout flags live in "
            "each child's child_meta.json."
        )

    base_speakers = sorted(
        s for s in train_pool["speaker"].unique() if s not in child_meta
    )
    ingest_order = training_children_in_ingest_order(child_meta)
    # only children that actually have exported clips
    ingest_order = [s for s in ingest_order if (train_pool["speaker"] == s).any()]

    if all_children:
        step_list = [len(ingest_order)]
    else:
        step_list = steps or [10, 20, 30, 40, 60]
        step_list = sorted({min(n, len(ingest_order)) for n in step_list if n > 0})
    if not step_list:
        raise SystemExit("No collected training children available yet.")

    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    holdout_speakers = sorted(holdout_df["speaker"].unique())
    print(f"=== Learning curve run {run_id} ===")
    print(f"Base speakers (always train): {base_speakers}")
    print(f"Training children available: {len(ingest_order)}; steps: {step_list}")
    print(f"Holdout children ({len(holdout_speakers)}): {holdout_speakers}")
    print(f"Holdout age bands: "
          f"{sorted({age_band_of(s, child_meta) for s in holdout_speakers})}")

    base_models_needed = sorted({MODEL_BASE_EMBEDDING[m] for m in model_types})
    embedding_indices = {bm: extract_embeddings(manifest, bm) for bm in base_models_needed}

    prediction_rows: List[dict] = []
    point_summaries: List[dict] = []

    for n_children in step_list:
        included = ingest_order[:n_children]
        train_df = train_pool[
            train_pool["speaker"].isin(set(base_speakers) | set(included))
        ].reset_index(drop=True)

        for model_type in model_types:
            embedding_index = embedding_indices[MODEL_BASE_EMBEDDING[model_type]]
            print(f"\n=== N={n_children} children, model={model_type} "
                  f"(train={len(train_df)}, eval={len(holdout_df)}) ===")

            model = build_model(model_type, canonical_labels, ctc_epochs=ctc_epochs)
            model.fit(train_df, embedding_index)

            for row in holdout_df.itertuples():
                scores = model.predict_scores(row.file_id, embedding_index)
                pred_idx = int(np.argmax(scores))
                prediction_rows.append({
                    "n_children": n_children,
                    "model": model_type,
                    "speaker": row.speaker,
                    "age_band": age_band_of(row.speaker, child_meta),
                    "file_id": row.file_id,
                    "true_phoneme": row.phoneme,
                    "predicted_phoneme": canonical_labels[pred_idx],
                    "correct": canonical_labels[pred_idx] == row.phoneme,
                })

            point = _summarize_point(prediction_rows, n_children, model_type)
            point_summaries.append(point)
            _print_point(point)

    summary = {
        "run_id": run_id,
        "base_speakers": base_speakers,
        "ingest_order": ingest_order,
        "holdout_speakers": holdout_speakers,
        "points": point_summaries,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_csv(
        run_dir / "predictions.csv", index=False, encoding="utf-8"
    )
    print(f"\n=== Results written to {run_dir} ===")
    return summary


def _summarize_point(prediction_rows: List[dict], n_children: int, model_type: str) -> dict:
    df = pd.DataFrame([
        r for r in prediction_rows
        if r["n_children"] == n_children and r["model"] == model_type
    ])
    overall = float(df["correct"].mean())
    per_band = {
        str(int(band)): float(g["correct"].mean())
        for band, g in df.groupby("age_band") if band is not None
    }
    per_child = {
        speaker: float(g["correct"].mean()) for speaker, g in df.groupby("speaker")
    }
    return {
        "n_children": n_children,
        "model": model_type,
        "overall_holdout_accuracy": overall,
        "per_age_band_accuracy": per_band,
        "per_child_accuracy": per_child,
        "n_eval_clips": int(len(df)),
        "below_bug_floor": overall < BUG_FLOOR_ACCURACY,
    }


def _print_point(point: dict) -> None:
    print(f"N={point['n_children']} {point['model']}: "
          f"overall {point['overall_holdout_accuracy']:.2%} "
          f"({point['n_eval_clips']} clips)")
    for band, acc in sorted(point["per_age_band_accuracy"].items()):
        print(f"    age {band}: {acc:.2%}")
    if point["below_bug_floor"]:
        print(f"    !! below {BUG_FLOOR_ACCURACY:.0%} bug floor - "
              f"investigate before trusting this point")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Learning curve: holdout-children accuracy vs number of "
                    "ingested training children, reported per age band"
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_TYPES,
                        choices=ALL_MODEL_TYPES)
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="N values (default: 10 20 30 40 60, capped at available)")
    parser.add_argument("--ctc-epochs", type=int, default=20)
    parser.add_argument("--all-children", action="store_true",
                        help="single point using every non-holdout child "
                             "(plain holdout evaluation)")
    args = parser.parse_args()

    run_learning_curve(
        model_types=args.models,
        steps=args.steps,
        ctc_epochs=args.ctc_epochs,
        all_children=args.all_children,
    )
