"""
Train all three models on the FULL corpus (all 5 speakers) from the harness
embedding caches, and persist them for the live-mic smoke test (PRD §5.4).

This is deliberately NOT a LOSO fold: the live-mic smoke test evaluates a
genuinely fresh recording session (new mic session, new clips), so every
existing speaker's data is fair game for training. Models are trained with
the exact same code path as the LOSO harness (models.py -> production
CTCTrainer / production MLP hyperparameters), no augmentation, 20 CTC epochs
by default - matching the no-augmentation LOSO baseline these smoke-test
results will be compared against.

Outputs (default evaluation/full_models/):
  canonical_labels.json  - the single fixed label order all models share
  mlp_control.pkl        - pickled sklearn MLPClassifier
  wav2vec2_ctc.pt        - torch checkpoint {"model_state_dict", "canonical_labels"}
  wavlm_ctc.pt           - same, WavLM-based
  training_info.json     - what was trained on, when, with what settings

Run:  poetry run python -m evaluation.harness.full_train [--models ...] [--ctc-epochs N]
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from evaluation.harness.dataset import build_manifest, canonical_phoneme_labels
from evaluation.harness.embeddings_cache import extract_embeddings
from evaluation.harness.models import MODEL_BASE_EMBEDDING, build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "evaluation" / "full_models"
ALL_MODEL_TYPES = ["mlp_control", "wav2vec2_ctc", "wavlm_ctc"]


def train_full_models(model_types=ALL_MODEL_TYPES, ctc_epochs: int = 20, out_dir: Path = DEFAULT_OUT_DIR,
                      channel_aug: bool = False, channel_variants=None):
    import torch

    manifest = build_manifest()
    canonical_labels = canonical_phoneme_labels(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "canonical_labels.json").write_text(
        json.dumps(canonical_labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base_models_needed = sorted({MODEL_BASE_EMBEDDING[m] for m in model_types})
    embedding_indices = {bm: extract_embeddings(manifest, bm) for bm in base_models_needed}

    train_manifest = manifest
    if channel_aug:
        from evaluation.harness.channel_augmentation import build_channel_augmented_manifest

        channel_manifest = build_channel_augmented_manifest(manifest)
        if channel_variants:
            suffixes = tuple(f"_ch_{label}.wav" for label in channel_variants)
            channel_manifest = channel_manifest[
                channel_manifest["file_id"].str.endswith(suffixes)
            ].reset_index(drop=True)
            print(f"Restricted to channel variants {channel_variants}")
        print(f"Channel-augmented manifest: {len(channel_manifest)} synthetic training rows")
        for bm in base_models_needed:
            aug_index = extract_embeddings(channel_manifest, bm)
            embedding_indices[bm].update(aug_index)
        train_manifest = pd.concat([manifest, channel_manifest], ignore_index=True)

    print(f"=== Full-corpus training: {len(train_manifest)} files "
          f"({len(manifest)} original), {len(canonical_labels)} classes, "
          f"models={model_types}, ctc_epochs={ctc_epochs}, channel_aug={channel_aug} ===")

    for model_type in model_types:
        start = time.time()
        print(f"\n--- Training {model_type} on full corpus ---")
        model = build_model(model_type, canonical_labels, ctc_epochs=ctc_epochs)
        model.fit(train_manifest, embedding_indices[MODEL_BASE_EMBEDDING[model_type]])

        if model_type == "mlp_control":
            out_path = out_dir / "mlp_control.pkl"
            with open(out_path, "wb") as f:
                pickle.dump({"clf": model.clf, "canonical_labels": canonical_labels}, f)
        else:
            out_path = out_dir / f"{model_type}.pt"
            torch.save(
                {"model_state_dict": model.model.state_dict(), "canonical_labels": canonical_labels},
                out_path,
            )
        print(f"Saved {out_path} ({time.time() - start:.0f}s)")

    training_info = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_training_files": len(train_manifest),
        "num_original_files": len(manifest),
        "speakers": sorted(manifest["speaker"].unique().tolist()),
        "num_classes": len(canonical_labels),
        "ctc_epochs": ctc_epochs,
        "augmentation": "channel" if channel_aug else False,
        "channel_variants": channel_variants or ("all" if channel_aug else None),
        "note": "Full-corpus (non-LOSO) models for the live-mic smoke test / "
                "channel-augmentation experiment (plans/channel-augmentation-experiment.md).",
    }
    (out_dir / "training_info.json").write_text(json.dumps(training_info, indent=2), encoding="utf-8")
    print(f"\n=== Done. Models in {out_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train full-corpus models for the live-mic smoke test")
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_TYPES, choices=ALL_MODEL_TYPES)
    parser.add_argument("--ctc-epochs", type=int, default=20)
    parser.add_argument("--channel-aug", action="store_true",
                        help="Add channel/mic-simulation augmented variants to the training set")
    parser.add_argument("--channel-variants", nargs="+", default=None,
                        choices=["tel", "cheapmic", "clip", "8k"],
                        help="Restrict channel augmentation to these variant types (default: all)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Where to save the trained models")
    args = parser.parse_args()
    train_full_models(model_types=args.models, ctc_epochs=args.ctc_epochs,
                      out_dir=args.out_dir, channel_aug=args.channel_aug,
                      channel_variants=args.channel_variants)
