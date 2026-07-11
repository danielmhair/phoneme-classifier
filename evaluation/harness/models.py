"""
Unified model interface for the LOSO harness: one .fit()/.predict_scores()
shape for MLP Control, Wav2Vec2 CTC, and WavLM CTC, so a fold-runner can
treat all three identically (per the PRD's "Multi-Model Bake-off Harness"
intent - later models/data should slot in without rework).

Every model's predict_scores() returns a score vector aligned to a single,
fixed `canonical_labels` list (see dataset.canonical_phoneme_labels) - not a
per-fold-refit label ordering. This closes the label-index-mismatch bug
class by construction: there is exactly one label->index mapping, computed
once, and every model/fold is required to conform to it (fit() asserts this).
"""
from typing import Dict, List, Protocol

import numpy as np
import pandas as pd

from workflows.shared.ctc_decode import ctc_predict


class PhonemeModel(Protocol):
    def fit(self, train_df: pd.DataFrame, embedding_index: Dict[str, str]) -> None: ...

    def predict_scores(self, file_id: str, embedding_index: Dict[str, str]) -> np.ndarray:
        """Return a [len(canonical_labels)] score vector (higher = more likely)."""
        ...


class MLPPhonemeModel:
    """Mean-pooled Wav2Vec2 embeddings -> sklearn MLPClassifier.

    Matches workflows/mlp_control_workflow/s3_classifier_encoder.py's
    architecture/hyperparameters (hidden_layer_sizes=(128, 64), max_iter=500,
    random_state=42), but fits against the fixed canonical_labels index space
    instead of a per-fold-inferred sklearn LabelEncoder.
    """

    def __init__(self, canonical_labels: List[str]):
        self.canonical_labels = canonical_labels
        self.label_to_idx = {label: i for i, label in enumerate(canonical_labels)}
        self.clf = None

    def fit(self, train_df: pd.DataFrame, embedding_index: Dict[str, str]) -> None:
        from sklearn.neural_network import MLPClassifier

        X, y = [], []
        for row in train_df.itertuples():
            emb = np.load(embedding_index[row.file_id])  # [T, 768]
            X.append(emb.mean(axis=0))  # mean-pool, matches production MLP features
            y.append(self.label_to_idx[row.phoneme])
        X = np.stack(X)
        y = np.array(y)

        present = set(y.tolist())
        missing = set(range(len(self.canonical_labels))) - present
        if missing:
            missing_names = [self.canonical_labels[i] for i in sorted(missing)]
            raise RuntimeError(
                f"MLP training fold is missing {len(missing)} canonical classes entirely: "
                f"{missing_names}. Cannot fit a stable {len(self.canonical_labels)}-way "
                f"classifier - this is a real dataset-coverage gap, not a bug to paper over."
            )

        self.clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        self.clf.fit(X, y)

        if list(self.clf.classes_) != list(range(len(self.canonical_labels))):
            raise RuntimeError(
                f"MLPClassifier.classes_ {list(self.clf.classes_)} does not align with "
                f"canonical_labels index space 0..{len(self.canonical_labels) - 1} - "
                f"predict_proba() columns would silently misalign with phoneme identity."
            )

    def predict_scores(self, file_id: str, embedding_index: Dict[str, str]) -> np.ndarray:
        emb = np.load(embedding_index[file_id]).mean(axis=0, keepdims=True)  # [1, 768]
        return self.clf.predict_proba(emb)[0]  # [len(canonical_labels)]


class _EmbeddingDataset:
    """Minimal torch Dataset over (df row -> cached embedding) for CTC training."""

    def __init__(self, df: pd.DataFrame, embedding_index: Dict[str, str], label_to_idx: Dict[str, int]):
        self.rows = list(df.itertuples())
        self.embedding_index = embedding_index
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import torch

        row = self.rows[idx]
        emb = np.load(self.embedding_index[row.file_id])  # [T, 768]
        phoneme_idx = self.label_to_idx[row.phoneme]
        return {
            "embedding": torch.FloatTensor(emb),
            "phoneme": torch.LongTensor([phoneme_idx]),
            "phoneme_length": torch.LongTensor([1]),
            "embedding_length": torch.LongTensor([emb.shape[0]]),
        }


class CTCPhonemeModel:
    """LSTM+CTC model over temporal embeddings, reusing the production
    training loop (workflows/ctc_w2v2_workflow/s3_ctc_classifier.py -
    identical file content in the wavlm workflow, so importing from either
    is equivalent). Inference uses the shared greedy-decode module instead
    of any of the three decode algorithms that used to be scattered across
    the codebase.
    """

    def __init__(self, canonical_labels: List[str], num_epochs: int = 20, batch_size: int = 32):
        self.canonical_labels = canonical_labels
        self.label_to_idx = {label: i for i, label in enumerate(canonical_labels)}
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, train_df: pd.DataFrame, embedding_index: Dict[str, str]) -> None:
        import torch
        from torch.utils.data import DataLoader

        from workflows.ctc_w2v2_workflow.models.ctc_model import create_ctc_model
        from workflows.ctc_w2v2_workflow.s3_ctc_classifier import CTCTrainer, collate_fn

        present = set(train_df["phoneme"].unique())
        missing = set(self.canonical_labels) - present
        if missing:
            raise RuntimeError(
                f"CTC training fold is missing {len(missing)} canonical classes entirely: "
                f"{sorted(missing)}. Cannot fit a stable {len(self.canonical_labels)}-way "
                f"classifier - this is a real dataset-coverage gap, not a bug to paper over."
            )

        dataset = _EmbeddingDataset(train_df, embedding_index, self.label_to_idx)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

        self.model = create_ctc_model(num_classes=len(self.canonical_labels))
        trainer = CTCTrainer(self.model, device="cpu")
        for epoch in range(self.num_epochs):
            train_loss = trainer.train_epoch(loader)
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                print(f"  [CTC] epoch {epoch + 1}/{self.num_epochs} train_loss={train_loss:.4f}")
        self.model.eval()

    def predict_scores(self, file_id: str, embedding_index: Dict[str, str]) -> np.ndarray:
        import torch

        emb = np.load(embedding_index[file_id])  # [T, 768]
        x = torch.FloatTensor(emb).unsqueeze(0)  # [1, T, 768]
        with torch.no_grad():
            log_probs, _ = self.model(x)  # [1, T, num_classes]
        log_probs_np = log_probs.squeeze(0).numpy()  # [T, num_classes]
        _, probabilities, _ = ctc_predict(log_probs_np)  # aligned to canonical_labels (blank = last index)
        return probabilities


def build_model(model_type: str, canonical_labels: List[str], ctc_epochs: int = 20) -> PhonemeModel:
    if model_type == "mlp_control":
        return MLPPhonemeModel(canonical_labels)
    elif model_type in ("wav2vec2_ctc", "wavlm_ctc"):
        return CTCPhonemeModel(canonical_labels, num_epochs=ctc_epochs)
    raise ValueError(f"Unknown model_type: {model_type}")


MODEL_BASE_EMBEDDING = {
    "mlp_control": "wav2vec2",
    "wav2vec2_ctc": "wav2vec2",
    "wavlm_ctc": "wavlm",
}
