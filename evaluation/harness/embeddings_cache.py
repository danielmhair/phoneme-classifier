"""
Temporal embedding extraction + on-disk cache, shared across LOSO folds.

Extraction is the expensive part (a Wav2Vec2/WavLM forward pass per file,
~6000 files in the current dataset). It doesn't depend on the train/test
split, so it runs exactly once per base model (not once per fold) and is
cached to disk keyed by file_id, resumable if interrupted.

MLP Control and Wav2Vec2 CTC share the Wav2Vec2 cache (MLP mean-pools the
same temporal embedding on the fly at fit time); WavLM CTC uses its own
WavLM cache. This mirrors what the production workflows actually train on.

Deliberately no on-the-fly augmentation here (unlike
workflows/ctc_w2v2_workflow/s2_extract_embeddings_temporal.py, which applies
stochastic noise/filter augmentation to every file at extraction time) - the
harness caches deterministic embeddings so every fold sees the same features
for a given file.
"""
import json
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "evaluation" / "harness_cache"

BaseModel = Literal["wav2vec2", "wavlm"]

_MODEL_HF_NAMES = {
    "wav2vec2": "facebook/wav2vec2-base",
    "wavlm": "microsoft/wavlm-base",
}


def _load_backbone(base_model: BaseModel):
    import torch
    # Wav2Vec2FeatureExtractor, not Wav2Vec2Processor: we only ever use audio
    # feature extraction, never text tokenization/decoding, and
    # microsoft/wavlm-base has no tokenizer config at all - Wav2Vec2Processor
    # (which bundles a required tokenizer) fails to load for it with an
    # OSError. This is the same bug present in
    # workflows/ctc_wavlm_workflow/s2_extract_embeddings_temporal.py, which
    # is a plausible real contributor to WavLM-CTC's near-random reported
    # accuracy - if embedding extraction itself never completed, training
    # would have run on garbage/empty embeddings.
    from transformers import Wav2Vec2FeatureExtractor

    hf_name = _MODEL_HF_NAMES[base_model]
    processor = Wav2Vec2FeatureExtractor.from_pretrained(hf_name)
    if base_model == "wav2vec2":
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(hf_name).eval()
    else:
        from transformers import WavLMModel
        model = WavLMModel.from_pretrained(hf_name).eval()
    return processor, model, torch


def extract_embeddings(
    manifest: pd.DataFrame,
    base_model: BaseModel,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    sample_rate: int = 16000,
    max_length: int = 1000,
    progress_every: int = 100,
) -> Dict[str, str]:
    """Extract (or reuse cached) temporal embeddings for every row in manifest.

    Returns a dict mapping file_id -> path of the cached .npy embedding
    ([T, 768] float32, no pooling, no augmentation).
    """
    out_dir = cache_dir / base_model
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"

    index: Dict[str, str] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())

    processor = model = torch = None  # lazy-loaded only if there's work to do
    to_process = [row for row in manifest.itertuples() if row.file_id not in index]

    if to_process:
        print(f"[{base_model}] extracting {len(to_process)} embeddings ({len(manifest) - len(to_process)} already cached)")
        processor, model, torch = _load_backbone(base_model)

    for i, row in enumerate(to_process):
        if i % progress_every == 0:
            print(f"[{base_model}] {i}/{len(to_process)}")

        audio, sr = sf.read(row.filepath)
        if sr != sample_rate:
            raise ValueError(f"{row.filepath}: expected {sample_rate}Hz, got {sr}Hz")

        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0).numpy()  # [T, 768]
        if emb.shape[0] > max_length:
            emb = emb[:max_length]

        safe_name = row.file_id.replace("/", "__").replace(".wav", ".npy")
        emb_path = out_dir / safe_name
        np.save(emb_path, emb.astype(np.float32))
        index[row.file_id] = str(emb_path)

        # Persist incrementally so an interrupted run doesn't lose progress.
        if i % progress_every == 0:
            index_path.write_text(json.dumps(index, indent=2))

    index_path.write_text(json.dumps(index, indent=2))

    missing = [fid for fid in manifest["file_id"] if fid not in index]
    if missing:
        raise RuntimeError(f"[{base_model}] {len(missing)} files failed to extract, e.g. {missing[:5]}")

    return index


def load_embedding(file_id: str, index: Dict[str, str]) -> np.ndarray:
    return np.load(index[file_id])
