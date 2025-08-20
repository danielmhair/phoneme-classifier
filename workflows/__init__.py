"""
Phoneme Classifier Workflows Package

Centralized configuration for all workflow paths and constants.
"""

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
DIST_DIR = PROJECT_ROOT / "dist"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data directories
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
ORGANIZED_RECORDINGS_DIR = DIST_DIR / "organized_recordings"
ORGANIZED_RECORDINGS_LOW_QUALITY = DIST_DIR / "organized_recordings_lower_quality"
PHONEME_EMBEDDINGS_DIR = DIST_DIR / "phoneme_embeddings"
PHONEME_EMBEDDINGS_TEMPORAL_DIR = DIST_DIR / "phoneme_embeddings_temporal"

# Model files
CLEAN_MODEL_PATH = DIST_DIR / "phoneme_classifier.pkl"
LABEL_ENCODER_PATH = DIST_DIR / "label_encoder.pkl"
PHONEME_LABELS_JSON_PATH = DIST_DIR / "phoneme_labels.json"

# CTC Model files
CTC_MODEL_PATH = DIST_DIR / "ctc_model_best.pt"
CTC_LABEL_ENCODER_PATH = DIST_DIR / "ctc_label_encoder.pkl"

# ONNX Model files
WAV2VEC2_ONNX_PATH = DIST_DIR / "wav2vec2.onnx"
PHONEME_MLP_ONNX_PATH = DIST_DIR / "phoneme_mlp.onnx"

# Ensure directories exist
DIST_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Convert to strings for backward compatibility with existing code
ORGANIZED_RECORDINGS_DIR = str(ORGANIZED_RECORDINGS_DIR)
ORGANIZED_RECORDINGS_LOW_QUALITY = str(ORGANIZED_RECORDINGS_LOW_QUALITY)
PHONEME_EMBEDDINGS_DIR = str(PHONEME_EMBEDDINGS_DIR)
PHONEME_EMBEDDINGS_TEMPORAL_DIR = str(PHONEME_EMBEDDINGS_TEMPORAL_DIR)
CLEAN_MODEL_PATH = str(CLEAN_MODEL_PATH)
LABEL_ENCODER_PATH = str(LABEL_ENCODER_PATH)
PHONEME_LABELS_JSON_PATH = str(PHONEME_LABELS_JSON_PATH)
CTC_MODEL_PATH = str(CTC_MODEL_PATH)
CTC_LABEL_ENCODER_PATH = str(CTC_LABEL_ENCODER_PATH)
WAV2VEC2_ONNX_PATH = str(WAV2VEC2_ONNX_PATH)
PHONEME_MLP_ONNX_PATH = str(PHONEME_MLP_ONNX_PATH)

# Version info
__version__ = "0.1.0"
__author__ = "Light Haven Team"
