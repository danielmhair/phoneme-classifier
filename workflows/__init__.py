"""
Phoneme Classifier Workflows Package

Centralized configuration for all workflow paths and constants.
Updated to support workflow-specific dist directories for better organization.
"""

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
LOGS_DIR = PROJECT_ROOT / "logs"

# Workflow-specific directories
MLP_WORKFLOW_DIR = WORKFLOWS_DIR / "mlp_control_workflow"
CTC_WORKFLOW_DIR = WORKFLOWS_DIR / "ctc_w2v2_workflow"
WAVLM_WORKFLOW_DIR = WORKFLOWS_DIR / "ctc_wavlm_workflow"
MLP_DIST_DIR = MLP_WORKFLOW_DIR / "dist"
CTC_DIST_DIR = CTC_WORKFLOW_DIR / "dist"
WAVLM_DIST_DIR = WAVLM_WORKFLOW_DIR / "dist"

# Legacy shared dist directory (for migration)
LEGACY_DIST_DIR = PROJECT_ROOT / "dist"

# Data directories (shared between workflows)
RECORDINGS_DIR = PROJECT_ROOT / "recordings"

# MLP-specific paths
MLP_ORGANIZED_RECORDINGS_DIR = MLP_DIST_DIR / "organized_recordings"
MLP_ORGANIZED_RECORDINGS_LOW_QUALITY = MLP_DIST_DIR / "organized_recordings_lower_quality"
MLP_PHONEME_EMBEDDINGS_DIR = MLP_DIST_DIR / "phoneme_embeddings"
MLP_MODEL_PATH = MLP_DIST_DIR / "phoneme_classifier.pkl"
MLP_LABEL_ENCODER_PATH = MLP_DIST_DIR / "label_encoder.pkl"
MLP_PHONEME_LABELS_JSON_PATH = MLP_DIST_DIR / "phoneme_labels.json"
MLP_WAV2VEC2_ONNX_PATH = MLP_DIST_DIR / "wav2vec2.onnx"
MLP_PHONEME_ONNX_PATH = MLP_DIST_DIR / "phoneme_mlp.onnx"

# CTC-specific paths
CTC_ORGANIZED_RECORDINGS_DIR = CTC_DIST_DIR / "organized_recordings"
CTC_ORGANIZED_RECORDINGS_LOW_QUALITY = CTC_DIST_DIR / "organized_recordings_lower_quality"
CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR = CTC_DIST_DIR / "phoneme_embeddings_temporal"
CTC_MODEL_PATH = CTC_DIST_DIR / "ctc_model_best.pt"
CTC_LABEL_ENCODER_PATH = CTC_DIST_DIR / "ctc_label_encoder.pkl"
CTC_PHONEME_LABELS_JSON_PATH = CTC_DIST_DIR / "phoneme_labels.json"
CTC_ONNX_PATH = CTC_DIST_DIR / "phoneme_ctc.onnx"

# WavLM CTC-specific paths
WAVLM_ORGANIZED_RECORDINGS_DIR = WAVLM_DIST_DIR / "organized_recordings"
WAVLM_ORGANIZED_RECORDINGS_LOW_QUALITY = WAVLM_DIST_DIR / "organized_recordings_lower_quality"
WAVLM_PHONEME_EMBEDDINGS_TEMPORAL_DIR = WAVLM_DIST_DIR / "phoneme_embeddings_temporal"
WAVLM_MODEL_PATH = WAVLM_DIST_DIR / "ctc_model_best.pt"
WAVLM_LABEL_ENCODER_PATH = WAVLM_DIST_DIR / "ctc_label_encoder.pkl"
WAVLM_PHONEME_LABELS_JSON_PATH = WAVLM_DIST_DIR / "phoneme_labels.json"
WAVLM_ONNX_PATH = WAVLM_DIST_DIR / "phoneme_ctc.onnx"

# Shared paths (for backward compatibility - use workflow-specific paths for new code)
ORGANIZED_RECORDINGS_DIR = MLP_ORGANIZED_RECORDINGS_DIR
ORGANIZED_RECORDINGS_LOW_QUALITY = MLP_ORGANIZED_RECORDINGS_LOW_QUALITY
PHONEME_EMBEDDINGS_DIR = MLP_PHONEME_EMBEDDINGS_DIR
PHONEME_EMBEDDINGS_TEMPORAL_DIR = CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR
CLEAN_MODEL_PATH = MLP_MODEL_PATH
LABEL_ENCODER_PATH = MLP_LABEL_ENCODER_PATH
PHONEME_LABELS_JSON_PATH = MLP_PHONEME_LABELS_JSON_PATH
WAV2VEC2_ONNX_PATH = MLP_WAV2VEC2_ONNX_PATH
PHONEME_MLP_ONNX_PATH = MLP_PHONEME_ONNX_PATH

# Ensure directories exist
MLP_DIST_DIR.mkdir(exist_ok=True)
CTC_DIST_DIR.mkdir(exist_ok=True)
WAVLM_DIST_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Convert to strings for backward compatibility with existing code
ORGANIZED_RECORDINGS_DIR = str(ORGANIZED_RECORDINGS_DIR)
ORGANIZED_RECORDINGS_LOW_QUALITY = str(ORGANIZED_RECORDINGS_LOW_QUALITY)
PHONEME_EMBEDDINGS_DIR = str(PHONEME_EMBEDDINGS_DIR)
PHONEME_EMBEDDINGS_TEMPORAL_DIR = str(PHONEME_EMBEDDINGS_TEMPORAL_DIR)
CLEAN_MODEL_PATH = str(CLEAN_MODEL_PATH)
LABEL_ENCODER_PATH = str(LABEL_ENCODER_PATH)
PHONEME_LABELS_JSON_PATH = str(PHONEME_LABELS_JSON_PATH)
WAV2VEC2_ONNX_PATH = str(WAV2VEC2_ONNX_PATH)
PHONEME_MLP_ONNX_PATH = str(PHONEME_MLP_ONNX_PATH)

# CTC-specific string paths
CTC_ORGANIZED_RECORDINGS_DIR = str(CTC_ORGANIZED_RECORDINGS_DIR)
CTC_ORGANIZED_RECORDINGS_LOW_QUALITY = str(CTC_ORGANIZED_RECORDINGS_LOW_QUALITY)
CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR = str(CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR)
CTC_MODEL_PATH = str(CTC_MODEL_PATH)
CTC_LABEL_ENCODER_PATH = str(CTC_LABEL_ENCODER_PATH)
CTC_PHONEME_LABELS_JSON_PATH = str(CTC_PHONEME_LABELS_JSON_PATH)
CTC_ONNX_PATH = str(CTC_ONNX_PATH)

# WavLM CTC-specific string paths
WAVLM_ORGANIZED_RECORDINGS_DIR = str(WAVLM_ORGANIZED_RECORDINGS_DIR)
WAVLM_ORGANIZED_RECORDINGS_LOW_QUALITY = str(WAVLM_ORGANIZED_RECORDINGS_LOW_QUALITY)
WAVLM_PHONEME_EMBEDDINGS_TEMPORAL_DIR = str(WAVLM_PHONEME_EMBEDDINGS_TEMPORAL_DIR)
WAVLM_MODEL_PATH = str(WAVLM_MODEL_PATH)
WAVLM_LABEL_ENCODER_PATH = str(WAVLM_LABEL_ENCODER_PATH)
WAVLM_PHONEME_LABELS_JSON_PATH = str(WAVLM_PHONEME_LABELS_JSON_PATH)
WAVLM_ONNX_PATH = str(WAVLM_ONNX_PATH)

# Version info
__version__ = "0.2.0"
__author__ = "Light Haven Team"

# Migration utilities
def get_mlp_paths():
    """Get all MLP-specific paths as a dictionary."""
    return {
        'dist_dir': str(MLP_DIST_DIR),
        'organized_recordings': str(MLP_ORGANIZED_RECORDINGS_DIR),
        'organized_recordings_low_quality': str(MLP_ORGANIZED_RECORDINGS_LOW_QUALITY),
        'phoneme_embeddings': str(MLP_PHONEME_EMBEDDINGS_DIR),
        'model_path': str(MLP_MODEL_PATH),
        'label_encoder_path': str(MLP_LABEL_ENCODER_PATH),
        'phoneme_labels_json': str(MLP_PHONEME_LABELS_JSON_PATH),
        'wav2vec2_onnx': str(MLP_WAV2VEC2_ONNX_PATH),
        'phoneme_onnx': str(MLP_PHONEME_ONNX_PATH),
    }

def get_ctc_paths():
    """Get all CTC-specific paths as a dictionary."""
    return {
        'dist_dir': str(CTC_DIST_DIR),
        'organized_recordings': str(CTC_ORGANIZED_RECORDINGS_DIR),
        'organized_recordings_low_quality': str(CTC_ORGANIZED_RECORDINGS_LOW_QUALITY),
        'phoneme_embeddings_temporal': str(CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR),
        'model_path': str(CTC_MODEL_PATH),
        'label_encoder_path': str(CTC_LABEL_ENCODER_PATH),
        'phoneme_labels_json': str(CTC_PHONEME_LABELS_JSON_PATH),
        'ctc_onnx': str(CTC_ONNX_PATH),
    }

def get_wavlm_paths():
    """Get all WavLM CTC-specific paths as a dictionary."""
    return {
        'dist_dir': str(WAVLM_DIST_DIR),
        'organized_recordings': str(WAVLM_ORGANIZED_RECORDINGS_DIR),
        'organized_recordings_low_quality': str(WAVLM_ORGANIZED_RECORDINGS_LOW_QUALITY),
        'phoneme_embeddings_temporal': str(WAVLM_PHONEME_EMBEDDINGS_TEMPORAL_DIR),
        'model_path': str(WAVLM_MODEL_PATH),
        'label_encoder_path': str(WAVLM_LABEL_ENCODER_PATH),
        'phoneme_labels_json': str(WAVLM_PHONEME_LABELS_JSON_PATH),
        'wavlm_onnx': str(WAVLM_ONNX_PATH),
    }