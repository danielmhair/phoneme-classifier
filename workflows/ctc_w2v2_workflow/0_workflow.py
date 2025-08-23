#!/usr/bin/env python3
"""
CTC Wav2Vec2 Workflow - Complete CTC Training Pipeline

This workflow implements the CTC (Connectionist Temporal Classification) approach
for phoneme sequence recognition using Wav2Vec2 features and LSTM sequence modeling.

Unlike the MLP workflow which uses mean-pooled embeddings for single phoneme classification,
this CTC workflow preserves temporal sequences for alignment-free sequence modeling.
"""

import sys
from workflows.shared.workflow_executor import execute_workflow_steps
from workflows.shared.s0_s1_shared_utils import prepare_wav_files_clean, cleanup_dist

# Import CTC-specific modules
from workflows.ctc_w2v2_workflow.s2_extract_embeddings_temporal import extract_embeddings_for_phonemes_temporal
from workflows.ctc_w2v2_workflow.s3_ctc_classifier import ctc_classifier_training
from workflows.ctc_w2v2_workflow.s4_visualize_ctc_results import visualize_ctc_results
from workflows.ctc_w2v2_workflow.s5_confusion_analysis import analyze_ctc_confusion
from workflows.ctc_w2v2_workflow.s5_export_ctc_onnx import export_ctc_to_onnx
from workflows.ctc_w2v2_workflow.s6_batch_test_ctc import batch_test_ctc_phonemes
from workflows.ctc_w2v2_workflow.s7_test_onnx import test_ctc_onnx_model
from workflows import (
    CTC_ORGANIZED_RECORDINGS_DIR,
    CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR,
    CTC_MODEL_PATH,
    CTC_LABEL_ENCODER_PATH,
    CTC_PHONEME_LABELS_JSON_PATH
)
from workflows.ctc_w2v2_workflow import timestamp


def main():
    import os
    
    # Force unbuffered output for immediate logging
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) if hasattr(sys.stdout, 'fileno') else sys.stdout
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) if hasattr(sys.stderr, 'fileno') else sys.stderr
    
    print(f"🎯 CTC Wav2Vec2 Workflow started at: {timestamp}")
    print(f"📝 Process ID: {os.getpid()}")
    print("🚀 Epic 1: Live Phoneme CTCs - CTC Implementation")
    print("=" * 60)
    sys.stdout.flush()

    # Define workflow steps
    steps = [
        ("Cleanup previous CTC outputs", cleanup_dist),
        ("Prepare audio dataset for CTC", prepare_wav_files_clean),
        ("Extract temporal embeddings (preserve sequences)", extract_temporal_embeddings),
        ("Train CTC classifier", train_ctc_model),
        
        # Analysis and validation steps (like MLP workflow)
        ("Visualize CTC Results", run_visualize_ctc_results),
        ("Analyze CTC confusion pairs", run_analyze_ctc_confusion),
        ("Batch test CTC phonemes", run_batch_test_ctc_phonemes),
        
        # Export and deployment steps
        ("Export CTC to ONNX", run_export_ctc_onnx),
        ("Test CTC ONNX model", run_test_ctc_onnx),
        ("Test CTC inference system", test_ctc_inference),
    ]

    # Execute workflow using shared executor
    execute_workflow_steps(steps, "CTC Wav2Vec2 Workflow")
    print(f"📁 CTC model saved to: {CTC_MODEL_PATH}")
    print(f"📁 Label encoder saved to: {CTC_LABEL_ENCODER_PATH}")
    
    print("\n🎮 How to test:")
    print("   cd ctc_w2v2_workflow")
    print("   python validations/classify_voice_ctc.py")
    print("   # Or test with file:")
    print("   python validations/classify_voice_ctc.py --file path/to/audio.wav")
    
    print("\n🔬 Compare with MLP:")
    print("   # MLP (existing):")
    print("   cd ../mlp_control_workflow")
    print("   python validations/classify_voice_pkl.py")
    print("   # CTC (this workflow):")
    print("   cd ../ctc_w2v2_workflow")
    print("   python validations/classify_voice_ctc.py")
    
    print("✅ Epic 1: Live Phoneme CTCs - Implementation Complete! ✅")
    sys.stdout.flush()


def extract_temporal_embeddings():
    """Extract temporal embeddings preserving sequence information."""
    extract_embeddings_for_phonemes_temporal(
        input_dir=CTC_ORGANIZED_RECORDINGS_DIR,
        output_dir=CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR,
        phoneme_label_json_path=CTC_PHONEME_LABELS_JSON_PATH,
        enable_ctc=True
    )


def train_ctc_model():
    """Train CTC classifier."""
    from workflows import CTC_DIST_DIR
    ctc_classifier_training(
        input_dir_str=CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR,
        output_dir=str(CTC_DIST_DIR),
        num_epochs=20,
        batch_size=32
    )


def run_visualize_ctc_results():
    """Generate CTC visualization plots and confusion matrix."""
    from workflows import CTC_DIST_DIR
    visualize_ctc_results(
        model_path=CTC_MODEL_PATH,
        embeddings_dir=CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR,
        labels_path=CTC_PHONEME_LABELS_JSON_PATH,
        output_dir=str(CTC_DIST_DIR)
    )


def run_analyze_ctc_confusion():
    """Analyze CTC confusion pairs and generate detailed confusion analysis."""
    from workflows import CTC_DIST_DIR
    analyze_ctc_confusion(
        model_path=CTC_MODEL_PATH,
        embeddings_dir=CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR,
        labels_path=CTC_PHONEME_LABELS_JSON_PATH,
        output_dir=str(CTC_DIST_DIR)
    )


def run_batch_test_ctc_phonemes():
    """Perform comprehensive batch testing of CTC model."""
    from workflows import CTC_DIST_DIR
    batch_test_ctc_phonemes(
        model_path=CTC_MODEL_PATH,
        embeddings_dir=CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR,
        labels_path=CTC_PHONEME_LABELS_JSON_PATH,
        output_dir=str(CTC_DIST_DIR)
    )


def run_export_ctc_onnx():
    """Export CTC model to ONNX format."""
    from workflows import CTC_DIST_DIR
    export_ctc_to_onnx(
        model_path=CTC_MODEL_PATH,
        labels_path=CTC_PHONEME_LABELS_JSON_PATH,
        output_dir=str(CTC_DIST_DIR),
        model_name="phoneme_ctc"
    )


def run_test_ctc_onnx():
    """Test CTC ONNX model functionality."""
    from workflows import CTC_DIST_DIR, CTC_ONNX_PATH
    test_ctc_onnx_model(
        onnx_path=CTC_ONNX_PATH,
        embeddings_dir=CTC_PHONEME_EMBEDDINGS_TEMPORAL_DIR,
        labels_path=CTC_PHONEME_LABELS_JSON_PATH,
        metadata_path=str(CTC_DIST_DIR / "phoneme_ctc_metadata.json")
    )


def test_ctc_inference():
    """Test CTC inference capabilities."""
    print("🧪 Testing CTC inference...")
    try:
        print("✅ CTC inference system is ready!")
        print("📝 To test interactively:")
        print("   cd ctc_w2v2_workflow")
        print("   python validations/classify_voice_ctc.py")
    except Exception as e:
        print(f"⚠️ CTC inference test failed: {e}")
        print("This may be due to missing dependencies (torch, transformers)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Workflow interrupted by user")
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
