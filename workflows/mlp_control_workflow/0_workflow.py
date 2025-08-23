# Import centralized paths and workflow modules
from workflows import (
    MLP_ORGANIZED_RECORDINGS_DIR,
    MLP_PHONEME_EMBEDDINGS_DIR,
    MLP_ORGANIZED_RECORDINGS_LOW_QUALITY,
    MLP_MODEL_PATH,
    MLP_LABEL_ENCODER_PATH,
    MLP_PHONEME_LABELS_JSON_PATH,
    MLP_DIST_DIR
)
from workflows.shared.workflow_executor import execute_workflow_steps
from workflows.shared.s0_s1_shared_utils import prepare_wav_files_clean, cleanup_dist
from workflows.mlp_control_workflow.s2_extract_embeddings_for_phonemes import extract_embeddings_for_phonemes
from workflows.mlp_control_workflow.s3_classifier_encoder import classifier_encoder
from workflows.mlp_control_workflow.s5_visualize_results import visualize_results
from workflows.mlp_control_workflow.s6_confusion_pairs import analyze_confusion
from workflows.mlp_control_workflow.s7_batch_test_phonemes import batch_test_phonemes
from workflows.mlp_control_workflow.s8_benchmark_inference_and_save import benchmark_and_save
from workflows.mlp_control_workflow.s9_trace_mlp_classifier import trace_mlp_classifier
from workflows.mlp_control_workflow.s10_onnx_export import onnx_export
from workflows.mlp_control_workflow.s11_onnx_test import onnx_batch_test
from workflows.mlp_control_workflow.s12_overwrite_onnx_unreal import overwrite_onnx_unreal
from workflows.mlp_control_workflow import timestamp


def main():
    import sys
    import os
    
    # Force unbuffered output for immediate logging
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) if hasattr(sys.stdout, 'fileno') else sys.stdout
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0) if hasattr(sys.stderr, 'fileno') else sys.stderr
    
    print(f"🕐 Workflow started at: {timestamp}")
    print(f"📝 Process ID: {os.getpid()}")
    print(f"🔄 Starting MLP Control Workflow...")
    print("=" * 60)
    sys.stdout.flush()

    steps = [
        ("Cleanup previous runs", cleanup_dist),
        ("Prepare the dataset (organize and augment)", prepare_wav_files_clean),
        ("Extract embeddings for phonemes", extract_embeddings_for_phonemes_clean),
        ("Save Classifier and encoder", classifier_encoder_clean),
        ("Visualize Results (before fine-tune)", visualize_results_clean),
        ("Analyze confusion pairs (before fine-tune)", analyze_confusion_clean),
        ("Batch test phonemes for initial", batch_test_phonemes_clean),

        # Save inference model and onnx
        ("Benchmark inference and save", benchmark_and_save),
        ("Trace MLP classifier", trace_mlp_classifier),
        ("Export to ONNX", onnx_export),
        ("Test ONNX model", onnx_batch_test),
        ("Copy to Unreal Engine", overwrite_onnx_unreal)
    ]

    # Execute workflow using shared executor
    execute_workflow_steps(steps, "MLP Control Workflow")
    print("You can test phonemes through the command line with w5_record_voice_cli.py")


def extract_embeddings_for_phonemes_clean():
    extract_embeddings_for_phonemes(
        input_dir=MLP_ORGANIZED_RECORDINGS_DIR,
        output_dir=MLP_PHONEME_EMBEDDINGS_DIR,
        phoneme_label_json_path=MLP_PHONEME_LABELS_JSON_PATH,
    )


def classifier_encoder_clean():
    classifier_encoder(
        input_dir_str=MLP_PHONEME_EMBEDDINGS_DIR,
        classifier_out=MLP_MODEL_PATH,
        encoder_out=MLP_LABEL_ENCODER_PATH
    )


def visualize_results_clean():
    # TODO: This should be getting the metadata.csv, but I think its saving it during organized_recordings. Is this accurate? Probably, but then we need to pass it into visualize_results
    visualize_results(
        classifier_path=MLP_MODEL_PATH,
        label_encoder_path=MLP_LABEL_ENCODER_PATH,
        type_name="initial",
        embeddings_dir=MLP_PHONEME_EMBEDDINGS_DIR,
    )


def analyze_confusion_clean():
    analyze_confusion(
        classifier_path=MLP_MODEL_PATH,
        label_encoder_path=MLP_LABEL_ENCODER_PATH,
        embeddings_dir=MLP_PHONEME_EMBEDDINGS_DIR,
    )


def batch_test_phonemes_clean():
    batch_test_phonemes(
        classifier_path=MLP_MODEL_PATH,
        label_encoder_path=MLP_LABEL_ENCODER_PATH,
        phoneme_label_json_path=MLP_PHONEME_LABELS_JSON_PATH,
        training_recordings_path=MLP_ORGANIZED_RECORDINGS_DIR,
        finetuned_recordings_path=MLP_ORGANIZED_RECORDINGS_LOW_QUALITY,
    )


if __name__ == "__main__":
    main()
