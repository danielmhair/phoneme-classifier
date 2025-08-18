from pathlib import Path
import sys
import os
import time
from datetime import datetime

# Add project root to path for shared utilities
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from workflows.shared.workflow_executor import execute_workflow_steps, create_workflow_logger

# Create logs/ directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = time.time()
log_file_path = f"logs/workflow_log_{timestamp}.txt"

# Redirect stdout and stderr to both terminal and log file
sys.stdout = create_workflow_logger(log_file_path)
sys.stderr = sys.stdout

from workflows.shared.s0_s1_shared_utils import prepare_wav_files_clean, cleanup_dist
from workflows.mlp_control_workflow.s2_extract_embeddings_for_phonemes import extract_embeddings_for_phonemes
from workflows.mlp_control_workflow.s3_classifier_encoder import classifier_encoder
from workflows.mlp_control_workflow.s4_fine_tune_on_noisy import fine_tune_classifier_on_noisy
from workflows.mlp_control_workflow.s5_visualize_results import visualize_results
from workflows.mlp_control_workflow.s6_confusion_pairs import analyze_confusion
from workflows.mlp_control_workflow.s7_batch_test_phonemes import batch_test_phonemes
from workflows.mlp_control_workflow.s8_benchmark_inference_and_save import benchmark_and_save
from workflows.mlp_control_workflow.s9_trace_mlp_classifier import trace_mlp_classifier
from workflows.mlp_control_workflow.s10_onnx_export import onnx_export
from workflows.mlp_control_workflow.s11_onnx_test import onnx_batch_test
from workflows.mlp_control_workflow.s12_overwrite_onnx_unreal import overwrite_onnx_unreal

ORGANIZED_RECORDINGS_DIR = "../../dist/organized_recordings"
PHONEME_EMBEDDINGS_DIR = '../../dist/phoneme_embeddings'
ORGANIZED_RECORDINGS_LOW_QUALITY = "../../dist/organized_recordings_lower_quality"

CLEAN_MODEL_PATH = "../../dist/phoneme_classifier.pkl"
LABEL_ENCODER_PATH = "../../dist/label_encoder.pkl"
PHONEME_LABELS_JSON_PATH = "../../dist/phoneme_labels.json"


def main():
    print(f"🕐 Workflow started at: {timestamp}\n")

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
        input_dir=ORGANIZED_RECORDINGS_DIR,
        output_dir=PHONEME_EMBEDDINGS_DIR,
        phoneme_label_json_path=PHONEME_LABELS_JSON_PATH,
    )


def classifier_encoder_clean():
    classifier_encoder(
        input_dir_str=PHONEME_EMBEDDINGS_DIR,
        classifier_out=CLEAN_MODEL_PATH,
        encoder_out=LABEL_ENCODER_PATH
    )


def visualize_results_clean():
    # TODO: This should be getting the metadata.csv, but I think its saving it during organized_recordings. Is this accurate? Probably, but then we need to pass it into visualize_results
    visualize_results(
        classifier_path=CLEAN_MODEL_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
        type_name="initial",
        embeddings_dir=PHONEME_EMBEDDINGS_DIR,
    )


def analyze_confusion_clean():
    analyze_confusion(
        classifier_path=CLEAN_MODEL_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
        embeddings_dir=PHONEME_EMBEDDINGS_DIR,
    )


def batch_test_phonemes_clean():
    batch_test_phonemes(
        classifier_path=CLEAN_MODEL_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
        phoneme_label_json_path=PHONEME_LABELS_JSON_PATH,
        training_recordings_path=ORGANIZED_RECORDINGS_DIR,
        finetuned_recordings_path=ORGANIZED_RECORDINGS_LOW_QUALITY,
    )

if __name__ == "__main__":
    main()
