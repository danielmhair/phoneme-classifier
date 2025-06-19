from pathlib import Path
import sys
import os
import time
from datetime import datetime

class Logger:
    def __init__(self, log_path):
        self.terminal = sys.__stdout__  # Preserve original terminal
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message) # type: ignore
        self.log.write(message)

    def flush(self):
        self.terminal.flush() # type: ignore
        self.log.flush()

# Create logs/ directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = time.time()
log_file_path = f"logs/workflow_log_{timestamp}.txt"

# Redirect stdout and stderr to both terminal and log file
sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout

from s0_cleanup import cleanup
from s0b_augment_audio import augment_audio
from s1_prepare_wav_files import prepare_wav_files, save_metadata, clean_previous_recordings
from s2_extract_embeddings_for_phonemes import extract_embeddings_for_phonemes
from s3_classifier_encoder import classifier_encoder
from s4_fine_tune_on_noisy import fine_tune_classifier_on_noisy
from s5_visualize_results import visualize_results
from s6_confusion_pairs import analyze_confusion
from s7_batch_test_phonemes import batch_test_phonemes
from s8_benchmark_inference_and_save import benchmark_and_save
from s9_trace_mlp_classifier import trace_mlp_classifier
from s10_onnx_export import onnx_export
from s11_onnx_test import onnx_batch_test
from s12_overwrite_onnx_unreal import overwrite_onnx_unreal

def main():
    print(f"🕐 Workflow started at: {timestamp}\n")

    RECORDINGS_DIR = "recordings"
    AUGMENTED_RECORDINGS_DIR = "dist/augmented_recordings"

    RECORDINGS_LOWER_QUALITY_DIRS = [
        "recordings_lower_quality",
        "recordings_lower_quality_2",
        "recordings_lowest_quality_1",
    ]

    
    ORGANIZED_RECORDINGS_DIR = "dist/organized_recordings"
    PHONEME_EMBEDDINGS_DIR = 'dist/phoneme_embeddings'
    ORGANIZED_RECORDINGS_LOW_QUALITY = "dist/organized_recordings_lower_quality"

    FINETUNED_EMBEDDINGS_DIR = 'dist/phoneme_embeddings_lower_quality'
    FINETUNED_MODEL_PATH = 'dist/phoneme_classifier_finetuned.pkl'

    CLEAN_MODEL_PATH = "dist/phoneme_classifier.pkl"
    LABEL_ENCODER_PATH = "dist/label_encoder.pkl"
    PHONEME_LABELS_JSON_PATH = "dist/phoneme_labels.json"

    def cleanup_dist():
        cleanup(folders=["dist"])

    def prepare_wav_files_clean():
        clean_previous_recordings(ORGANIZED_RECORDINGS_DIR)

        if not Path(RECORDINGS_DIR).exists():
            print(f"⚠️ Warning: {RECORDINGS_DIR} does not exist. Skipping.")
            raise Exception(f"Source directory {RECORDINGS_DIR} does not exist.")

        metadata = []
        new_metadata = prepare_wav_files(
            source_dir=RECORDINGS_DIR,
            target_dir=ORGANIZED_RECORDINGS_DIR,
            clean=False  # Don't delete between merges
        )
        metadata.extend(new_metadata)

        augment_audio(
            input_root=ORGANIZED_RECORDINGS_DIR,
            output_root=AUGMENTED_RECORDINGS_DIR,
            noise_path="recordings/silence.wav",
            noise_on_original_pct=0.4,
            noise_on_augmented_pct=0.2,
            noise_reduction_range=(10, 25)  # SNR control: dB range for how quiet to make noise
        )

        all_source_dirs = [
            *RECORDINGS_LOWER_QUALITY_DIRS,  # noisy mics
            AUGMENTED_RECORDINGS_DIR,  # augmented
        ]

        for rec_dir in all_source_dirs:
            if not Path(rec_dir).exists():
                print(f"⚠️ Warning: {rec_dir} does not exist. Skipping.")
                continue
            new_metadata = prepare_wav_files(
                source_dir=rec_dir,
                target_dir=ORGANIZED_RECORDINGS_DIR,
                clean=False  # Don't delete between merges
            )
            metadata.extend(new_metadata)

        save_metadata(metadata, ORGANIZED_RECORDINGS_DIR)
        print(f"✅ Prepared {len(metadata)} recordings in {ORGANIZED_RECORDINGS_DIR}.")
        print(f"✅ Metadata saved to {ORGANIZED_RECORDINGS_DIR}/metadata.csv.")
    
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

    for label, func, *params in steps:
        print(f"\n🚀 Starting: {label}...")
        step_start = time.time()
        if params:
            func(*params)
        else:
            func()
        step_end = time.time()
        duration = step_end - step_start
        print(f"✅ Finished: {label} in {duration:.2f} seconds\n")

    total_time = time.time() - start_time
    print("\n\n🏁 All steps completed successfully. 🏁")
    print("You can test phonemes through the command line with w5_record_voice_cli.py")
    print(f"✅✅ Workflow complete! Total time: {total_time:.2f} seconds ✅✅")

if __name__ == "__main__":
    main()
