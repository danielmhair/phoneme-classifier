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

from w0_cleanup import cleanup
from w1_prepare_wav_files import prepare_wav_files
from w2_extract_embeddings_for_phonemes import extract_embeddings_for_phonemes
from w3_embeddings import embeddings
from w4_batch_test_phonemes import batch_test_phonemes
from w6_benchmark_inference_and_save import benchmark_and_save
from w7_trace_mlp_classifier import trace_mlp_classifier
from w8_onnx_export import onnx_export
from w9_onnx_test import onnx_test
from w10_visualize_results import visualize_results
from w11_confusion_pairs import analyze_confusion
from w12_overwrite_onnx_unreal import overwrite_onnx_unreal

def main():
    print(f"🕐 Workflow started at: {timestamp}\n")

    steps = [
        # ("Cleanup previous runs", cleanup),
        # ("Prepare the dataset", prepare_wav_files),
        # ("Extract embeddings for phonemes", extract_embeddings_for_phonemes),
        # ("Extract embeddings", embeddings),
        # ("Batch test phonemes", batch_test_phonemes),
        # ("Benchmark inference and save", benchmark_and_save),
        # ("Trace MLP classifier", trace_mlp_classifier),
        # ("Visualize Results", visualize_results),
        # ("Export to ONNX", onnx_export),
        ("Test ONNX model", onnx_test),
        ("Analyze confusion matrix", analyze_confusion),
        ("Copy to Unreal Engine", overwrite_onnx_unreal)
    ]

    DIST_DIR = Path("dist")
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    for label, func in steps:
        print(f"\n🚀 Starting: {label}...")
        step_start = time.time()
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
    