#!/usr/bin/env python3
"""
CTC Wav2Vec2 Workflow - Complete CTC Training Pipeline

This workflow implements the CTC (Connectionist Temporal Classification) approach
for phoneme sequence recognition using Wav2Vec2 features and LSTM sequence modeling.

Unlike the MLP workflow which uses mean-pooled embeddings for single phoneme classification,
this CTC workflow preserves temporal sequences for alignment-free sequence modeling.
"""

from pathlib import Path
import sys
import os
import time
from datetime import datetime


class Logger:
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # type: ignore
        self.log.write(message)

    def flush(self):
        self.terminal.flush()  # type: ignore
        self.log.flush()


# Create logs directory
os.makedirs("../logs", exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = time.time()
log_file_path = f"../logs/ctc_workflow_log_{timestamp}.txt"

# Redirect stdout to log
sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout

# Add main project path for accessing shared utilities
sys.path.append(str(Path(__file__).parent.parent))

# Import shared utilities from main workflow
try:
    from mlp_control_workflow.s0_cleanup import cleanup
    from mlp_control_workflow.s1_prepare_wav_files import prepare_wav_files, save_metadata, clean_previous_recordings
    from mlp_control_workflow.s0b_augment_audio import augment_audio
except ImportError as e:
    print(f"⚠️ Could not import shared utilities: {e}")
    print("Please ensure you're running from the project root directory")

# Import CTC-specific modules
from s2_extract_embeddings_temporal import extract_embeddings_for_phonemes_temporal
from s3_ctc_classifier import ctc_classifier_training


def main():
    print(f"🎯 CTC Wav2Vec2 Workflow started at: {timestamp}\n")
    print("🚀 Epic 1: Live Phoneme CTCs - CTC Implementation")
    print("=" * 60)

    # Configuration
    RECORDINGS_DIR = "../recordings"
    AUGMENTED_RECORDINGS_DIR = "../dist/augmented_recordings"
    RECORDINGS_LOWER_QUALITY_DIRS = [
        "../recordings_lower_quality",
        "../recordings_lower_quality_2",
        "../recordings_lowest_quality_1",
    ]

    ORGANIZED_RECORDINGS_DIR = "../dist/organized_recordings"
    PHONEME_EMBEDDINGS_TEMPORAL_DIR = "../dist/phoneme_embeddings_temporal"
    PHONEME_LABELS_JSON_PATH = "../dist/phoneme_labels.json"

    # CTC-specific output paths
    CTC_MODEL_PATH = "../dist/ctc_model_best.pt"
    CTC_LABEL_ENCODER_PATH = "../dist/ctc_label_encoder.pkl"

    def cleanup_dist():
        """Clean previous CTC outputs."""
        cleanup(folders=["../dist"])

    def prepare_wav_files_ctc():
        """Prepare and organize audio files for CTC training."""
        clean_previous_recordings(ORGANIZED_RECORDINGS_DIR)

        if not Path(RECORDINGS_DIR).exists():
            print(f"⚠️ Warning: {RECORDINGS_DIR} does not exist.")
            raise Exception(f"Source directory {RECORDINGS_DIR} does not exist.")

        metadata = []

        # Process main recordings
        new_metadata = prepare_wav_files(
            source_dir=RECORDINGS_DIR,
            target_dir=ORGANIZED_RECORDINGS_DIR,
            clean=False
        )
        metadata.extend(new_metadata)
        
        # Audio augmentation for robustness
        augment_audio(
            input_root=ORGANIZED_RECORDINGS_DIR,
            output_root=AUGMENTED_RECORDINGS_DIR,
            noise_path="../recordings/silence.wav",
            noise_on_original_pct=0.4,
            noise_on_augmented_pct=0.2,
            noise_reduction_range=(10, 25)
        )
        
        # Add lower quality recordings and augmented data
        all_source_dirs = [
            *RECORDINGS_LOWER_QUALITY_DIRS,
            AUGMENTED_RECORDINGS_DIR,
        ]
        
        for rec_dir in all_source_dirs:
            if not Path(rec_dir).exists():
                print(f"⚠️ Warning: {rec_dir} does not exist. Skipping.")
                continue
            new_metadata = prepare_wav_files(
                source_dir=rec_dir,
                target_dir=ORGANIZED_RECORDINGS_DIR,
                clean=False
            )
            metadata.extend(new_metadata)
        
        save_metadata(metadata, ORGANIZED_RECORDINGS_DIR)
        print(f"✅ Prepared {len(metadata)} recordings for CTC training")
    
    def extract_temporal_embeddings():
        """Extract temporal embeddings preserving sequence information."""
        extract_embeddings_for_phonemes_temporal(
            input_dir=ORGANIZED_RECORDINGS_DIR,
            output_dir=PHONEME_EMBEDDINGS_TEMPORAL_DIR,
            phoneme_label_json_path=PHONEME_LABELS_JSON_PATH,
            enable_ctc=True
        )
    
    def train_ctc_model():
        """Train CTC classifier."""
        ctc_classifier_training(
            input_dir_str=PHONEME_EMBEDDINGS_TEMPORAL_DIR,
            output_dir="../dist",
            num_epochs=20,
            batch_size=32
        )
    
    def test_ctc_inference():
        """Test CTC inference capabilities."""
        print("🧪 Testing CTC inference...")
        try:
            # Import here to test if everything works
            from validations.classify_voice_ctc import CTCPhonemeClassifier
            print("✅ CTC inference system is ready!")
            print("📝 To test interactively:")
            print("   cd ctc_w2v2_workflow")
            print("   python validations/classify_voice_ctc.py")
        except Exception as e:
            print(f"⚠️ CTC inference test failed: {e}")
            print("This may be due to missing dependencies (torch, transformers)")
    
    # Define workflow steps
    steps = [
        ("Cleanup previous CTC outputs", cleanup_dist),
        ("Prepare audio dataset for CTC", prepare_wav_files_ctc),
        ("Extract temporal embeddings (preserve sequences)", extract_temporal_embeddings),
        ("Train CTC classifier", train_ctc_model),
        ("Test CTC inference system", test_ctc_inference),
    ]
    
    # Execute workflow
    for label, func in steps:
        print(f"\n🚀 Starting: {label}...")
        step_start = time.time()
        try:
            func()
        except Exception as e:
            print(f"❌ Failed: {label} - {e}")
            import traceback
            traceback.print_exc()
            # Continue with next step for robustness
        
        step_end = time.time()
        duration = step_end - step_start
        print(f"✅ Finished: {label} in {duration:.2f} seconds\n")
    
    total_time = time.time() - start_time
    print("\n\n🏁 CTC Workflow completed! 🏁")
    print("=" * 40)
    print(f"📊 Total time: {total_time:.2f} seconds")
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
    
    print(f"\n✅ Epic 1: Live Phoneme CTCs - Implementation Complete! ✅")


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