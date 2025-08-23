from pathlib import Path
from workflows.shared.s0_cleanup import cleanup
from workflows.shared.s0b_augment_audio import augment_audio
from workflows.shared.s1_prepare_wav_files import prepare_wav_files, save_metadata, clean_previous_recordings

ROOT_DIR = "."

# Configuration
RECORDINGS_DIR = f"{ROOT_DIR}/recordings"
AUGMENTED_RECORDINGS_DIR = f"{ROOT_DIR}/dist/augmented_recordings"
RECORDINGS_LOWER_QUALITY_DIRS = [
    f"{ROOT_DIR}/recordings_lower_quality",
    f"{ROOT_DIR}/recordings_lower_quality_2",
    f"{ROOT_DIR}/recordings_lowest_quality_1",
]

ORGANIZED_RECORDINGS_DIR = f"{ROOT_DIR}/dist/organized_recordings"
PHONEME_LABELS_JSON_PATH = f"{ROOT_DIR}/dist/phoneme_labels.json"
SILENCE_WAV_FILE = f"{ROOT_DIR}/recordings/silence.wav"


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
        noise_path=SILENCE_WAV_FILE,
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
