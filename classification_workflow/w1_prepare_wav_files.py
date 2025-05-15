import os
import shutil
import csv
from pathlib import Path

# Set source and target directories
SOURCE_DIR = Path("recordings")
TARGET_DIR = Path("organized_recordings")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import soundfile as sf

def trim_and_normalize(audio, threshold=0.01, buffer=1000):
    nonzero_indices = np.where(np.abs(audio) > threshold)[0]
    if len(nonzero_indices) == 0:
        return None
    start = max(0, nonzero_indices[0] - buffer)
    end = min(len(audio), nonzero_indices[-1] + buffer)
    trimmed = audio[start:end]
    max_amp = np.max(np.abs(trimmed))
    return trimmed / max_amp if max_amp > 0 else trimmed

def prepare_wav_files():
    if not SOURCE_DIR.exists():
        print(f"Error: {SOURCE_DIR} does not exist. You must have that folder to begin.")
        return

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
        print(f"✅ Cleaned previous recordings in {TARGET_DIR}.")
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory {TARGET_DIR} for saving organized recordings.")
    
    # Prepare metadata list
    metadata = []

    # Iterate over child folders in the source directory
    for child_dir in SOURCE_DIR.iterdir():
        if child_dir.is_dir():
            child_name = child_dir.name
            # Iterate over phoneme (letter) folders within each child folder
            for phoneme_dir in child_dir.iterdir():
                if phoneme_dir.is_dir():
                    phoneme = phoneme_dir.name.lower()  # the letter (phoneme)
                    # Create target subdirectory for this phoneme if it doesn't exist
                    target_subdir = TARGET_DIR / phoneme
                    target_subdir.mkdir(parents=True, exist_ok=True)
                    
                    # Counter for filenames per phoneme per child
                    counter = 1
                    # Iterate over all WAV files in the current phoneme folder
                    for wav_file in phoneme_dir.glob("*.wav"):
                        # Create a new filename: childname_phoneme_counter.wav
                        new_filename = f"{child_name}_{phoneme}_{counter:03d}.wav"
                        target_path = target_subdir / new_filename
                        
                        # Load and process the audio
                        audio, sr = sf.read(str(wav_file))
                        if sr != 16000:
                            print(f"⚠️ Skipping {wav_file.name}: expected 16kHz sample rate, got {sr}")
                            continue

                        processed = trim_and_normalize(audio)
                        if processed is None or len(processed) < 1000:
                            print(f"⚠️ Skipping {wav_file.name}: silent or too short after trimming.")
                            continue

                        # Save the cleaned audio to the target location
                        sf.write(str(target_path), processed, samplerate=sr)
                        
                        # Record metadata: new filename, child name, phoneme, original path
                        metadata.append([new_filename, child_name, phoneme, str(wav_file)])
                        
                        counter += 1

    # Save metadata CSV in the target directory
    metadata_csv_path = TARGET_DIR / "metadata.csv"
    with open(metadata_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["new_filename", "child_name", "phoneme", "original_file"])
        writer.writerows(metadata)

    print(f"✅ Reorganization complete! Files have been organized into '{TARGET_DIR}'.")
    print(f"Metadata CSV saved at: {metadata_csv_path}")

if __name__ == "__main__":
    prepare_wav_files()