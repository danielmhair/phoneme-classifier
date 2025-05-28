import os
from pathlib import Path
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, HighPassFilter, LowPassFilter, Normalize
import numpy as np

def augment_audio(source_dirs: list[str], output_dir_str: str):
    """
    Augment audio files in the specified source directories and save them to the output directory.
    
    Args:
        source_dirs (list[str]): List of source directories containing phoneme subdirectories with .wav files.
        output_dir (str): Directory where augmented audio files will be saved.
    """
        
    # Augmentation pipeline
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
        HighPassFilter(min_cutoff_freq=100.0, max_cutoff_freq=300.0, p=0.8),
        LowPassFilter(min_cutoff_freq=3000.0, max_cutoff_freq=8000.0, p=0.8),
        Normalize(p=1.0),
    ])

    output_dir = Path(output_dir_str)

    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    for source_dir_str in source_dirs:
        source_dir = Path(source_dir_str)
        if not source_dir.exists():
            continue

        for phoneme_dir in source_dir.glob("*"):
            if not phoneme_dir.is_dir():
                continue

            phoneme = phoneme_dir.name
            target_phoneme_dir = output_dir / phoneme
            target_phoneme_dir.mkdir(parents=True, exist_ok=True)

            for wav_path in phoneme_dir.glob("*.wav"):
                try:
                    audio, sr = sf.read(str(wav_path))
                    if sr != 16000:
                        continue
                    augmented = augment(samples=audio, sample_rate=sr) # type:ignore
                    new_filename = wav_path.stem + "_aug.wav"
                    target_path = target_phoneme_dir / new_filename
                    sf.write(str(target_path), augmented, sr)
                    print(f"Saved: {target_path}")
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")
