import os
import soundfile as sf
import numpy as np
from pathlib import Path
import librosa
from tqdm import tqdm

# TODO: We can probably move this to the w1_prepare_wav_files.py script so we can have those when preparing the classifier
# Make sure to install: pip install librosa tqdm

# Parameters
SOURCE_DIR = Path("organized_recordings")
AUGMENTED_DIR = Path("augmented_recordings")
AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

# Define phonemes to augment more aggressively based on W11
TARGET_PHONEMES = {"th", "f", "h", "t", "p", "m", "n", "ng", "v", "dh", "k"}

# Augmentation variants
AUGMENTATIONS = [
    ("pitch_up", lambda y, sr: librosa.effects.pitch_shift(y, sr, n_steps=2)),
    ("pitch_down", lambda y, sr: librosa.effects.pitch_shift(y, sr, n_steps=-2)),
    ("speed_up", lambda y, sr: librosa.effects.time_stretch(y, rate=1.1)),
    ("slow_down", lambda y, sr: librosa.effects.time_stretch(y, rate=0.9)),
    ("add_noise", lambda y, sr: y + 0.005 * np.random.randn(len(y)))
]


def augment_phoneme_folder(phoneme):
    source_path = SOURCE_DIR / phoneme
    target_path = AUGMENTED_DIR / phoneme
    target_path.mkdir(parents=True, exist_ok=True)

    for wav_file in tqdm(list(source_path.glob("*.wav")), desc=f"Augmenting '{phoneme}'"):
        y, sr = sf.read(str(wav_file))

        # Save original
        base_name = wav_file.stem
        sf.write(target_path / f"{base_name}_orig.wav", y, sr)

        # Apply augmentations
        for aug_name, aug_func in AUGMENTATIONS:
            try:
                y_aug = aug_func(y.copy(), sr)
                # Ensure clipping doesn't occur
                y_aug = np.clip(y_aug, -1.0, 1.0)
                sf.write(target_path / f"{base_name}_{aug_name}.wav", y_aug, sr)
            except Exception as e:
                print(f"  ⚠️ Skipped {aug_name} for {wav_file.name}: {e}")


def main():
    phoneme_dirs = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir() and d.name in TARGET_PHONEMES]
    for phoneme in phoneme_dirs:
        augment_phoneme_folder(phoneme)

    print("✅ Augmentation complete. Files saved to 'augmented_recordings'.")


if __name__ == "__main__":
    main()
