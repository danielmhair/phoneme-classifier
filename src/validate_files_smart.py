import scipy
import librosa
import os
import numpy as np
import resampy
import soundfile as sf
import pygame

# Constants
AUDIO_DIR = "recordings"
MIN_REQUIRED_SAMPLES = int(0.3 * 16000)
MAX_ALLOWED_SAMPLES = int(2.0 * 16000)

# Init audio
pygame.mixer.init()

def play_numpy_audio(audio, sr):
    sf.write("temp.wav", audio, sr)
    pygame.mixer.music.load("temp.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def manual_decision(path, audio, sr):
    while True:
        action = input("✅ [y] keep / ❌ [n] mark bad / 🔁 [r] replay: ").strip().lower()
        if action == "y":
            return "y"
        elif action == "n":
            os.rename(path, os.path.join(os.path.dirname(path), "n_" + os.path.basename(path)))
            return "n"
        elif action == "r":
            play_numpy_audio(audio, sr)
        else:
            print("Please press 'y', 'n', or 'r'.")

def trim_silence_simple(audio, threshold_db=-40, frame_size=2048, hop_size=512):
    """
    Trim leading/trailing silence from a NumPy audio array based on an RMS energy threshold.
    """
    threshold = 10 ** (threshold_db / 20)
    rms = np.array([
        np.sqrt(np.mean(audio[i:i+frame_size]**2))
        for i in range(0, len(audio) - frame_size, hop_size)
    ])

    above_thresh = np.where(rms > threshold)[0]
    if len(above_thresh) == 0:
        return np.array([])  # all silence

    start = above_thresh[0] * hop_size
    end = min(len(audio), above_thresh[-1] * hop_size + frame_size)
    return audio[start:end]

def trim_silence(audio, top_db=40):
    """
    Trims leading and trailing silence from a NumPy audio array.
    - top_db: The threshold (in decibels) below reference to consider as silence.
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def high_pass_filter(audio, sr=16000, cutoff=100):
    """Apply a high-pass filter to remove mic bumps and low-frequency noise."""
    b, a = scipy.signal.butter(N=2, Wn=cutoff / (sr / 2), btype='high')
    return scipy.signal.filtfilt(b, a, audio)

def validate_and_review(path):
    audio, sr = sf.read(path)

    if np.max(np.abs(audio)) < 0.01:
        print(f"⚠️ No significant sound: {path}")
        play_numpy_audio(audio, sr)
        return manual_decision(path, audio, sr)

    # Step 1: Resample
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)
        sr = 16000

    # Step 2: Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Step 3: High-pass filter to remove mic bumps
    audio = high_pass_filter(audio, sr)

    # Step 4: Trim silence
    trimmed_audio = trim_silence_simple(audio)

    # Step 5: Length checks (on trimmed)
    if len(trimmed_audio) < MIN_REQUIRED_SAMPLES:
        print(f"⚠️ Too short after trimming: {path}")
        
        if len(trimmed_audio) == 0:
            print("⚠️ Skipping playback – no audio after trimming.")
            os.rename(path, os.path.join(os.path.dirname(path), "n_" + os.path.basename(path)))
            return "n"
        else:
            play_numpy_audio(trimmed_audio, sr)

        return manual_decision(path, audio, sr)  # Use original audio for review

    if len(trimmed_audio) > MAX_ALLOWED_SAMPLES:
        print(f"⚠️ Too long after trimming: {path}")
        play_numpy_audio(trimmed_audio, sr)
        return manual_decision(path, trimmed_audio, sr)

    return "y"


def review_all():
    for root, _, files in os.walk(AUDIO_DIR):
        for file in sorted(files):
            if not file.endswith(".wav") or file.startswith("n_"):
                continue
            fpath = os.path.join(root, file)
            result = validate_and_review(fpath)
            if result == "y":
                print(f"✅ Kept: {file}")
            elif result == "n":
                print(f"❌ Marked bad: {file}")

if __name__ == "__main__":
    review_all()
