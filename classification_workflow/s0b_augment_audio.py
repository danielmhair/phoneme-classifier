
from pydub import AudioSegment
import random
import os
import random
import librosa
import soundfile as sf
from pydub import AudioSegment

def augment_audio(
    input_root="organized_recordings",
    output_root="augmented_recordings",
    noise_path="recordings/silence.wav",
    noise_on_original_pct=0.4,
    noise_on_augmented_pct=0.2,
    noise_reduction_range=(10, 25)  # SNR control: dB range for how quiet to make noise
):
    os.makedirs(output_root, exist_ok=True)

    pitch_shifts = {
        "pitch_up": 2,
        "pitch_down": -2,
    }

    speed_changes = {
        "speed_0.9x": 0.9,
        "speed_1.1x": 1.1,
    }

    # Load noise and normalize
    noise_audio = AudioSegment.from_wav(noise_path)

    for dirpath, _, filenames in os.walk(input_root):
        print(f"Augmenting from directory: {dirpath}")
        for filename in filenames:
            if not filename.lower().endswith(".wav"):
                continue

            input_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(dirpath, input_root)
            output_dir = os.path.join(output_root, "aug", rel_path)
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(filename)[0]
            sound = AudioSegment.from_wav(input_path)
            y, sr = librosa.load(input_path, sr=None)

            # --- Add noise to original ---
            if random.random() < noise_on_original_pct:
                noisy = overlay_noise(sound, noise_audio, noise_reduction_db=random.randint(*noise_reduction_range))
                noisy_output = os.path.join(output_dir, f"{base_name}_orig_noisy.wav")
                noisy.export(noisy_output, format="wav")

            # --- Speed Augmentations ---
            for label, factor in speed_changes.items():
                altered = change_speed(sound, factor)
                aug_output = os.path.join(output_dir, f"{base_name}_{label}.wav")
                altered.export(aug_output, format="wav")

                # Optionally add noise
                if random.random() < noise_on_augmented_pct:
                    noisy = overlay_noise(altered, noise_audio, noise_reduction_db=random.randint(*noise_reduction_range))
                    noisy_output = os.path.join(output_dir, f"{base_name}_{label}_noisy.wav")
                    noisy.export(noisy_output, format="wav")

            # --- Pitch Augmentations ---
            for label, steps in pitch_shifts.items():
                y_shifted = change_pitch(y, sr, n_steps=steps)
                pitch_output = os.path.join(output_dir, f"{base_name}_{label}.wav")
                sf.write(pitch_output, y_shifted, sr)

                # Optionally add noise
                if random.random() < noise_on_augmented_pct:
                    # Convert back to AudioSegment for noise mixing
                    temp_path = pitch_output + "_temp.wav"
                    sf.write(temp_path, y_shifted, sr)
                    pitch_seg = AudioSegment.from_wav(temp_path)
                    noisy = overlay_noise(pitch_seg, noise_audio, noise_reduction_db=random.randint(*noise_reduction_range))
                    noisy_output = os.path.join(output_dir, f"{base_name}_{label}_noisy.wav")
                    noisy.export(noisy_output, format="wav")
                    os.remove(temp_path)

    print("✅ Augmentation complete with speed, pitch, and realistic noise.")


def change_speed(sound, speed=1.0):
    new_frame_rate = int(sound.frame_rate * speed)
    return sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(sound.frame_rate)


def change_pitch(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def overlay_noise(clean_audio: AudioSegment, background_noise: AudioSegment, noise_reduction_db: int = 18) -> AudioSegment:
    clean_duration = len(clean_audio)
    bg_duration = len(background_noise)

    if bg_duration > clean_duration:
        max_start = bg_duration - clean_duration
        start = random.randint(0, max_start)
        noise_slice = background_noise[start:start + clean_duration]
    else:
        loops = (clean_duration // bg_duration) + 1
        noise_slice = (background_noise * loops)[:clean_duration]

    # Ensure the type is AudioSegment
    if not isinstance(noise_slice, AudioSegment):
        raise TypeError(f"Expected AudioSegment, got {type(noise_slice)}")

    noise_slice = noise_slice - noise_reduction_db
    return clean_audio.overlay(noise_slice)
