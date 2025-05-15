import os
import csv
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Paths
AUGMENTED_DIR = Path("augmented_recordings")
EMBEDDINGS_DIR = Path("phoneme_embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = EMBEDDINGS_DIR / "metadata.csv"

# Load processor and model once
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

def extract_embedding_from_audio(audio, sr):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def extract_augmented_embeddings():
    # Read existing metadata to avoid duplicates
    existing_files = set()
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_files.add(row["embedding_filename"])

    new_metadata = []

    for phoneme_dir in AUGMENTED_DIR.iterdir():
        if not phoneme_dir.is_dir():
            continue

        phoneme = phoneme_dir.name.lower()
        for wav_file in phoneme_dir.glob("*.wav"):
            embedding_filename = wav_file.stem + ".npy"
            if embedding_filename in existing_files:
                continue

            try:
                audio, sr = sf.read(str(wav_file))
                if sr != 16000:
                    print(f"Skipping {wav_file.name}: sample rate {sr} != 16000")
                    continue

                emb = extract_embedding_from_audio(audio, sr)
                np.save(EMBEDDINGS_DIR / embedding_filename, emb)
                new_metadata.append([wav_file.name, phoneme, embedding_filename])
            except Exception as e:
                print(f"⚠️ Error processing {wav_file.name}: {e}")

    if new_metadata:
        with open(METADATA_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if METADATA_PATH.stat().st_size == 0:
                writer.writerow(["wav_filename", "phoneme", "embedding_filename"])
            writer.writerows(new_metadata)

    print(f"\n✅ Extracted {len(new_metadata)} new embeddings from augmented data.")

if __name__ == "__main__":
    extract_augmented_embeddings()
