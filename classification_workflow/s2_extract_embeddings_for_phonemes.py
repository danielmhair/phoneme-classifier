import json
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from pathlib import Path
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from audiomentations import Compose, AddGaussianNoise, HighPassFilter, LowPassFilter, Normalize

# Define augmentation pipeline for simulating bad mic conditions
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
    HighPassFilter(min_cutoff_freq=100.0, max_cutoff_freq=300.0, p=0.4),
    LowPassFilter(min_cutoff_freq=3000.0, max_cutoff_freq=8000.0, p=0.4),
    Normalize(p=1.0),
])

def extract_embeddings_for_phonemes(input_dir="organized_recordings", output_dir="phoneme_embeddings", phoneme_label_json_path="dist/phoneme_labels.json"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = input_dir / "metadata.csv"
    meta = pd.read_csv(metadata_csv)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
    phoneme_dirs = list(input_dir.iterdir())
    print(f"Found {len(phoneme_dirs)} subdirectories in {input_dir}")

    phoneme_labels = sorted([d.name for d in phoneme_dirs if d.is_dir()])
    with open(phoneme_label_json_path, "w") as f:
        json.dump(phoneme_labels, f)
    print(f"✅ Saved phoneme label order to dist/phoneme_labels.json with {len(phoneme_labels)} labels.")

    embeddings = []
    labels = []
    for _, row in meta.iterrows():
        if len(embeddings) % 500 == 0:
            print(f"Adding embedding (.npy) files ({len(embeddings)} so far)...")

        wav_path = input_dir / row["phoneme"] / row["new_filename"]
        if not wav_path.exists():
            continue
        audio, sr = sf.read(str(wav_path))
        if sr != 16000:
            continue
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True) # type:ignore
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        emb_filename = str(row["new_filename"])
        if emb_filename.endswith(".wav"):
            emb_filename = emb_filename[:-4] + ".npy"
        np.save(output_dir / emb_filename, emb)
        embeddings.append(emb)
        labels.append(row["phoneme"])
    
    print(f"✅ Added {len(embeddings)} embedding files (.npy)")
    # Save new metadata.csv in output_dir
    meta_out = meta.copy()
    meta_out["embedding_filename"] = meta_out["new_filename"].astype(str).str.replace(".wav", ".npy")
    print(f"✅ Saving metadata to {output_dir / 'metadata.csv'}")
    meta_out.to_csv(output_dir / "metadata.csv", index=False)

    if (len(embeddings) > 0):
        # Save embeddings as a numpy array
        embeddings_array = np.array(embeddings)
        np.save(output_dir / "embeddings.npy", embeddings_array)
        print(f"✅ Saved embeddings to {output_dir / 'embeddings.npy'}")
        print(f"✅ Embedding extraction complete for {input_dir} → {output_dir}")
    else:
        print("⚠️ No valid embeddings found. Check your input directory and audio files.")
        raise Exception("No valid embeddings found. Check your input directory and audio files.")
