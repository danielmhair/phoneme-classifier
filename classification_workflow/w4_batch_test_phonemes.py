import json
import sys
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pickle

def load_audio(file_path, target_sr=16000):
    audio, sr = sf.read(file_path)
    if sr != target_sr:
        raise ValueError(f"Sample rate of {file_path} is {sr}, expected {target_sr}.")
    return audio

def extract_embedding(audio, processor, model):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Average over time dimension to get fixed-size vector
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def batch_test_phonemes():        
    folder_path = Path("organized_recordings")
    if not folder_path.exists():
        print(f"Folder {folder_path} does not exist.")
        sys.exit(1)
        
    # Load the Wav2Vec2 processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()
    
    # Load the trained classifier and label encoder
    with open("dist/phoneme_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("dist/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    
    # Iterate over all WAV files in the folder (recursively)
    wav_files = list(folder_path.glob("**/*.wav"))
    if not wav_files:
        print(f"No WAV files found in {folder_path}")
        sys.exit(0)
    
    print(f"Found {len(wav_files)} WAV files. Testing...")
    
    with open("dist/phoneme_labels.json", "r") as f:
        phoneme_labels = json.load(f)
    
    matching_labels = list(le.classes_) == phoneme_labels
    if not matching_labels:
        raise Exception("❌ LabelEncoder mismatch! Make sure phoneme_labels.json matches the training labels.")

    phoneme_counts = {phoneme: 0 for phoneme in phoneme_labels}
    total_counts = {phoneme: 0 for phoneme in phoneme_labels}

    i = 0
    for wav_file in wav_files:
        try:
            audio = load_audio(str(wav_file))
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue
        
        # Extract embedding from the audio file
        embedding = extract_embedding(audio, processor, model)
        embedding = embedding.reshape(1, -1)
        
        # Predict phoneme using classifier
        pred = clf.predict(embedding)
        pred_prob = clf.predict_proba(embedding)[0]
        predicted_phoneme = le.inverse_transform(pred)[0]
        
        # Print the result for the file
        expected_phoneme = wav_file.parent.name
        total_counts[expected_phoneme] += 1
        if str(predicted_phoneme) not in phoneme_counts:
            phoneme_counts[str(predicted_phoneme)] = 0
        if expected_phoneme == str(predicted_phoneme):
            phoneme_counts[str(predicted_phoneme)] += 1
        if (i % 500 == 0):
            print(f"Tested {i}/{len(wav_files)} files ({i/len(wav_files) * 100:.2f}%)...")
        i += 1

    print("\nFinished Tests!\nPhoneme counts:")
    for phoneme in phoneme_labels:
        correct = phoneme_counts.get(phoneme, 0)
        total = total_counts.get(phoneme, 0)
        acc = (correct / total) * 100 if total > 0 else 0
        print(f"  {phoneme}: {correct}/{total} - {acc:.2f}%")

if __name__ == "__main__":
    batch_test_phonemes()

