import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
from collections import defaultdict

# Setup
BASE_DIR = "dist/organized_recordings"
EXPECTED_SAMPLE_RATE = 16000

# Load model and tokenizer
tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h-phoneme")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h-phoneme")
model.eval()

# Optional resampler if your files aren't 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=EXPECTED_SAMPLE_RATE)

# For tracking predictions
results = defaultdict(list)

for label in sorted(os.listdir(BASE_DIR)):
    label_path = os.path.join(BASE_DIR, label)
    if not os.path.isdir(label_path):
        continue
    for file in os.listdir(label_path):
        if not file.endswith(".wav"):
            continue
        filepath = os.path.join(label_path, file)
        waveform, sr = torchaudio.load(filepath)
        if sr != EXPECTED_SAMPLE_RATE:
            waveform = resampler(waveform)
        
        with torch.no_grad():
            input_values = tokenizer(waveform[0], return_tensors="pt").input_values
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_phonemes = tokenizer.batch_decode(pred_ids)[0]
        
        results[label].append(pred_phonemes)
        print(f"File: {file} | Label: {label} | Predicted: {pred_phonemes}")

# Summarize
print("\n--- Summary by Phoneme ---")
for true_phoneme, preds in results.items():
    unique_preds = set(p.strip() for p in preds)
    print(f"{true_phoneme}: {len(preds)} samples → Predictions: {unique_preds}")
