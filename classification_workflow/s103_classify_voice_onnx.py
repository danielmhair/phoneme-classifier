import json
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from transformers import Wav2Vec2Processor

# Constants
SAMPLE_RATE = 16000
DURATION = 1.0
MIN_AMP_THRESHOLD = 0.01
BUFFER_SAMPLES = 1000

# Load phoneme labels
with open("dist/phoneme_labels.json", "r", encoding="utf-8") as f:
    phoneme_labels = json.load(f)

# Load Wav2Vec2 processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Load ONNX sessions
wav2vec_session = ort.InferenceSession("dist/wav2vec2.onnx")
mlp_session = ort.InferenceSession("dist/phoneme_mlp.onnx")

def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration} seconds. Please speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def process_audio(audio):
    nonzero_indices = np.where(np.abs(audio) > MIN_AMP_THRESHOLD)[0]
    if len(nonzero_indices) == 0:
        return None
    start = max(0, nonzero_indices[0] - BUFFER_SAMPLES)
    end = min(len(audio), nonzero_indices[-1] + BUFFER_SAMPLES)
    trimmed = audio[start:end]
    return trimmed / np.max(np.abs(trimmed)) if np.max(np.abs(trimmed)) > 0 else trimmed

def extract_embedding_onnx(audio_np):
    print("raw audio shape:", audio_np.shape)
    inputs = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="np", padding=True) # type:ignore
    print("processed input shape:", inputs['input_values'].shape)

    ort_inputs = {wav2vec_session.get_inputs()[0].name: inputs['input_values']}
    outputs = wav2vec_session.run(None, ort_inputs)
    print("wav2vec output count:", len(outputs))
    print("wav2vec output shape:", outputs[0].shape)

    embedding = outputs[0]  # [1, T, H]
    # embedding = last_hidden.mean(axis=1)  # [1, H]
    print("embedding shape (before cast):", embedding.shape)
    return embedding.astype(np.float32)


def predict_phoneme(embedding):
    ort_inputs = {mlp_session.get_inputs()[0].name: embedding}
    logits = mlp_session.run(None, ort_inputs)[0]  # shape: [1, num_classes]
    logits = logits[0]  # shape: [num_classes]
    pred_index = int(np.argmax(logits))
    return phoneme_labels[pred_index], logits


def main():
    print("🎙️ ONNX Voice Classifier")
    while True:
        try:
            input("Press Enter to record...")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        audio = record_audio()
        processed = process_audio(audio)
        if processed is None:
            print("No significant audio. Try again.")
            continue

        print("Extracting features...")
        embedding = extract_embedding_onnx(processed)
        print("Classifying phoneme...")
        print("MLP input shape:", embedding.shape)
        pred_label, logits = predict_phoneme(embedding)

        print("Logits (confidence):")
        for i, (label, score) in enumerate(zip(phoneme_labels, logits)):
            print(f" [{i:02}] {label}: {score:.2f}")
        print(f"\nPredicted phoneme: 🔤 {pred_label}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
