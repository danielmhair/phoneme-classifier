import json
import os
import soundfile as sf
import numpy as np
import onnxruntime as ort
from collections import defaultdict

def onnx_batch_test():
    # Test both high-quality and low-quality sets
    test_sets = [
        # ("High-quality (organized_recordings)", "dist/organized_recordings"),
        ("Low-quality (low_quality_recordings)", "recordings_lower_quality"),
        ("Low-quality (low_quality_recordings)", "recordings_lower_quality_2"),
        ("Low-quality (low_quality_recordings)", "recordings_lowest_quality_1"),
    ]

    # Load ONNX Runtime sessions
    sess_w2v = ort.InferenceSession("dist/wav2vec2.onnx")
    sess_mlp = ort.InferenceSession("dist/phoneme_mlp.onnx")

    # Load phoneme labels
    with open("dist/phoneme_labels.json", "r") as f:
        phoneme_labels = json.load(f)

    for set_name, folder_path in test_sets:
        if not os.path.exists(folder_path):
            print(f"Skipping {set_name}: folder {folder_path} does not exist.")
            continue
        # Gather all wav files
        wav_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        if not wav_files:
            print(f"No WAV files found in {folder_path}")
            continue
        print(f"\n=== ONNX Testing {set_name}: {len(wav_files)} files ===")
        phoneme_counts = {phoneme: 0 for phoneme in phoneme_labels}
        total_counts = {phoneme: 0 for phoneme in phoneme_labels}
        i = 0
        all_confusion = []
        for wav_file in wav_files:
            try:
                audio, sr = sf.read(wav_file)
                if sr != 16000:
                    print(f"Warning: {wav_file} has sample rate {sr}, expected 16000. Skipping.")
                    continue
            except Exception as e:
                print(f"Error loading {wav_file}: {e}")
                continue
            # ONNX pipeline
            emb = sess_w2v.run([
                "embedding"], {"audio": audio.astype(np.float32)[None, :]}
            )[0]
            onnx_probs = sess_mlp.run([
                "phoneme_probs"], {"embedding": emb}
            )[0]
            pred_idx = int(np.argmax(onnx_probs))
            if pred_idx < 0 or pred_idx >= len(phoneme_labels):
                print(f"[ERROR] pred_idx {pred_idx} out of range for phoneme_labels (len={len(phoneme_labels)}). onnx_probs={onnx_probs}")
                continue
            predicted_phoneme = phoneme_labels[pred_idx]
            expected_phoneme = os.path.basename(os.path.dirname(wav_file))
            all_confusion.append((expected_phoneme, predicted_phoneme))
            
            # If expected_phoneme is not in total_counts, add it (handles unseen/extra phonemes)
            if expected_phoneme not in total_counts:
                total_counts[expected_phoneme] = 0
            total_counts[expected_phoneme] += 1

            if predicted_phoneme not in phoneme_counts:
                phoneme_counts[predicted_phoneme] = 0
            if expected_phoneme == predicted_phoneme:
                phoneme_counts[predicted_phoneme] += 1
            if (i % 500 == 0):
                print(f"Testing files - Currently {i}/{len(wav_files)} ({i/len(wav_files) * 100:.2f}%)...")
            i += 1
        print(f"\nONNX Results for {set_name}:")
        for phoneme in phoneme_labels:
            correct = phoneme_counts.get(phoneme, 0)
            total = total_counts.get(phoneme, 0)
            acc = (correct / total) * 100 if total > 0 else 0
            print(f"  {phoneme}: {correct}/{total} - {acc:.2f}%")
        
        # Overall confusion summary (all files)
        overall_confusion = defaultdict(int)
        for exp, pred in all_confusion:
            if exp != pred:
                overall_confusion[(exp, pred)] += 1
        if overall_confusion:
            print("\nOverall confusion pairs (all files):")
            for (exp, pred), count in sorted(overall_confusion.items(), key=lambda x: -x[1]):
                print(f"  {exp} → {pred}: {count}")
        overall = sum(phoneme_counts.values()) / max(1, sum(total_counts.values()))
        print(f"Overall accuracy: {overall*100:.2f}%")

if __name__ == "__main__":
    onnx_batch_test()
