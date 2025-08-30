#!/usr/bin/env python3
"""
Epic 1 Style Validation Test for Epic 2 Models

Tests Epic 2 model loader using Epic 1's methodology:
- 1-second recordings (like Epic 1)
- Complete audio processing
- All three models comparison

This isolates whether accuracy issues are from:
1. Model/ONNX export problems, or  
2. Real-time streaming chunk size limitations
"""
import sys
import os
import time
import numpy as np
import sounddevice as sd

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from inference.cli.model_loader import ModelLoader
from inference.cli.audio_feature_extractor import AudioFeatureExtractor

# Epic 1 Constants
SAMPLE_RATE = 16000
DURATION = 1.0  # Epic 1 used 1-second clips
MIN_AMP_THRESHOLD = 0.01
BUFFER_SAMPLES = 1000


def record_audio_clip(duration=DURATION, fs=SAMPLE_RATE):
    """Record audio clip using Epic 1 methodology."""
    print(f"🎤 Recording for {duration} seconds. Speak clearly...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def process_audio_epic1_style(audio):
    """Process audio using Epic 1 methodology."""
    # Find non-silent regions
    nonzero_indices = np.where(np.abs(audio) > MIN_AMP_THRESHOLD)[0]
    if len(nonzero_indices) == 0:
        return None
    
    # Trim silence with buffer (Epic 1 approach)
    start = max(0, nonzero_indices[0] - BUFFER_SAMPLES)
    end = min(len(audio), nonzero_indices[-1] + BUFFER_SAMPLES)
    trimmed = audio[start:end]
    
    # Normalize (Epic 1 approach)
    max_val = np.max(np.abs(trimmed))
    if max_val > 0:
        trimmed = trimmed / max_val
    
    return trimmed.astype(np.float32)


def test_model_accuracy(model_id: str, audio_clip: np.ndarray):
    """Test a specific model with Epic 1 methodology."""
    model_loader = ModelLoader()
    feature_extractor = AudioFeatureExtractor()
    
    print(f"\n🧪 Testing {model_id.upper()} Model")
    print("=" * 40)
    
    # Load model
    if not model_loader.load_model(model_id):
        print(f"❌ Failed to load {model_id}")
        return None
    
    try:
        # Extract features using Epic 2's feature extractor
        feature_result = feature_extractor.extract_features_for_model(
            audio_clip, 
            model_loader.current_model_info.model_type
        )
        
        if not feature_result['preprocessing_success']:
            print(f"❌ Feature extraction failed: {feature_result.get('error', 'Unknown')}")
            return None
        
        print(f"✅ Audio processed: {len(audio_clip)} samples → {feature_result['audio_length']} samples")
        print(f"✅ Features extracted: {feature_result['feature_type']}")
        
        # Run model inference
        probabilities = model_loader.run_inference(feature_result['features'])
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        print(f"\n🎯 Top 5 Predictions:")
        for i, idx in enumerate(top_indices):
            phoneme = model_loader.current_labels[idx]
            confidence = probabilities[idx]
            print(f"  {i+1}. {phoneme}: {confidence:.3f} ({confidence*100:.1f}%)")
        
        # Return top prediction
        top_phoneme = model_loader.current_labels[top_indices[0]]
        top_confidence = probabilities[top_indices[0]]
        
        return {
            'phoneme': top_phoneme,
            'confidence': top_confidence,
            'all_probabilities': probabilities
        }
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None


def main():
    """Main validation test comparing Epic 1 vs Epic 2 approaches."""
    print("🧪 Epic 1 Style Validation Test")
    print("=" * 50)
    print("Testing Epic 2 models using Epic 1 methodology")
    print("(1-second clips vs. 64ms streaming chunks)")
    print()
    
    models_to_test = ['mlp_control', 'wav2vec2_ctc', 'wavlm_ctc']
    
    while True:
        try:
            phoneme_said = input("\n📝 What phoneme will you say? (e.g., 'b', 'ah', 's') [Enter to record, Ctrl+C to exit]: ").strip().upper()
            if not phoneme_said:
                phoneme_said = "UNKNOWN"
        except KeyboardInterrupt:
            print("\n🛑 Exiting...")
            break
        
        # Record using Epic 1 methodology
        audio_clip = record_audio_clip()
        processed_audio = process_audio_epic1_style(audio_clip)
        
        if processed_audio is None:
            print("❌ No significant audio detected. Try again.")
            continue
        
        print(f"\n✅ Audio captured: {len(processed_audio)} samples ({len(processed_audio)/SAMPLE_RATE:.2f}s)")
        print(f"📝 Expected phoneme: {phoneme_said}")
        
        # Test all models
        results = {}
        for model_id in models_to_test:
            result = test_model_accuracy(model_id, processed_audio)
            if result:
                results[model_id] = result
        
        # Summary
        print(f"\n📊 ACCURACY COMPARISON")
        print("=" * 40)
        for model_id, result in results.items():
            correct = "✅" if result['phoneme'].upper() == phoneme_said else "❌"
            print(f"{correct} {model_id.upper()}: {result['phoneme']} ({result['confidence']:.3f})")
        
        print(f"\n💡 Expected: {phoneme_said}")
        print("🔄 Testing next phoneme...\n")


if __name__ == "__main__":
    main()