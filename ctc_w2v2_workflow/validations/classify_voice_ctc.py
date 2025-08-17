"""
CTC Voice Classification Validation

This script provides real-time phoneme sequence classification using the trained CTC model.
It supports both live audio recording and file-based inference, with CTC decoding
for variable-length phoneme sequence prediction.
"""

import json
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add parent directory to path for imports  
sys.path.append(str(Path(__file__).parent.parent))

# Try to import dependencies with graceful fallback
try:
    import torch
    import soundfile as sf
    from transformers import Wav2Vec2Processor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch/Transformers not available. CTC validation will use mock mode.")

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ sounddevice not available. Live recording disabled.")

# Import CTC model
if TORCH_AVAILABLE:
    try:
        from models.ctc_model import CTCModel, create_ctc_model
        CTC_MODEL_AVAILABLE = True
    except ImportError:
        CTC_MODEL_AVAILABLE = False
        print("⚠️ CTC model not available.")
else:
    CTC_MODEL_AVAILABLE = False


# Constants
SAMPLE_RATE = 16000
DEFAULT_DURATION = 2.0  # seconds
MIN_AMP_THRESHOLD = 0.01
BUFFER_SAMPLES = 1000


class MockCTCClassifier:
    """Mock CTC classifier for testing when dependencies are unavailable."""
    
    def __init__(self, phoneme_labels):
        self.phoneme_labels = phoneme_labels
        self.num_classes = len(phoneme_labels)
        print(f"🤖 Mock CTC classifier initialized with {self.num_classes} phoneme classes")
    
    def predict_sequence(self, audio_data):
        """Mock prediction returning random phoneme sequence."""
        import random
        # Generate random phoneme sequence
        seq_length = random.randint(1, 5)
        predicted_indices = [random.randint(0, self.num_classes-1) for _ in range(seq_length)]
        predicted_phonemes = [self.phoneme_labels[idx] for idx in predicted_indices]
        confidence = random.uniform(0.7, 0.95)
        
        return {
            'phoneme_sequence': predicted_phonemes,
            'indices': predicted_indices,
            'confidence': confidence,
            'sequence_length': seq_length
        }


class CTCPhonemeClassifier:
    """CTC-based phoneme sequence classifier."""
    
    def __init__(self, model_path: str, phoneme_labels_path: str, device: str = 'cpu'):
        """
        Initialize CTC classifier.
        
        Args:
            model_path: Path to trained CTC model
            phoneme_labels_path: Path to phoneme labels JSON
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load phoneme labels
        with open(phoneme_labels_path, 'r') as f:
            self.phoneme_labels = json.load(f)
        print(f"📝 Loaded {len(self.phoneme_labels)} phoneme labels")
        
        # Load model
        if TORCH_AVAILABLE and CTC_MODEL_AVAILABLE:
            self.model = create_ctc_model(num_classes=len(self.phoneme_labels))
            
            # Load trained weights
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded CTC model from {model_path}")
            else:
                print(f"⚠️ Model file not found: {model_path}. Using untrained model.")
            
            self.model.eval()
            self.model.to(device)
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            
        else:
            # Use mock classifier
            self.model = MockCTCClassifier(self.phoneme_labels)
            self.processor = None
            print("🤖 Using mock CTC classifier")
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for CTC inference.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio
        """
        # Remove silence and normalize
        nonzero_indices = np.where(np.abs(audio_data) > MIN_AMP_THRESHOLD)[0]
        
        if len(nonzero_indices) == 0:
            return audio_data  # Return as-is if completely silent
        
        # Trim silence
        start = max(0, nonzero_indices[0] - BUFFER_SAMPLES)
        end = min(len(audio_data), nonzero_indices[-1] + BUFFER_SAMPLES)
        trimmed = audio_data[start:end]
        
        # Normalize
        if np.max(np.abs(trimmed)) > 0:
            trimmed = trimmed / np.max(np.abs(trimmed))
        
        return trimmed
    
    def predict_sequence(self, audio_data: np.ndarray, beam_width: int = 1) -> dict:
        """
        Predict phoneme sequence from audio.
        
        Args:
            audio_data: Audio data (1D numpy array)
            beam_width: Beam search width (1 for greedy)
            
        Returns:
            Dictionary with prediction results
        """
        if not TORCH_AVAILABLE or not CTC_MODEL_AVAILABLE:
            return self.model.predict_sequence(audio_data)
        
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio_data)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(processed_audio).unsqueeze(0)  # Add batch dimension
        audio_tensor = audio_tensor.to(self.device)
        
        # Predict with CTC model
        with torch.no_grad():
            predictions = self.model.predict(audio_tensor, beam_width=beam_width)
        
        # Get prediction for first (and only) sample in batch
        predicted_indices = predictions[0] if predictions else []
        
        # Convert indices to phoneme labels
        predicted_phonemes = []
        for idx in predicted_indices:
            if 0 <= idx < len(self.phoneme_labels):
                predicted_phonemes.append(self.phoneme_labels[idx])
            else:
                print(f"⚠️ Invalid phoneme index: {idx}")
        
        # Calculate confidence (simplified - could use actual CTC scores)
        confidence = 0.8 if predicted_phonemes else 0.0
        
        return {
            'phoneme_sequence': predicted_phonemes,
            'indices': predicted_indices,
            'confidence': confidence,
            'sequence_length': len(predicted_phonemes),
            'audio_length': len(processed_audio)
        }
    
    def predict_from_file(self, audio_file: str, beam_width: int = 1) -> dict:
        """
        Predict phoneme sequence from audio file.
        
        Args:
            audio_file: Path to audio file
            beam_width: Beam search width
            
        Returns:
            Prediction results
        """
        try:
            # Load audio file
            audio_data, sr = sf.read(audio_file)
            
            if sr != SAMPLE_RATE:
                print(f"⚠️ Audio file sample rate {sr} != {SAMPLE_RATE}. Results may be inaccurate.")
            
            print(f"🔊 Loaded audio: {len(audio_data)/sr:.2f}s, {len(audio_data)} samples")
            
            # Predict
            result = self.predict_sequence(audio_data, beam_width)
            result['source_file'] = audio_file
            result['sample_rate'] = sr
            result['duration'] = len(audio_data) / sr
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing audio file {audio_file}: {e}")
            return {
                'phoneme_sequence': [],
                'indices': [],
                'confidence': 0.0,
                'error': str(e)
            }


def record_audio(duration: float = DEFAULT_DURATION, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Recorded audio data
    """
    if not AUDIO_AVAILABLE:
        print("❌ Audio recording not available. Install sounddevice: pip install sounddevice")
        return np.zeros(int(duration * sample_rate))
    
    print(f"🎤 Recording for {duration} seconds. Please speak now...")
    
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait for recording to complete
        return audio.flatten()
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return np.zeros(int(duration * sample_rate))


def interactive_mode(classifier: CTCPhonemeClassifier):
    """
    Interactive mode for real-time phoneme sequence classification.
    
    Args:
        classifier: CTC classifier instance
    """
    print("\n🎙️ Interactive CTC Phoneme Sequence Classification")
    print("Commands:")
    print("  'r' or 'record' - Record and classify audio")
    print("  'f <filename>' - Classify audio file")
    print("  'q' or 'quit' - Exit")
    print("  'h' or 'help' - Show this help")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif command in ['h', 'help']:
                print("Commands: 'r' (record), 'f <file>' (file), 'q' (quit), 'h' (help)")
            
            elif command in ['r', 'record']:
                # Record and classify
                audio_data = record_audio(duration=DEFAULT_DURATION)
                
                if np.max(np.abs(audio_data)) < MIN_AMP_THRESHOLD:
                    print("⚠️ No audio detected. Please try again.")
                    continue
                
                print("🔄 Classifying phoneme sequence...")
                result = classifier.predict_sequence(audio_data, beam_width=1)
                
                print(f"📊 Results:")
                print(f"   Phoneme sequence: {' -> '.join(result['phoneme_sequence'])}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Sequence length: {result['sequence_length']}")
                print(f"   Audio length: {result.get('audio_length', 'unknown')} samples")
            
            elif command.startswith('f '):
                # Classify file
                filename = command[2:].strip()
                if not filename:
                    print("❌ Please provide a filename: f <filename>")
                    continue
                
                if not Path(filename).exists():
                    print(f"❌ File not found: {filename}")
                    continue
                
                print(f"🔄 Classifying file: {filename}")
                result = classifier.predict_from_file(filename, beam_width=1)
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"📊 Results:")
                    print(f"   File: {result.get('source_file', filename)}")
                    print(f"   Duration: {result.get('duration', 'unknown'):.2f}s")
                    print(f"   Phoneme sequence: {' -> '.join(result['phoneme_sequence'])}")
                    print(f"   Confidence: {result['confidence']:.3f}")
                    print(f"   Sequence length: {result['sequence_length']}")
            
            else:
                print("❌ Unknown command. Type 'h' for help.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function for CTC voice classification validation."""
    parser = argparse.ArgumentParser(description="CTC Phoneme Sequence Classification")
    parser.add_argument("--model", type=str, default="dist/ctc_model_best.pt",
                       help="Path to trained CTC model")
    parser.add_argument("--labels", type=str, default="dist/phoneme_labels.json",
                       help="Path to phoneme labels JSON")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--file", type=str,
                       help="Audio file to classify (non-interactive mode)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                       help="Recording duration for live mode")
    parser.add_argument("--beam_width", type=int, default=1,
                       help="Beam search width (1 for greedy)")
    
    args = parser.parse_args()
    
    # Initialize classifier
    print("🚀 Initializing CTC Phoneme Classifier...")
    
    try:
        classifier = CTCPhonemeClassifier(
            model_path=args.model,
            phoneme_labels_path=args.labels,
            device=args.device
        )
        print("✅ Classifier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return 1
    
    # Run classification
    if args.file:
        # File mode
        print(f"🔄 Classifying file: {args.file}")
        result = classifier.predict_from_file(args.file, beam_width=args.beam_width)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return 1
        
        print(f"\n📊 Results:")
        print(f"   File: {args.file}")
        print(f"   Duration: {result.get('duration', 'unknown'):.2f}s")
        print(f"   Phoneme sequence: {' -> '.join(result['phoneme_sequence'])}")
        print(f"   Indices: {result['indices']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Sequence length: {result['sequence_length']}")
        
    else:
        # Interactive mode
        interactive_mode(classifier)
    
    return 0


if __name__ == "__main__":
    exit(main())