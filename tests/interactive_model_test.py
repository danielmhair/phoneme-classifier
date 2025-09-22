#!/usr/bin/env python3
"""
Interactive Model Testing CLI

Usage:
    python tests/interactive_model_test.py mlp AE      # Test MLP with AE phoneme
    python tests/interactive_model_test.py ctc BAT     # Test CTC with BAT sequence
    python tests/interactive_model_test.py mlp         # Interactive MLP mode
    python tests/interactive_model_test.py ctc         # Interactive CTC mode
"""

import sys
import os
import json
import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import argparse
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.cli.model_loader import ModelLoader
from inference.cli.audio_feature_extractor import AudioFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveModelTester:
    """Interactive CLI for testing MLP and CTC models with manual input"""
    
    def __init__(self):
        self.model_loader = None
        self.feature_extractor = None
        self.results_dir = Path("tests/interactive_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Audio recording settings
        self.sample_rate = 16000
        self.recording_duration = 3.0  # seconds
        
        # Load valid phoneme labels
        self.valid_phonemes = self._load_valid_phonemes()
        
        print("🎤 Interactive Model Tester")
        print("=" * 50)
        print(f"✅ Loaded {len(self.valid_phonemes)} valid phoneme labels")

    def _load_valid_phonemes(self) -> List[str]:
        """Load valid phoneme labels from the project's phoneme_labels.json"""
        labels_path = Path("workflows/mlp_control_workflow/dist/phoneme_labels.json")
        
        if not labels_path.exists():
            print(f"⚠️  Warning: {labels_path} not found, using default labels")
            return []
        
        try:
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            return [label.upper() for label in labels]  # Normalize to uppercase
        except Exception as e:
            print(f"⚠️  Warning: Failed to load phoneme labels: {e}")
            return []

    def validate_phoneme_input(self, phoneme: str) -> bool:
        """Validate that the phoneme is in our accepted list"""
        if not self.valid_phonemes:
            return True  # No validation if labels not loaded
        
        return phoneme.upper() in self.valid_phonemes

    def validate_sequence_input(self, sequence: str) -> bool:
        """Validate that all phonemes in the sequence are valid"""
        if not self.valid_phonemes:
            return True  # No validation if labels not loaded
        
        # For sequences like "BAT", check if it could be a word
        # or if each character represents a phoneme
        sequence_upper = sequence.upper()
        
        # Check if the whole sequence is a valid phoneme
        if sequence_upper in self.valid_phonemes:
            return True
        
        # For word-like sequences, we'll be more permissive
        # since they represent phoneme combinations
        return len(sequence) >= 2  # At least 2 characters for a sequence

    def show_valid_phonemes(self):
        """Display valid phonemes in a nice format"""
        if not self.valid_phonemes:
            print("⚠️  No phoneme labels loaded")
            return
        
        print("\n📋 Valid Phonemes (38 total):")
        print("=" * 60)
        
        # Group phonemes for better display
        vowels = [p for p in self.valid_phonemes if any(v in p.lower() for v in 'aeiou')]
        consonants = [p for p in self.valid_phonemes if p not in vowels]
        
        print("🔤 Vowels:")
        for i, phoneme in enumerate(vowels, 1):
            print(f"  {phoneme}", end="  " if i % 6 != 0 else "\n")
        if len(vowels) % 6 != 0:
            print()
        
        print("\n🔤 Consonants:")
        for i, phoneme in enumerate(consonants, 1):
            print(f"  {phoneme}", end="  " if i % 8 != 0 else "\n")
        if len(consonants) % 8 != 0:
            print()
        
        print("\n💡 Examples for testing:")
        print("  MLP: B, K, AE, S, SH, TH")
        print("  CTC: BAT, CAT, DOG, FISH")
        print("=" * 60)

    def setup_models(self):
        """Initialize model components"""
        try:
            self.model_loader = ModelLoader()
            self.feature_extractor = AudioFeatureExtractor()
            print("✅ Model components initialized")
        except Exception as e:
            print(f"❌ Failed to initialize models: {e}")
            raise

    def record_audio(self, duration: float = None) -> np.ndarray:
        """Record audio from microphone"""
        if duration is None:
            duration = self.recording_duration
            
        print(f"🔴 Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate), 
            samplerate=self.sample_rate, 
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        print("⏹️  Recording complete!")
        return audio_data.flatten()

    def test_mlp_phoneme(self, target_phoneme: str, audio_data: np.ndarray = None) -> Dict:
        """Test MLP model with single phoneme"""
        if audio_data is None:
            print(f"\n🎯 Testing MLP with phoneme: {target_phoneme}")
            print(f"Say the '{target_phoneme}' sound clearly...")
            audio_data = self.record_audio()
        
        try:
            # Load MLP model
            available_models = self.model_loader.list_available_models()
            if 'mlp_control' not in available_models:
                raise ValueError("MLP Control model not available")
            
            self.model_loader.load_model('mlp_control')
            
            # Extract features
            feature_result = self.feature_extractor.extract_features_for_model(
                audio_data, self.sample_rate, 'mlp_control'
            )
            
            # Run inference
            probabilities = self.model_loader.run_inference(feature_result['features'])
            
            # Get prediction
            predicted_idx = np.argmax(probabilities)
            predicted_phoneme = self.model_loader.get_labels()[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Check for issues
            issues = []
            if confidence == 1.0 or confidence == 0.0:
                issues.append(f"Binary confidence detected: {confidence}")
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                {
                    'phoneme': self.model_loader.get_labels()[idx],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_3_indices
            ]
            
            result = {
                'model_type': 'mlp_control',
                'target_phoneme': target_phoneme,
                'predicted_phoneme': predicted_phoneme,
                'confidence': confidence,
                'correct': (target_phoneme.upper() == predicted_phoneme.upper()),
                'top_3_predictions': top_3_predictions,
                'issues': issues,
                'timestamp': datetime.now().isoformat(),
                'probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.model_loader.get_labels(), probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'model_type': 'mlp_control',
                'target_phoneme': target_phoneme,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def test_ctc_sequence(self, target_sequence: str, audio_data: np.ndarray = None, model_name: str = 'wavlm_ctc') -> Dict:
        """Test CTC model with phoneme sequence"""
        if audio_data is None:
            print(f"\n🎯 Testing CTC with sequence: {target_sequence}")
            print(f"Say the word '{target_sequence}' clearly (like '{'-'.join(target_sequence)}')...")
            audio_data = self.record_audio()
        
        try:
            # Load CTC model
            available_models = self.model_loader.list_available_models()
            if model_name not in available_models:
                raise ValueError(f"CTC model {model_name} not available")
            
            self.model_loader.load_model(model_name)
            
            # Extract features
            feature_result = self.feature_extractor.extract_features_for_model(
                audio_data, self.sample_rate, model_name
            )
            
            # Run CTC inference
            probabilities = self.model_loader.run_inference(feature_result['features'])
            
            # Check for binary outputs (indicates broken CTC)
            unique_probs = np.unique(probabilities)
            issues = []
            
            if len(unique_probs) <= 2 and set(unique_probs).issubset({0.0, 1.0}):
                issues.append(f"CTC binary output detected: {unique_probs}")
            
            # For now, get most probable single phoneme (need proper CTC decoding)
            predicted_idx = np.argmax(probabilities)
            predicted_phoneme = self.model_loader.get_labels()[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                {
                    'phoneme': self.model_loader.get_labels()[idx],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_3_indices
            ]
            
            # Note: This is incomplete without proper CTC decoding
            issues.append("CTC sequence decoding not implemented - showing single phoneme only")
            
            result = {
                'model_type': model_name,
                'target_sequence': target_sequence,
                'predicted_phoneme': predicted_phoneme,  # Single phoneme until CTC decoding implemented
                'confidence': confidence,
                'correct': False,  # Can't evaluate sequences without CTC decoding
                'top_3_predictions': top_3_predictions,
                'issues': issues,
                'timestamp': datetime.now().isoformat(),
                'probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.model_loader.get_labels(), probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'model_type': model_name,
                'target_sequence': target_sequence,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def print_result(self, result: Dict):
        """Print test result in a nice format"""
        print("\n" + "="*60)
        print("🔍 TEST RESULT")
        print("="*60)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Basic info
        print(f"Model: {result['model_type']}")
        if 'target_phoneme' in result:
            print(f"Target: {result['target_phoneme']}")
            print(f"Predicted: {result['predicted_phoneme']}")
        else:
            print(f"Target: {result['target_sequence']}")
            print(f"Predicted: {result['predicted_phoneme']} (single phoneme only)")
        
        print(f"Confidence: {result['confidence']:.3f}")
        
        # Correctness
        if result.get('correct'):
            print("✅ CORRECT!")
        else:
            print("❌ INCORRECT")
        
        # Top 3 predictions
        print(f"\nTop 3 Predictions:")
        for i, pred in enumerate(result.get('top_3_predictions', []), 1):
            print(f"  {i}. {pred['phoneme']}: {pred['confidence']:.3f}")
        
        # Issues
        if result.get('issues'):
            print(f"\n⚠️  Issues Found:")
            for issue in result['issues']:
                print(f"  - {issue}")
        
        print("="*60)

    def save_result(self, result: Dict):
        """Save result to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = result.get('model_type', 'unknown')
        
        if 'target_phoneme' in result:
            target = result['target_phoneme']
        else:
            target = result.get('target_sequence', 'unknown')
        
        filename = f"test_{model_type}_{target}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Result saved to: {filepath}")

    def interactive_mode(self, model_type: str):
        """Interactive mode for continuous testing"""
        print(f"\n🎮 Interactive {model_type.upper()} Testing Mode")
        print("Commands: 'help' for phoneme list, 'quit' to exit")
        
        while True:
            if model_type == 'mlp':
                target = input(f"\nEnter phoneme to test (e.g., AE, B, K) or 'help': ").strip()
            else:
                target = input(f"\nEnter sequence to test (e.g., BAT, CAT) or 'help': ").strip()
            
            if target.lower() in ['quit', 'exit', 'q']:
                break
            
            if target.lower() in ['help', 'h']:
                self.show_valid_phonemes()
                continue
            
            if not target:
                continue
            
            # Validate input
            if model_type == 'mlp':
                if not self.validate_phoneme_input(target):
                    print(f"❌ '{target.upper()}' is not a valid phoneme.")
                    print("💡 Type 'help' to see all 38 valid phonemes")
                    continue
            else:
                if not self.validate_sequence_input(target):
                    print(f"❌ '{target.upper()}' is not a valid sequence.")
                    print("💡 Sequences should be 2+ characters (e.g., BAT, CAT)")
                    continue
            
            try:
                if model_type == 'mlp':
                    result = self.test_mlp_phoneme(target.upper())
                else:
                    result = self.test_ctc_sequence(target.upper())
                
                self.print_result(result)
                
                # Ask if user wants to save
                save = input("💾 Save this result? (y/n): ").strip().lower()
                if save in ['y', 'yes']:
                    self.save_result(result)
                
            except KeyboardInterrupt:
                print("\n⏹️  Recording interrupted")
                continue
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive Model Testing CLI')
    parser.add_argument('model_type', choices=['mlp', 'ctc'], 
                       help='Model type to test (mlp or ctc)')
    parser.add_argument('target', nargs='?', 
                       help='Phoneme (for mlp) or sequence (for ctc) to test')
    parser.add_argument('--ctc-model', default='wavlm_ctc',
                       choices=['wav2vec2_ctc', 'wavlm_ctc'],
                       help='Which CTC model to use')
    
    args = parser.parse_args()
    
    tester = InteractiveModelTester()
    tester.setup_models()
    
    if args.target:
        # Single test mode - validate input first
        target_upper = args.target.upper()
        
        if args.model_type == 'mlp':
            if not tester.validate_phoneme_input(target_upper):
                print(f"❌ '{target_upper}' is not a valid phoneme.")
                print("💡 Valid phonemes:")
                tester.show_valid_phonemes()
                return 1
        else:
            if not tester.validate_sequence_input(target_upper):
                print(f"❌ '{target_upper}' is not a valid sequence.")
                print("💡 Sequences should be 2+ characters representing phoneme combinations")
                return 1
        
        try:
            if args.model_type == 'mlp':
                result = tester.test_mlp_phoneme(target_upper)
            else:
                result = tester.test_ctc_sequence(target_upper, model_name=args.ctc_model)
            
            tester.print_result(result)
            
            # Ask if user wants to save
            save = input("💾 Save this result? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                tester.save_result(result)
                
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Interactive mode
        try:
            tester.interactive_mode(args.model_type)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()