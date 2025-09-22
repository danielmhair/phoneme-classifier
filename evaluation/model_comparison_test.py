#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Epic 1 & Epic 2 Integration

Tests all three models (MLP Control, Wav2Vec2 CTC, WavLM CTC) against real phoneme recordings
and generates detailed JSON reports for analysis.

Usage:
    poetry run python evaluation/model_comparison_test.py
    
Outputs:
    - detailed_results.json: Per-phoneme top 5 predictions from each model
    - accuracy_summary.json: Percentage correct per model per phoneme
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.cli.model_loader import ModelLoader
from inference.cli.audio_feature_extractor import AudioFeatureExtractor


class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, recordings_dir: str = "recordings"):
        """Initialize evaluator with recordings directory."""
        self.recordings_dir = Path(recordings_dir)
        self.model_loader = ModelLoader()
        self.feature_extractor = AudioFeatureExtractor()
        
        # Models to test
        self.models = ['mlp_control', 'wav2vec2_ctc', 'wavlm_ctc']
        
        # Load phoneme labels (same for all models)
        self.phoneme_labels = self._load_phoneme_labels()
        
        # Results storage
        self.detailed_results: Dict[str, Any] = {}
        self.accuracy_summary: Dict[str, Dict[str, float]] = {}
        
    def _load_phoneme_labels(self) -> List[str]:
        """Load phoneme labels from any available model."""
        labels_path = Path("workflows/mlp_control_workflow/dist/phoneme_labels.json")
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError("Could not find phoneme labels. Run Epic 1 training first.")
    
    def _extract_phoneme_from_path(self, file_path: Path) -> str:
        """Extract expected phoneme from file directory structure."""
        # recordings/callie/a_æ/callie_a_æ_ep-æ_20250422_180744_1.wav
        # Expected phoneme is the parent directory name
        return file_path.parent.name
    
    def _load_audio_file(self, file_path: Path) -> np.ndarray:
        """Load audio file with error handling."""
        try:
            audio_data, sample_rate = sf.read(file_path)
            
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed (most files should already be 16kHz)
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            return None
    
    def _get_top_predictions(self, probabilities: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top K predictions with scores."""
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for i, idx in enumerate(top_indices):
            predictions.append({
                'rank': i + 1,
                'phoneme': self.phoneme_labels[idx],
                'probability': float(probabilities[idx]),
                'percentage': float(probabilities[idx] * 100)
            })
        
        return predictions
    
    def _evaluate_single_file(self, file_path: Path, expected_phoneme: str) -> Dict[str, Any]:
        """Evaluate a single audio file with all models."""
        file_results = {
            'file_path': str(file_path),
            'expected_phoneme': expected_phoneme,
            'expected_phoneme_index': self.phoneme_labels.index(expected_phoneme) if expected_phoneme in self.phoneme_labels else -1,
            'models': {}
        }
        
        # Load audio
        audio_data = self._load_audio_file(file_path)
        if audio_data is None:
            return file_results
        
        # Test each model
        for model_id in self.models:
            model_result = {
                'model_id': model_id,
                'success': False,
                'inference_time_ms': 0,
                'top_predictions': [],
                'correct_prediction': False,
                'expected_rank': None,
                'expected_probability': 0.0,
                'error': None
            }
            
            try:
                # Load model
                load_success = self.model_loader.load_model(model_id)
                if not load_success:
                    model_result['error'] = f"Failed to load model {model_id}"
                    file_results['models'][model_id] = model_result
                    continue
                
                # Extract features
                features_result = self.feature_extractor.extract_features_for_model(audio_data, model_id)
                
                if not features_result['preprocessing_success'] or features_result['features'] is None:
                    model_result['error'] = f"Feature extraction failed: {features_result.get('error', 'unknown')}"
                    file_results['models'][model_id] = model_result
                    continue
                
                # Run inference with timing
                start_time = time.time()
                probabilities = self.model_loader.run_inference(features_result['features'])
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Get top predictions
                top_predictions = self._get_top_predictions(probabilities, top_k=5)
                
                # Check if expected phoneme was predicted correctly
                if expected_phoneme in self.phoneme_labels:
                    expected_idx = self.phoneme_labels.index(expected_phoneme)
                    expected_prob = float(probabilities[expected_idx])
                    
                    # Find rank of expected phoneme
                    sorted_indices = np.argsort(probabilities)[::-1]
                    expected_rank = int(np.where(sorted_indices == expected_idx)[0][0] + 1)
                    
                    correct_prediction = expected_rank == 1
                    
                    model_result.update({
                        'correct_prediction': correct_prediction,
                        'expected_rank': expected_rank,
                        'expected_probability': expected_prob
                    })
                
                model_result.update({
                    'success': True,
                    'inference_time_ms': inference_time,
                    'top_predictions': top_predictions
                })
                
            except Exception as e:
                model_result['error'] = str(e)
            
            file_results['models'][model_id] = model_result
        
        return file_results
    
    def _scan_recordings(self) -> List[Tuple[Path, str]]:
        """Scan recordings directory for all phoneme audio files."""
        audio_files = []
        
        for speaker_dir in self.recordings_dir.iterdir():
            if not speaker_dir.is_dir() or speaker_dir.name in ['evaluation1.json', 'evaluation1.md', 'silence.wav']:
                continue
                
            for phoneme_dir in speaker_dir.iterdir():
                if not phoneme_dir.is_dir():
                    continue
                    
                expected_phoneme = phoneme_dir.name
                
                # Only test phonemes that are in our labels
                if expected_phoneme not in self.phoneme_labels:
                    print(f"⚠️  Skipping unknown phoneme: {expected_phoneme}")
                    continue
                
                for audio_file in phoneme_dir.glob("*.wav"):
                    audio_files.append((audio_file, expected_phoneme))
        
        return sorted(audio_files)
    
    def run_evaluation(self) -> None:
        """Run comprehensive evaluation on all recordings."""
        print("🎯 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 50)
        print(f"📁 Recordings directory: {self.recordings_dir}")
        print(f"🎙️  Available models: {self.models}")
        print(f"🏷️  Phoneme labels: {len(self.phoneme_labels)} phonemes")
        print()
        
        # Scan for audio files
        audio_files = self._scan_recordings()
        print(f"📊 Found {len(audio_files)} audio files to evaluate")
        print()
        
        # Initialize results structure
        self.detailed_results = {
            'metadata': {
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_files': len(audio_files),
                'models_tested': self.models,
                'phoneme_labels': self.phoneme_labels,
                'total_phonemes': len(self.phoneme_labels)
            },
            'per_file_results': []
        }
        
        # Initialize accuracy tracking
        self.accuracy_summary = {
            'metadata': self.detailed_results['metadata'].copy(),
            'per_phoneme_accuracy': {},
            'overall_accuracy': {}
        }
        
        for phoneme in self.phoneme_labels:
            self.accuracy_summary['per_phoneme_accuracy'][phoneme] = {
                model_id: {'correct': 0, 'total': 0, 'percentage': 0.0} 
                for model_id in self.models
            }
        
        # Process each audio file
        print("🔄 Processing audio files...")
        start_time = time.time()
        
        for i, (file_path, expected_phoneme) in enumerate(tqdm(audio_files, desc="Evaluating")):
            file_result = self._evaluate_single_file(file_path, expected_phoneme)
            self.detailed_results['per_file_results'].append(file_result)
            
            # Update accuracy tracking
            for model_id in self.models:
                if model_id in file_result['models']:
                    model_result = file_result['models'][model_id]
                    if model_result['success']:
                        self.accuracy_summary['per_phoneme_accuracy'][expected_phoneme][model_id]['total'] += 1
                        if model_result['correct_prediction']:
                            self.accuracy_summary['per_phoneme_accuracy'][expected_phoneme][model_id]['correct'] += 1
            
            # Progress update and time estimation every 100 files
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed  # files per second
                remaining = len(audio_files) - (i + 1)
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600
                
                print(f"   📊 Processed {i + 1}/{len(audio_files)} files")
                print(f"   ⏱️  Rate: {rate:.1f} files/sec, ETA: {eta_hours:.1f} hours")
                
                # Save intermediate results every 500 files
                if (i + 1) % 500 == 0:
                    self._save_intermediate_results(i + 1)
        
        # Calculate final percentages
        self._calculate_accuracy_percentages()
        
        # Save results
        self._save_results()
        
        print("\n✅ EVALUATION COMPLETED")
        print(f"📊 Processed {len(audio_files)} files")
        print(f"💾 Results saved to:")
        print(f"   📋 detailed_results.json")
        print(f"   📈 accuracy_summary.json")
    
    def _calculate_accuracy_percentages(self) -> None:
        """Calculate accuracy percentages for all models and phonemes."""
        # Per-phoneme accuracy
        for phoneme in self.phoneme_labels:
            for model_id in self.models:
                stats = self.accuracy_summary['per_phoneme_accuracy'][phoneme][model_id]
                if stats['total'] > 0:
                    stats['percentage'] = (stats['correct'] / stats['total']) * 100
                else:
                    stats['percentage'] = 0.0
        
        # Overall accuracy per model
        overall_stats = {}
        for model_id in self.models:
            total_correct = 0
            total_files = 0
            
            for phoneme in self.phoneme_labels:
                stats = self.accuracy_summary['per_phoneme_accuracy'][phoneme][model_id]
                total_correct += stats['correct']
                total_files += stats['total']
            
            overall_percentage = (total_correct / total_files * 100) if total_files > 0 else 0.0
            
            overall_stats[model_id] = {
                'correct': total_correct,
                'total': total_files,
                'percentage': overall_percentage
            }
        
        self.accuracy_summary['overall_accuracy'] = overall_stats
    
    def _save_intermediate_results(self, files_processed: int) -> None:
        """Save intermediate results during long evaluation."""
        eval_dir = Path("evaluation")
        eval_dir.mkdir(exist_ok=True)
        
        # Calculate current percentages
        self._calculate_accuracy_percentages()
        
        # Save intermediate results with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        intermediate_path = eval_dir / f"intermediate_results_{files_processed}_{timestamp}.json"
        
        intermediate_data = {
            'files_processed': files_processed,
            'total_files': self.detailed_results['metadata']['total_files'],
            'progress_percentage': (files_processed / self.detailed_results['metadata']['total_files']) * 100,
            'current_accuracy': self.accuracy_summary['overall_accuracy'],
            'timestamp': timestamp
        }
        
        with open(intermediate_path, 'w') as f:
            json.dump(intermediate_data, f, indent=2)
        
        print(f"   💾 Intermediate results saved: {intermediate_path}")
    
    def _save_results(self) -> None:
        """Save evaluation results to JSON files."""
        # Create evaluation directory if it doesn't exist
        eval_dir = Path("evaluation")
        eval_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_path = eval_dir / "detailed_results.json"
        with open(detailed_path, 'w') as f:
            json.dump(self.detailed_results, f, indent=2, ensure_ascii=False)
        
        # Save accuracy summary
        summary_path = eval_dir / "accuracy_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.accuracy_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results: {detailed_path}")
        print(f"💾 Accuracy summary: {summary_path}")
    
    def print_quick_summary(self) -> None:
        """Print a quick summary of results."""
        if not self.accuracy_summary:
            print("❌ No results available")
            return
        
        print("\n📊 QUICK ACCURACY SUMMARY")
        print("=" * 40)
        
        overall = self.accuracy_summary['overall_accuracy']
        for model_id in self.models:
            if model_id in overall:
                stats = overall[model_id]
                print(f"{model_id:15}: {stats['percentage']:6.2f}% ({stats['correct']:4}/{stats['total']:4})")
        
        print("\n🔝 TOP 3 PHONEMES PER MODEL:")
        for model_id in self.models:
            phoneme_accuracies = []
            for phoneme in self.phoneme_labels:
                stats = self.accuracy_summary['per_phoneme_accuracy'][phoneme][model_id]
                if stats['total'] > 0:
                    phoneme_accuracies.append((phoneme, stats['percentage'], stats['total']))
            
            # Sort by accuracy, then by sample count
            phoneme_accuracies.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            print(f"\n{model_id}:")
            for i, (phoneme, percentage, total) in enumerate(phoneme_accuracies[:3]):
                print(f"  {i+1}. {phoneme:10}: {percentage:6.2f}% (n={total})")


def main():
    """Main evaluation function."""
    print("🚀 Starting Model Comparison Evaluation")
    print("🎯 Testing all three Epic 1 models on real recordings")
    print()
    
    # Create evaluation directory
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run evaluation
    try:
        evaluator.run_evaluation()
        evaluator.print_quick_summary()
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        print("💾 Partial results may be available in evaluation/ directory")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("🔧 Check that all Epic 1 models are trained and available")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())