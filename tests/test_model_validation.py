#!/usr/bin/env python3
"""
Model Validation Tests - Clean test recordings separate from training data

This module provides automated testing for both MLP and CTC models using
pre-recorded test sounds that are completely separate from training data.
"""

import sys
import os
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.cli.model_loader import ModelLoader
from inference.cli.audio_feature_extractor import AudioFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Single test result for a phoneme prediction"""
    phoneme: str
    predicted_phoneme: str
    confidence: float
    probabilities: Dict[str, float]
    correct: bool
    model_type: str
    timestamp: str

@dataclass
class ValidationResults:
    """Complete validation results for a model"""
    model_type: str
    total_tests: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    test_results: List[TestResult]
    timestamp: str
    issues_found: List[str]

class ModelValidator:
    """Validates MLP and CTC models using clean test recordings"""
    
    def __init__(self, test_recordings_dir: str = "tests/test_recordings"):
        self.test_recordings_dir = Path(test_recordings_dir)
        self.results_dir = Path("tests/validation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize model components
        self.model_loader = None
        self.feature_extractor = None
        
        logger.info(f"Initialized ModelValidator with test recordings dir: {self.test_recordings_dir}")

    def setup_models(self):
        """Initialize model loader and feature extractor"""
        try:
            self.model_loader = ModelLoader()
            self.feature_extractor = AudioFeatureExtractor()
            logger.info("Model components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def validate_mlp_model(self) -> ValidationResults:
        """
        Validate MLP Control model using single phoneme recordings
        
        Expected test structure:
        tests/test_recordings/mlp/
        ├── AE_test_001.wav (target: AE phoneme)
        ├── B_test_001.wav  (target: B phoneme)
        └── ... (one file per phoneme to test)
        """
        logger.info("Starting MLP model validation...")
        
        mlp_test_dir = self.test_recordings_dir / "mlp"
        if not mlp_test_dir.exists():
            raise FileNotFoundError(f"MLP test directory not found: {mlp_test_dir}")
        
        test_results = []
        issues_found = []
        
        # Load MLP model
        try:
            available_models = self.model_loader.list_available_models()
            if 'mlp_control' not in available_models:
                raise ValueError("MLP Control model not found")
                
            self.model_loader.load_model('mlp_control')
            logger.info("MLP Control model loaded successfully")
        except Exception as e:
            issues_found.append(f"Model loading failed: {e}")
            return self._create_failed_validation("mlp_control", issues_found)

        # Test each recording
        for wav_file in sorted(mlp_test_dir.glob("*.wav")):
            try:
                # Extract target phoneme from filename (e.g., "AE_test_001.wav" → "AE")
                target_phoneme = wav_file.stem.split('_')[0]
                
                # Load and process audio
                audio_data, sample_rate = sf.read(wav_file)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)  # Convert to mono
                
                # Extract features
                feature_result = self.feature_extractor.extract_features_for_model(
                    audio_data, sample_rate, 'mlp_control'
                )
                
                # Run inference  
                probabilities = self.model_loader.run_inference(feature_result['features'])
                
                # Get prediction
                predicted_idx = np.argmax(probabilities)
                predicted_phoneme = self.model_loader.get_labels()[predicted_idx]
                confidence = float(probabilities[predicted_idx])
                
                # Check for issues
                if confidence == 1.0 or confidence == 0.0:
                    issues_found.append(f"Binary confidence for {wav_file.name}: {confidence}")
                
                # Create probability dict
                prob_dict = {
                    label: float(prob) 
                    for label, prob in zip(self.model_loader.get_labels(), probabilities)
                }
                
                # Record result
                result = TestResult(
                    phoneme=target_phoneme,
                    predicted_phoneme=predicted_phoneme,
                    confidence=confidence,
                    probabilities=prob_dict,
                    correct=(target_phoneme == predicted_phoneme),
                    model_type="mlp_control",
                    timestamp=datetime.now().isoformat()
                )
                
                test_results.append(result)
                logger.info(f"MLP Test: {target_phoneme} → {predicted_phoneme} (conf: {confidence:.3f})")
                
            except Exception as e:
                issues_found.append(f"Failed to process {wav_file.name}: {e}")
                logger.error(f"Error processing {wav_file.name}: {e}")

        # Calculate metrics
        if test_results:
            correct_count = sum(1 for r in test_results if r.correct)
            accuracy = correct_count / len(test_results)
            avg_confidence = np.mean([r.confidence for r in test_results])
        else:
            correct_count = 0
            accuracy = 0.0
            avg_confidence = 0.0
            issues_found.append("No test results generated")

        validation_result = ValidationResults(
            model_type="mlp_control",
            total_tests=len(test_results),
            correct_predictions=correct_count,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            test_results=test_results,
            timestamp=datetime.now().isoformat(),
            issues_found=issues_found
        )
        
        logger.info(f"MLP Validation Complete: {accuracy:.1%} accuracy ({correct_count}/{len(test_results)})")
        return validation_result

    def validate_ctc_model(self, model_name: str = 'wavlm_ctc') -> ValidationResults:
        """
        Validate CTC model using phoneme sequence recordings
        
        Expected test structure:
        tests/test_recordings/ctc/
        ├── BAT_sequence_001.wav (target: B-AE-T sequence)
        ├── CAT_sequence_001.wav (target: K-AE-T sequence)  
        └── ... (sequence recordings)
        """
        logger.info(f"Starting CTC model validation for {model_name}...")
        
        ctc_test_dir = self.test_recordings_dir / "ctc"
        if not ctc_test_dir.exists():
            raise FileNotFoundError(f"CTC test directory not found: {ctc_test_dir}")
        
        test_results = []
        issues_found = []
        
        # Load CTC model
        try:
            available_models = self.model_loader.list_available_models()
            if model_name not in available_models:
                raise ValueError(f"CTC model {model_name} not found")
                
            self.model_loader.load_model(model_name)
            logger.info(f"{model_name} model loaded successfully")
        except Exception as e:
            issues_found.append(f"Model loading failed: {e}")
            return self._create_failed_validation(model_name, issues_found)

        # Test each sequence recording
        for wav_file in sorted(ctc_test_dir.glob("*.wav")):
            try:
                # Extract target sequence from filename (e.g., "BAT_sequence_001.wav" → "BAT")
                target_word = wav_file.stem.split('_')[0]
                
                # Load and process audio
                audio_data, sample_rate = sf.read(wav_file)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)  # Convert to mono
                
                # Extract features
                feature_result = self.feature_extractor.extract_features_for_model(
                    audio_data, sample_rate, model_name
                )
                
                # Run CTC inference
                probabilities = self.model_loader.run_inference(feature_result['features'])
                
                # Check for binary outputs (indicates broken CTC)
                unique_probs = np.unique(probabilities)
                if len(unique_probs) <= 2 and set(unique_probs).issubset({0.0, 1.0}):
                    issues_found.append(f"CTC binary output for {wav_file.name}: {unique_probs}")
                
                # For now, get the most probable single phoneme (we'll improve this)
                predicted_idx = np.argmax(probabilities)
                predicted_phoneme = self.model_loader.get_labels()[predicted_idx]
                confidence = float(probabilities[predicted_idx])
                
                # Create probability dict
                prob_dict = {
                    label: float(prob) 
                    for label, prob in zip(self.model_loader.get_labels(), probabilities)
                }
                
                # For sequence testing, we need to implement proper CTC decoding
                # For now, mark as "needs_ctc_decoding"
                correct = False  # We can't properly evaluate without CTC decoding
                if len(target_word) == 1:  # Single phoneme test
                    correct = (target_word == predicted_phoneme)
                
                result = TestResult(
                    phoneme=target_word,  # Could be sequence
                    predicted_phoneme=predicted_phoneme,
                    confidence=confidence,
                    probabilities=prob_dict,
                    correct=correct,
                    model_type=model_name,
                    timestamp=datetime.now().isoformat()
                )
                
                test_results.append(result)
                logger.info(f"CTC Test: {target_word} → {predicted_phoneme} (conf: {confidence:.3f})")
                
            except Exception as e:
                issues_found.append(f"Failed to process {wav_file.name}: {e}")
                logger.error(f"Error processing {wav_file.name}: {e}")

        # Add CTC-specific issues
        if not any("CTC decoding" in issue for issue in issues_found):
            issues_found.append("CTC decoding not implemented - sequence evaluation incomplete")

        # Calculate metrics
        if test_results:
            correct_count = sum(1 for r in test_results if r.correct)
            accuracy = correct_count / len(test_results)
            avg_confidence = np.mean([r.confidence for r in test_results])
        else:
            correct_count = 0
            accuracy = 0.0
            avg_confidence = 0.0
            issues_found.append("No test results generated")

        validation_result = ValidationResults(
            model_type=model_name,
            total_tests=len(test_results),
            correct_predictions=correct_count,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            test_results=test_results,
            timestamp=datetime.now().isoformat(),
            issues_found=issues_found
        )
        
        logger.info(f"CTC Validation Complete: {accuracy:.1%} accuracy ({correct_count}/{len(test_results)})")
        logger.warning("CTC sequence evaluation incomplete - needs proper CTC decoding")
        return validation_result

    def _create_failed_validation(self, model_type: str, issues: List[str]) -> ValidationResults:
        """Create a failed validation result"""
        return ValidationResults(
            model_type=model_type,
            total_tests=0,
            correct_predictions=0,
            accuracy=0.0,
            avg_confidence=0.0,
            test_results=[],
            timestamp=datetime.now().isoformat(),
            issues_found=issues
        )

    def save_validation_results(self, results: ValidationResults):
        """Save validation results to JSON for future reference"""
        filename = f"validation_{results.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        # Convert to dict for JSON serialization
        results_dict = {
            'model_type': results.model_type,
            'total_tests': results.total_tests,
            'correct_predictions': results.correct_predictions,
            'accuracy': results.accuracy,
            'avg_confidence': results.avg_confidence,
            'timestamp': results.timestamp,
            'issues_found': results.issues_found,
            'test_results': [
                {
                    'phoneme': r.phoneme,
                    'predicted_phoneme': r.predicted_phoneme,
                    'confidence': r.confidence,
                    'probabilities': r.probabilities,
                    'correct': r.correct,
                    'model_type': r.model_type,
                    'timestamp': r.timestamp
                }
                for r in results.test_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Validation results saved to: {filepath}")
        return filepath

    def run_all_validations(self) -> Dict[str, ValidationResults]:
        """Run validation for all available models"""
        self.setup_models()
        
        results = {}
        
        # Validate MLP
        try:
            mlp_results = self.validate_mlp_model()
            results['mlp_control'] = mlp_results
            self.save_validation_results(mlp_results)
        except Exception as e:
            logger.error(f"MLP validation failed: {e}")
        
        # Validate CTC models
        for ctc_model in ['wav2vec2_ctc', 'wavlm_ctc']:
            try:
                ctc_results = self.validate_ctc_model(ctc_model)
                results[ctc_model] = ctc_results
                self.save_validation_results(ctc_results)
            except Exception as e:
                logger.error(f"{ctc_model} validation failed: {e}")
        
        return results

    def print_summary(self, results: Dict[str, ValidationResults]):
        """Print a summary of all validation results"""
        print("\n" + "="*60)
        print("MODEL VALIDATION SUMMARY")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {result.accuracy:.1%} ({result.correct_predictions}/{result.total_tests})")
            print(f"  Avg Confidence: {result.avg_confidence:.3f}")
            
            if result.issues_found:
                print(f"  Issues Found:")
                for issue in result.issues_found:
                    print(f"    - {issue}")
        
        print("\n" + "="*60)

def main():
    """Main function to run model validation"""
    print("Starting Model Validation Tests...")
    print("NOTE: This requires test recordings in tests/test_recordings/")
    print("  - tests/test_recordings/mlp/ for MLP single phoneme tests")
    print("  - tests/test_recordings/ctc/ for CTC sequence tests")
    
    validator = ModelValidator()
    
    try:
        results = validator.run_all_validations()
        validator.print_summary(results)
        
        # Check for critical issues
        critical_issues = []
        for model_name, result in results.items():
            for issue in result.issues_found:
                if any(keyword in issue.lower() for keyword in ['binary', 'failed', '1.0', '0.0']):
                    critical_issues.append(f"{model_name}: {issue}")
        
        if critical_issues:
            print("\n🚨 CRITICAL ISSUES FOUND:")
            for issue in critical_issues:
                print(f"  - {issue}")
            print("\nThese issues need to be fixed before reliable testing.")
        else:
            print("\n✅ No critical issues detected in models.")
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())