#!/usr/bin/env python3
"""
Model Comparison Framework - Compare MLP vs CTC Performance

This module provides systematic comparison capabilities between MLP and CTC models
for phoneme classification tasks, including accuracy, inference time, and model characteristics.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from workflows import (
    MLP_MODEL_PATH, MLP_LABEL_ENCODER_PATH, MLP_PHONEME_LABELS_JSON_PATH,
    CTC_MODEL_PATH, CTC_LABEL_ENCODER_PATH, CTC_PHONEME_LABELS_JSON_PATH,
    MLP_DIST_DIR, CTC_DIST_DIR
)


class ModelComparison:
    """Systematic comparison between MLP and CTC phoneme classification models."""
    
    def __init__(self):
        self.mlp_model = None
        self.mlp_encoder = None
        self.ctc_model = None
        self.ctc_encoder = None
        self.phoneme_labels = None
        self.comparison_results = {}
        
    def load_models(self):
        """Load both MLP and CTC models for comparison."""
        print("🔧 Loading models for comparison...")
        
        try:
            # Load MLP model
            with open(MLP_MODEL_PATH, 'rb') as f:
                self.mlp_model = pickle.load(f)
            with open(MLP_LABEL_ENCODER_PATH, 'rb') as f:
                self.mlp_encoder = pickle.load(f)
            print("✅ MLP model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load MLP model: {e}")
            
        try:
            # Load CTC model
            checkpoint = torch.load(CTC_MODEL_PATH, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Import here to avoid circular imports
                from workflows.ctc_w2v2_workflow.models.ctc_model import CTCModel
                self.ctc_model = CTCModel(num_classes=len(self._load_phoneme_labels()) + 1)  # +1 for blank token
                self.ctc_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.ctc_model = checkpoint
            self.ctc_model.eval()
            
            with open(CTC_LABEL_ENCODER_PATH, 'rb') as f:
                self.ctc_encoder = pickle.load(f)
            print("✅ CTC model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load CTC model: {e}")
            
        # Load phoneme labels
        self.phoneme_labels = self._load_phoneme_labels()
        
    def _load_phoneme_labels(self) -> List[str]:
        """Load phoneme labels from JSON."""
        try:
            with open(MLP_PHONEME_LABELS_JSON_PATH, 'r') as f:
                return json.load(f)
        except:
            with open(CTC_PHONEME_LABELS_JSON_PATH, 'r') as f:
                return json.load(f)
    
    def compare_model_characteristics(self) -> Dict[str, Any]:
        """Compare basic model characteristics."""
        print("📊 Comparing model characteristics...")
        
        characteristics = {
            'mlp': {},
            'ctc': {},
            'comparison': {}
        }
        
        # MLP characteristics
        if self.mlp_model:
            mlp_params = sum(p.numel() for p in self.mlp_model.coefs_[0].flatten()) if hasattr(self.mlp_model, 'coefs_') else 0
            characteristics['mlp'] = {
                'type': 'MLPClassifier (scikit-learn)',
                'parameters': mlp_params,
                'layers': len(self.mlp_model.coefs_) if hasattr(self.mlp_model, 'coefs_') else 0,
                'activation': getattr(self.mlp_model, 'activation', 'unknown'),
                'solver': getattr(self.mlp_model, 'solver', 'unknown')
            }
        
        # CTC characteristics
        if self.ctc_model:
            ctc_params = sum(p.numel() for p in self.ctc_model.parameters())
            characteristics['ctc'] = {
                'type': 'CTC + LSTM (PyTorch)',
                'parameters': ctc_params,
                'input_dim': getattr(self.ctc_model, 'input_dim', 768),
                'hidden_dim': getattr(self.ctc_model, 'hidden_dim', 256),
                'num_classes': getattr(self.ctc_model, 'num_classes', len(self.phoneme_labels))
            }
            
        # Parameter comparison
        if self.mlp_model and self.ctc_model:
            characteristics['comparison']['parameter_ratio'] = characteristics['ctc']['parameters'] / max(characteristics['mlp']['parameters'], 1)
            
        return characteristics
    
    def benchmark_inference_speed(self, num_samples: int = 100) -> Dict[str, Any]:
        """Benchmark inference speed for both models."""
        print(f"⚡ Benchmarking inference speed ({num_samples} samples)...")
        
        # Generate dummy data for benchmarking
        dummy_embeddings = np.random.randn(num_samples, 768).astype(np.float32)
        
        results = {
            'mlp': {'times': [], 'avg_time': 0, 'throughput': 0},
            'ctc': {'times': [], 'avg_time': 0, 'throughput': 0},
            'comparison': {}
        }
        
        # Benchmark MLP
        if self.mlp_model:
            print("  🔄 Benchmarking MLP inference...")
            for i in range(num_samples):
                start_time = time.perf_counter()
                _ = self.mlp_model.predict([dummy_embeddings[i]])
                end_time = time.perf_counter()
                results['mlp']['times'].append(end_time - start_time)
                
            results['mlp']['avg_time'] = np.mean(results['mlp']['times'])
            results['mlp']['std_time'] = np.std(results['mlp']['times'])
            results['mlp']['throughput'] = 1.0 / results['mlp']['avg_time']
        
        # Benchmark CTC
        if self.ctc_model:
            print("  🔄 Benchmarking CTC inference...")
            with torch.no_grad():
                for i in range(num_samples):
                    dummy_sequence = torch.FloatTensor(dummy_embeddings[i:i+1]).unsqueeze(0)
                    start_time = time.perf_counter()
                    _ = self.ctc_model(dummy_sequence)
                    end_time = time.perf_counter()
                    results['ctc']['times'].append(end_time - start_time)
                    
            results['ctc']['avg_time'] = np.mean(results['ctc']['times'])
            results['ctc']['std_time'] = np.std(results['ctc']['times'])
            results['ctc']['throughput'] = 1.0 / results['ctc']['avg_time']
        
        # Speed comparison
        if results['mlp']['avg_time'] > 0 and results['ctc']['avg_time'] > 0:
            results['comparison']['speed_ratio'] = results['ctc']['avg_time'] / results['mlp']['avg_time']
            results['comparison']['faster_model'] = 'MLP' if results['mlp']['avg_time'] < results['ctc']['avg_time'] else 'CTC'
            
        return results
    
    def compare_accuracy_metrics(self, test_embeddings_dir: str) -> Dict[str, Any]:
        """Compare accuracy metrics on test data."""
        print("🎯 Comparing accuracy metrics...")
        
        # This would need actual test data - placeholder for now
        results = {
            'mlp': {'accuracy': 0.0, 'precision': {}, 'recall': {}, 'f1': {}},
            'ctc': {'accuracy': 0.0, 'precision': {}, 'recall': {}, 'f1': {}},
            'comparison': {}
        }
        
        # TODO: Implement actual accuracy comparison with test data
        # This requires loading test embeddings and running inference
        print("⚠️ Accuracy comparison requires test data - implement based on available embeddings")
        
        return results
    
    def generate_comparison_report(self, output_dir: str = None) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        print("📋 Generating comprehensive comparison report...")
        
        if output_dir is None:
            output_dir = str(Path(MLP_DIST_DIR).parent / "model_comparison")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load models if not already loaded
        if not (self.mlp_model or self.ctc_model):
            self.load_models()
        
        # Run all comparisons
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models': {
                'mlp_available': self.mlp_model is not None,
                'ctc_available': self.ctc_model is not None
            },
            'characteristics': self.compare_model_characteristics(),
            'performance': self.benchmark_inference_speed(num_samples=50),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = Path(output_dir) / "model_comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Comparison report saved to: {report_path}")
        self._print_summary(report)
        
        return report
    
    def _generate_recommendations(self) -> Dict[str, str]:
        """Generate usage recommendations based on comparison."""
        recommendations = {}
        
        if self.mlp_model and self.ctc_model:
            recommendations['general'] = "Both models are available for comparison"
            recommendations['speed'] = "MLP typically faster for single phoneme classification"
            recommendations['accuracy'] = "CTC better for sequence modeling and context"
            recommendations['deployment'] = "MLP preferred for real-time applications, CTC for batch processing"
        elif self.mlp_model:
            recommendations['available'] = "Only MLP model available - train CTC for comparison"
        elif self.ctc_model:
            recommendations['available'] = "Only CTC model available - train MLP for comparison"
        else:
            recommendations['error'] = "No models available - train both workflows first"
            
        return recommendations
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print a summary of the comparison results."""
        print("\n" + "="*60)
        print("🏆 MODEL COMPARISON SUMMARY")
        print("="*60)
        
        if report['models']['mlp_available'] and report['models']['ctc_available']:
            chars = report['characteristics']
            perf = report['performance']
            
            print(f"📊 Parameter Count:")
            print(f"   MLP: {chars['mlp'].get('parameters', 'N/A'):,}")
            print(f"   CTC: {chars['ctc'].get('parameters', 'N/A'):,}")
            
            print(f"\n⚡ Inference Speed:")
            print(f"   MLP: {perf['mlp'].get('avg_time', 0)*1000:.2f}ms/sample")
            print(f"   CTC: {perf['ctc'].get('avg_time', 0)*1000:.2f}ms/sample")
            
            if 'faster_model' in perf['comparison']:
                print(f"   Faster: {perf['comparison']['faster_model']}")
                
        print(f"\n💡 Recommendations:")
        for key, value in report['recommendations'].items():
            print(f"   {key}: {value}")
        
        print("="*60)


def main():
    """Run model comparison analysis."""
    comparison = ModelComparison()
    report = comparison.generate_comparison_report()
    return report


if __name__ == "__main__":
    main()