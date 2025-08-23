#!/usr/bin/env python3
"""
Ensemble Methods - Combine MLP and CTC Model Predictions

This module provides ensemble learning capabilities to combine predictions
from both MLP and CTC models for improved phoneme classification accuracy.
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F

from workflows import (
    MLP_MODEL_PATH, MLP_LABEL_ENCODER_PATH, MLP_PHONEME_LABELS_JSON_PATH,
    CTC_MODEL_PATH, CTC_LABEL_ENCODER_PATH, CTC_PHONEME_LABELS_JSON_PATH,
    MLP_DIST_DIR, CTC_DIST_DIR
)


class EnsembleClassifier:
    """Ensemble classifier combining MLP and CTC model predictions."""
    
    def __init__(self, voting_strategy: str = 'soft', weights: Optional[List[float]] = None):
        """
        Initialize ensemble classifier.
        
        Args:
            voting_strategy: 'soft' (probability averaging) or 'hard' (majority voting)
            weights: Optional weights for each model [mlp_weight, ctc_weight]
        """
        self.voting_strategy = voting_strategy
        self.weights = weights or [0.5, 0.5]  # Equal weights by default
        
        self.mlp_model = None
        self.mlp_encoder = None
        self.ctc_model = None
        self.ctc_encoder = None
        self.phoneme_labels = None
        
        self.is_loaded = False
        
    def load_models(self):
        """Load both MLP and CTC models."""
        print("🔧 Loading models for ensemble...")
        
        try:
            # Load MLP model
            with open(MLP_MODEL_PATH, 'rb') as f:
                self.mlp_model = pickle.load(f)
            with open(MLP_LABEL_ENCODER_PATH, 'rb') as f:
                self.mlp_encoder = pickle.load(f)
            print("✅ MLP model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load MLP model: {e}")
            return False
            
        try:
            # Load CTC model
            checkpoint = torch.load(CTC_MODEL_PATH, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                from workflows.ctc_w2v2_workflow.models.ctc_model import CTCPhonemeClassifier
                self.phoneme_labels = self._load_phoneme_labels()
                self.ctc_model = CTCPhonemeClassifier(num_classes=len(self.phoneme_labels))
                self.ctc_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.ctc_model = checkpoint
            self.ctc_model.eval()
            
            with open(CTC_LABEL_ENCODER_PATH, 'rb') as f:
                self.ctc_encoder = pickle.load(f)
            print("✅ CTC model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load CTC model: {e}")
            return False
        
        # Load phoneme labels
        if not self.phoneme_labels:
            self.phoneme_labels = self._load_phoneme_labels()
            
        self.is_loaded = True
        return True
    
    def _load_phoneme_labels(self) -> List[str]:
        """Load phoneme labels from JSON."""
        try:
            with open(MLP_PHONEME_LABELS_JSON_PATH, 'r') as f:
                return json.load(f)
        except:
            with open(CTC_PHONEME_LABELS_JSON_PATH, 'r') as f:
                return json.load(f)
    
    def predict_mlp(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get MLP predictions and probabilities."""
        if self.mlp_model is None:
            raise ValueError("MLP model not loaded")
            
        # Get predictions and probabilities
        predictions = self.mlp_model.predict(embeddings)
        probabilities = self.mlp_model.predict_proba(embeddings)
        
        return predictions, probabilities
    
    def predict_ctc(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get CTC predictions and probabilities."""
        if self.ctc_model is None:
            raise ValueError("CTC model not loaded")
            
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for embedding in embeddings:
                # Convert to tensor and add batch/sequence dimensions
                input_tensor = torch.FloatTensor(embedding).unsqueeze(0).unsqueeze(0)
                
                # Get model output
                output = self.ctc_model(input_tensor)
                
                # Apply softmax to get probabilities
                probs = F.softmax(output, dim=-1)
                
                # Get the most likely prediction (simple decoding)
                pred_idx = torch.argmax(probs, dim=-1).item()
                predictions.append(pred_idx)
                probabilities.append(probs.squeeze().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def ensemble_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Make ensemble predictions combining both models."""
        if not self.is_loaded:
            if not self.load_models():
                raise ValueError("Failed to load models for ensemble prediction")
        
        # Get predictions from both models
        mlp_preds, mlp_probs = self.predict_mlp(embeddings)
        ctc_preds, ctc_probs = self.predict_ctc(embeddings)
        
        ensemble_preds = []
        ensemble_probs = []
        
        for i in range(len(embeddings)):
            if self.voting_strategy == 'soft':
                # Weighted average of probabilities
                combined_probs = (self.weights[0] * mlp_probs[i] + 
                                self.weights[1] * ctc_probs[i])
                ensemble_pred = np.argmax(combined_probs)
                ensemble_probs.append(combined_probs)
                
            elif self.voting_strategy == 'hard':
                # Majority voting with weights
                votes = {}
                votes[mlp_preds[i]] = votes.get(mlp_preds[i], 0) + self.weights[0]
                votes[ctc_preds[i]] = votes.get(ctc_preds[i], 0) + self.weights[1]
                
                ensemble_pred = max(votes, key=votes.get)
                
                # Create pseudo-probabilities for hard voting
                pseudo_probs = np.zeros(len(self.phoneme_labels))
                pseudo_probs[ensemble_pred] = max(votes.values()) / sum(self.weights)
                ensemble_probs.append(pseudo_probs)
            
            ensemble_preds.append(ensemble_pred)
        
        # Create detailed results
        results = {
            'mlp_predictions': mlp_preds,
            'ctc_predictions': ctc_preds,
            'ensemble_predictions': np.array(ensemble_preds),
            'mlp_probabilities': mlp_probs,
            'ctc_probabilities': ctc_probs,
            'ensemble_probabilities': np.array(ensemble_probs),
            'voting_strategy': self.voting_strategy,
            'weights': self.weights
        }
        
        return np.array(ensemble_preds), np.array(ensemble_probs), results
    
    def optimize_weights(self, validation_embeddings: np.ndarray, 
                        validation_labels: np.ndarray, 
                        weight_range: Tuple[float, float] = (0.0, 1.0),
                        weight_steps: int = 11) -> Dict[str, Any]:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            validation_embeddings: Validation embedding data
            validation_labels: True labels for validation data
            weight_range: Range of weights to search
            weight_steps: Number of weight combinations to try
        
        Returns:
            Dictionary with optimal weights and performance metrics
        """
        print("🔍 Optimizing ensemble weights...")
        
        if not self.is_loaded:
            if not self.load_models():
                raise ValueError("Failed to load models for weight optimization")
        
        # Generate weight combinations
        weights_to_try = []
        weight_values = np.linspace(weight_range[0], weight_range[1], weight_steps)
        
        for w1 in weight_values:
            w2 = 1.0 - w1  # Ensure weights sum to 1
            if weight_range[0] <= w2 <= weight_range[1]:
                weights_to_try.append([w1, w2])
        
        best_accuracy = 0.0
        best_weights = self.weights.copy()
        results = []
        
        print(f"  Testing {len(weights_to_try)} weight combinations...")
        
        for weights in weights_to_try:
            # Temporarily set weights
            original_weights = self.weights.copy()
            self.weights = weights
            
            try:
                # Make predictions with current weights
                ensemble_preds, _, _ = self.ensemble_predict(validation_embeddings)
                
                # Calculate accuracy
                accuracy = accuracy_score(validation_labels, ensemble_preds)
                
                results.append({
                    'weights': weights.copy(),
                    'accuracy': accuracy
                })
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights.copy()
                    
            except Exception as e:
                print(f"⚠️ Error with weights {weights}: {e}")
                
            # Restore original weights
            self.weights = original_weights
        
        # Set best weights
        self.weights = best_weights
        
        optimization_results = {
            'best_weights': best_weights,
            'best_accuracy': best_accuracy,
            'all_results': results,
            'improvement_over_equal_weights': best_accuracy - results[5]['accuracy'] if len(results) > 5 else 0.0  # Assuming middle result is [0.5, 0.5]
        }
        
        print(f"✅ Optimal weights found: MLP={best_weights[0]:.2f}, CTC={best_weights[1]:.2f}")
        print(f"   Best validation accuracy: {best_accuracy:.4f}")
        
        return optimization_results
    
    def evaluate_ensemble(self, test_embeddings: np.ndarray, 
                         test_labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble performance against individual models."""
        print("📊 Evaluating ensemble performance...")
        
        if not self.is_loaded:
            if not self.load_models():
                raise ValueError("Failed to load models for evaluation")
        
        # Get predictions from all methods
        mlp_preds, _ = self.predict_mlp(test_embeddings)
        ctc_preds, _ = self.predict_ctc(test_embeddings)
        ensemble_preds, _, detailed_results = self.ensemble_predict(test_embeddings)
        
        # Calculate accuracies
        mlp_accuracy = accuracy_score(test_labels, mlp_preds)
        ctc_accuracy = accuracy_score(test_labels, ctc_preds)
        ensemble_accuracy = accuracy_score(test_labels, ensemble_preds)
        
        # Generate classification reports
        target_names = self.phoneme_labels
        
        evaluation = {
            'accuracies': {
                'mlp': mlp_accuracy,
                'ctc': ctc_accuracy,
                'ensemble': ensemble_accuracy
            },
            'improvements': {
                'over_mlp': ensemble_accuracy - mlp_accuracy,
                'over_ctc': ensemble_accuracy - ctc_accuracy,
                'over_best_individual': ensemble_accuracy - max(mlp_accuracy, ctc_accuracy)
            },
            'classification_reports': {
                'mlp': classification_report(test_labels, mlp_preds, target_names=target_names, output_dict=True),
                'ctc': classification_report(test_labels, ctc_preds, target_names=target_names, output_dict=True),
                'ensemble': classification_report(test_labels, ensemble_preds, target_names=target_names, output_dict=True)
            },
            'ensemble_config': {
                'voting_strategy': self.voting_strategy,
                'weights': self.weights
            }
        }
        
        return evaluation
    
    def save_ensemble_config(self, output_path: str):
        """Save ensemble configuration for later use."""
        config = {
            'voting_strategy': self.voting_strategy,
            'weights': self.weights,
            'phoneme_labels': self.phoneme_labels,
            'model_paths': {
                'mlp_model': str(MLP_MODEL_PATH),
                'mlp_encoder': str(MLP_LABEL_ENCODER_PATH),
                'ctc_model': str(CTC_MODEL_PATH),
                'ctc_encoder': str(CTC_LABEL_ENCODER_PATH)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Ensemble configuration saved to: {output_path}")
    
    def load_ensemble_config(self, config_path: str):
        """Load ensemble configuration from file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.voting_strategy = config['voting_strategy']
        self.weights = config['weights']
        self.phoneme_labels = config['phoneme_labels']
        
        print(f"📁 Ensemble configuration loaded from: {config_path}")


def create_ensemble_workflow(output_dir: str = None) -> Dict[str, Any]:
    """Create and evaluate ensemble workflow with multiple strategies."""
    if output_dir is None:
        output_dir = str(Path(MLP_DIST_DIR).parent / "ensemble_results")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("🎭 Creating ensemble workflow...")
    
    # Test different ensemble strategies
    strategies = ['soft', 'hard']
    results = {}
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy} voting strategy...")
        
        ensemble = EnsembleClassifier(voting_strategy=strategy)
        
        if not ensemble.load_models():
            print(f"⚠️ Skipping {strategy} strategy - models not available")
            continue
            
        # Save configuration
        config_path = Path(output_dir) / f"ensemble_{strategy}_config.json"
        ensemble.save_ensemble_config(str(config_path))
        
        results[strategy] = {
            'config_path': str(config_path),
            'strategy': strategy,
            'default_weights': ensemble.weights.copy()
        }
    
    # Save overall results
    results_path = Path(output_dir) / "ensemble_workflow_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Ensemble workflow created")
    print(f"📁 Results saved to: {output_dir}")
    
    return results


def main():
    """Run ensemble methods demonstration."""
    results = create_ensemble_workflow()
    return results


if __name__ == "__main__":
    main()