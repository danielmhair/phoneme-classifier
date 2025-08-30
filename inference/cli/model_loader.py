"""
ONNX model loader for Epic 2 temporal brain CLI tool.

Supports loading all three model types from Epic 1:
- MLP Control (converted to ONNX)
- Wav2Vec2 CTC (direct ONNX)
- WavLM CTC (direct ONNX)
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import onnxruntime as ort


class ModelInfo:
    """Information about a loaded model."""
    
    def __init__(self, model_id: str, name: str, path: str, labels_path: str, 
                 model_type: str, description: str = ""):
        self.model_id = model_id
        self.name = name
        self.path = path
        self.labels_path = labels_path
        self.model_type = model_type
        self.description = description


class ModelLoader:
    """Loads and manages ONNX models from Epic 1 for temporal brain testing."""
    
    def __init__(self, models_dir: str = "dist"):
        """Initialize model loader.
        
        Args:
            models_dir: Directory containing Epic 1 generated models
        """
        self.models_dir = Path(models_dir)
        self.available_models: Dict[str, ModelInfo] = {}
        self.current_session: Optional[ort.InferenceSession] = None
        self.current_model_info: Optional[ModelInfo] = None
        self.current_labels: List[str] = []
        
        # Discover available models
        self._discover_models()
    
    def _discover_models(self):
        """Discover available ONNX models from Epic 1 workflows."""
        model_configs = [
            # MLP Control workflow models
            {
                'model_id': 'mlp_control',
                'name': 'MLP Control',
                'path': 'phoneme_mlp.onnx',
                'labels_path': 'phoneme_labels.json',
                'model_type': 'mlp',
                'description': 'Traditional MLP classifier from Epic 1'
            },
            # Wav2Vec2 CTC workflow models  
            {
                'model_id': 'wav2vec2_ctc',
                'name': 'Wav2Vec2 CTC',
                'path': 'wav2vec2.onnx',
                'labels_path': 'phoneme_labels.json',
                'model_type': 'ctc',
                'description': 'Facebook Wav2Vec2 CTC model from Epic 1'
            },
            # WavLM CTC workflow models (if available)
            {
                'model_id': 'wavlm_ctc',
                'name': 'WavLM CTC',
                'path': 'wavlm.onnx',
                'labels_path': 'phoneme_labels.json',
                'model_type': 'ctc',
                'description': 'Microsoft WavLM CTC model from Epic 1 (85.35% accuracy)'
            }
        ]
        
        for config in model_configs:
            model_path = self.models_dir / config['path']
            labels_path = self.models_dir / config['labels_path']
            
            if model_path.exists() and labels_path.exists():
                self.available_models[config['model_id']] = ModelInfo(**config)
            else:
                print(f"⚠️  Model not found: {config['name']} at {model_path}")
    
    def list_available_models(self) -> List[ModelInfo]:
        """Get list of available models.
        
        Returns:
            List of ModelInfo objects for available models
        """
        return list(self.available_models.values())
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model by ID.
        
        Args:
            model_id: Model identifier ('mlp_control', 'wav2vec2_ctc', 'wavlm_ctc')
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if model_id not in self.available_models:
            print(f"❌ Model '{model_id}' not available")
            return False
        
        model_info = self.available_models[model_id]
        model_path = self.models_dir / model_info.path
        labels_path = self.models_dir / model_info.labels_path
        
        try:
            # Load ONNX session
            self.current_session = ort.InferenceSession(str(model_path))
            
            # Load phoneme labels
            with open(labels_path, 'r') as f:
                self.current_labels = json.load(f)
            
            self.current_model_info = model_info
            
            print(f"✅ Loaded model: {model_info.name}")
            print(f"   📁 Path: {model_path}")
            print(f"   🏷️  Labels: {len(self.current_labels)} phonemes")
            print(f"   📝 Type: {model_info.model_type.upper()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model '{model_id}': {e}")
            return False
    
    def get_current_model(self) -> Optional[ModelInfo]:
        """Get information about currently loaded model.
        
        Returns:
            ModelInfo object or None if no model loaded
        """
        return self.current_model_info
    
    def get_phoneme_labels(self) -> List[str]:
        """Get phoneme labels for current model.
        
        Returns:
            List of phoneme label strings
        """
        return self.current_labels.copy()
    
    def run_inference(self, audio_features: np.ndarray) -> np.ndarray:
        """Run inference on audio features.
        
        Args:
            audio_features: Preprocessed audio features
            
        Returns:
            Probability distribution over phonemes
            
        Raises:
            RuntimeError: If no model is loaded
            ValueError: If input shape is incorrect
        """
        if self.current_session is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Get input/output names
        input_name = self.current_session.get_inputs()[0].name
        output_name = self.current_session.get_outputs()[0].name
        
        # Ensure correct input shape
        if len(audio_features.shape) == 1:
            audio_features = audio_features.reshape(1, -1)
        
        try:
            # Run inference
            outputs = self.current_session.run(
                [output_name], 
                {input_name: audio_features}
            )
            
            probabilities = outputs[0]
            
            # Handle different output shapes
            if len(probabilities.shape) > 1:
                probabilities = probabilities[0]  # Take first batch item
            
            # Apply softmax if needed (CTC models might need this)
            if self.current_model_info.model_type == 'ctc':
                probabilities = self._softmax(probabilities)
            
            return probabilities
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
    
    def get_model_info_summary(self) -> str:
        """Get formatted summary of current model.
        
        Returns:
            Formatted string with model information
        """
        if not self.current_model_info:
            return "No model loaded"
        
        return (f"Model: {self.current_model_info.name}\n"
                f"Type: {self.current_model_info.model_type.upper()}\n"
                f"Phonemes: {len(self.current_labels)}\n"
                f"Description: {self.current_model_info.description}")
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax normalization.
        
        Args:
            x: Input array
            
        Returns:
            Softmax normalized array
        """
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def __repr__(self) -> str:
        """String representation of model loader."""
        current = self.current_model_info.name if self.current_model_info else "None"
        available = len(self.available_models)
        return f"ModelLoader(current={current}, available={available})"


def get_default_model_loader() -> ModelLoader:
    """Get default model loader configured for Epic 1 models.
    
    Returns:
        Configured ModelLoader instance
    """
    return ModelLoader()