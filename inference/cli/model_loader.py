"""
ONNX model loader for Epic 2 temporal brain CLI tool.

Supports loading all three model types from Epic 1:
- MLP Control (two-stage: Wav2Vec2 ONNX + MLP ONNX)
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
        
        # For MLP two-stage inference
        self.wav2vec_session: Optional[ort.InferenceSession] = None
        self.mlp_session: Optional[ort.InferenceSession] = None
        
        # Discover available models
        self._discover_models()
    
    def _discover_models(self):
        """Discover available ONNX models from Epic 1 workflows."""
        model_configs = [
            # MLP Control workflow models (two-stage inference working)
            {
                'model_id': 'mlp_control',
                'name': 'MLP Control',
                'path': 'phoneme_mlp.onnx',
                'labels_path': 'phoneme_labels.json',
                'model_type': 'mlp',
                'description': 'Traditional MLP classifier from Epic 1 (two-stage: Wav2Vec2 → MLP)'
            },
            # NOTE: CTC models not yet properly exported to ONNX for temporal brain
            # The wav2vec2.onnx currently only contains feature extractor, not full classifier
            # Wav2Vec2 CTC workflow models (pending proper ONNX export)
            # {
            #     'model_id': 'wav2vec2_ctc',
            #     'name': 'Wav2Vec2 CTC',
            #     'path': 'wav2vec2_ctc.onnx',  # This doesn't exist yet
            #     'labels_path': 'phoneme_labels.json',
            #     'model_type': 'ctc',
            #     'description': 'Facebook Wav2Vec2 CTC model from Epic 1'
            # },
            # WavLM CTC workflow models (pending proper ONNX export)  
            # {
            #     'model_id': 'wavlm_ctc',
            #     'name': 'WavLM CTC',
            #     'path': 'wavlm_ctc.onnx',  # This doesn't exist yet
            #     'labels_path': 'phoneme_labels.json',
            #     'model_type': 'ctc',
            #     'description': 'Microsoft WavLM CTC model from Epic 1 (85.35% accuracy)'
            # }
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
            # Load phoneme labels first
            with open(labels_path, 'r') as f:
                self.current_labels = json.load(f)
            
            if model_info.model_type == 'mlp':
                # MLP requires two-stage loading: Wav2Vec2 + MLP
                wav2vec_path = self.models_dir / "wav2vec2.onnx"
                mlp_path = model_path  # This should be phoneme_mlp.onnx
                
                if not wav2vec_path.exists():
                    print(f"❌ Wav2Vec2 feature extractor not found: {wav2vec_path}")
                    return False
                
                # Load both stages
                self.wav2vec_session = ort.InferenceSession(str(wav2vec_path))
                self.mlp_session = ort.InferenceSession(str(mlp_path))
                self.current_session = None  # Not used for two-stage
                
                print(f"✅ Loaded MLP Control (two-stage):")
                print(f"   🧠 Stage 1: Wav2Vec2 feature extractor")
                print(f"   🎯 Stage 2: MLP classifier")
                
            else:
                # CTC models: direct single-stage inference
                self.current_session = ort.InferenceSession(str(model_path))
                self.wav2vec_session = None
                self.mlp_session = None
                
                print(f"✅ Loaded {model_info.name} (single-stage)")
                print(f"   📁 Path: {model_path}")
            
            self.current_model_info = model_info
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
            audio_features: Preprocessed audio features (format depends on model type)
            
        Returns:
            Probability distribution over phonemes
            
        Raises:
            RuntimeError: If no model is loaded or inference fails
        """
        if self.current_model_info is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        try:
            if self.current_model_info.model_type == 'mlp':
                return self._run_mlp_inference(audio_features)
            else:
                return self._run_ctc_inference(audio_features)
                
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
    
    def _run_mlp_inference(self, wav2vec_features: np.ndarray) -> np.ndarray:
        """Run two-stage MLP inference: Wav2Vec2 → embeddings → MLP → probabilities."""
        if self.wav2vec_session is None or self.mlp_session is None:
            raise RuntimeError("MLP two-stage models not loaded properly")
        
        # Stage 1: Extract embeddings using Wav2Vec2 ONNX
        wav2vec_input_name = self.wav2vec_session.get_inputs()[0].name
        wav2vec_outputs = self.wav2vec_session.run(None, {wav2vec_input_name: wav2vec_features})
        embeddings = wav2vec_outputs[0]  # Shape: [1, T, H]
        
        # Stage 2: Classify using MLP ONNX
        mlp_input_name = self.mlp_session.get_inputs()[0].name
        mlp_outputs = self.mlp_session.run(None, {mlp_input_name: embeddings})
        logits = mlp_outputs[0]  # Shape: [1, num_classes]
        
        # Extract probabilities from first batch item
        if len(logits.shape) > 1:
            logits = logits[0]  # Shape: [num_classes]
        
        # Apply softmax to convert logits to probabilities
        probabilities = self._softmax(logits)
        return probabilities
    
    def _run_ctc_inference(self, audio_features: np.ndarray) -> np.ndarray:
        """Run single-stage CTC inference: audio → probabilities."""
        if self.current_session is None:
            raise RuntimeError("CTC model not loaded properly")
        
        # Get input/output names
        input_name = self.current_session.get_inputs()[0].name
        output_name = self.current_session.get_outputs()[0].name
        
        # Ensure correct input shape
        if len(audio_features.shape) == 1:
            audio_features = audio_features.reshape(1, -1)
        
        # Run inference
        outputs = self.current_session.run(
            [output_name], 
            {input_name: audio_features}
        )
        
        probabilities = outputs[0]
        
        # Handle different output shapes
        if len(probabilities.shape) > 1:
            probabilities = probabilities[0]  # Take first batch item
        
        # Apply softmax for CTC models
        probabilities = self._softmax(probabilities)
        return probabilities
    
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