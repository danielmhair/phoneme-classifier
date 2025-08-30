"""
Audio feature extraction system for Epic 2 temporal brain CLI tool.

Handles different preprocessing pipelines for each Epic 1 model type:
- MLP Control: Audio → Wav2Vec2 features → MLP classifier
- Wav2Vec2 CTC: Audio → Direct CTC inference  
- WavLM CTC: Audio → Direct CTC inference
"""
import numpy as np
import librosa
from typing import Optional, Dict, Any
from transformers import Wav2Vec2Processor


class AudioFeatureExtractor:
    """Extracts audio features compatible with different Epic 1 model types."""
    
    def __init__(self):
        """Initialize the audio feature extractor."""
        self._wav2vec_processor: Optional[Wav2Vec2Processor] = None
        self._sample_rate = 16000
        
        # Audio preprocessing constants
        self.MIN_AMP_THRESHOLD = 0.01
        self.BUFFER_SAMPLES = 1000
        
    def _get_wav2vec_processor(self) -> Wav2Vec2Processor:
        """Get or create Wav2Vec2 processor for MLP models."""
        if self._wav2vec_processor is None:
            try:
                self._wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Wav2Vec2 processor: {e}")
        return self._wav2vec_processor
    
    def preprocess_audio(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess raw audio data with silence trimming and normalization.
        
        Args:
            audio_data: Raw audio data from capture
            
        Returns:
            Preprocessed audio or None if silent
        """
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data.T)
        
        # Find non-silent regions
        nonzero_indices = np.where(np.abs(audio_data) > self.MIN_AMP_THRESHOLD)[0]
        
        if len(nonzero_indices) == 0:
            return None  # Silent audio
        
        # Trim silence with buffer
        start = max(0, nonzero_indices[0] - self.BUFFER_SAMPLES)
        end = min(len(audio_data), nonzero_indices[-1] + self.BUFFER_SAMPLES)
        trimmed = audio_data[start:end]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(trimmed))
        if max_val > 0:
            trimmed = trimmed / max_val
        
        return trimmed.astype(np.float32)
    
    def extract_features_for_mlp(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract Wav2Vec2 features for MLP Control model.
        
        This is Stage 1 of the MLP pipeline:
        Audio → Wav2Vec2 ONNX → Embeddings (for Stage 2: MLP ONNX)
        
        Args:
            audio_data: Preprocessed audio data
            
        Returns:
            Wav2Vec2 feature embeddings ready for MLP classifier
        """
        processor = self._get_wav2vec_processor()
        
        # Process audio through Wav2Vec2 processor
        inputs = processor(
            audio_data, 
            sampling_rate=self._sample_rate, 
            return_tensors="np", 
            padding=True
        )
        
        # Return input_values for ONNX inference
        # Shape: [1, sequence_length] for ONNX wav2vec2 model
        return inputs['input_values'].astype(np.float32)
    
    def extract_features_for_ctc(self, audio_data: np.ndarray, model_type: str = "wav2vec2") -> np.ndarray:
        """
        Extract features for CTC models (direct audio input).
        
        Args:
            audio_data: Preprocessed audio data
            model_type: "wav2vec2" or "wavlm" for different CTC models
            
        Returns:
            Audio features ready for CTC model inference
        """
        # For CTC models, we typically feed raw audio directly
        # But may need resampling to ensure 16kHz
        
        # Ensure correct sample rate
        if audio_data.shape[0] < self._sample_rate:
            # Pad short audio to minimum length
            pad_length = self._sample_rate - audio_data.shape[0]
            audio_data = np.pad(audio_data, (0, pad_length), mode='constant')
        
        # Add batch dimension if needed: [sequence] → [1, sequence]
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)
        
        return audio_data.astype(np.float32)
    
    def extract_features_for_model(self, audio_data: np.ndarray, model_type: str) -> Dict[str, Any]:
        """
        Extract features based on model type.
        
        Args:
            audio_data: Raw audio data from capture
            model_type: "mlp", "ctc", or specific model identifier
            
        Returns:
            Dictionary with extracted features and metadata
        """
        # Preprocess audio (silence trimming, normalization)
        processed_audio = self.preprocess_audio(audio_data)
        
        if processed_audio is None:
            return {
                'features': None,
                'is_silent': True,
                'audio_length': 0,
                'preprocessing_success': False
            }
        
        try:
            if model_type == 'mlp':
                # MLP requires Wav2Vec2 feature extraction
                features = self.extract_features_for_mlp(processed_audio)
                feature_type = 'wav2vec2_features'
            elif model_type == 'ctc':
                # CTC models take direct audio input
                features = self.extract_features_for_ctc(processed_audio, "wav2vec2")
                feature_type = 'raw_audio'
            elif model_type == 'wavlm_ctc':
                # WavLM CTC model
                features = self.extract_features_for_ctc(processed_audio, "wavlm")
                feature_type = 'raw_audio'
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return {
                'features': features,
                'feature_type': feature_type,
                'is_silent': False,
                'audio_length': len(processed_audio),
                'original_length': len(audio_data),
                'preprocessing_success': True,
                'sample_rate': self._sample_rate
            }
            
        except Exception as e:
            return {
                'features': None,
                'is_silent': False,
                'audio_length': len(processed_audio) if processed_audio is not None else 0,
                'preprocessing_success': False,
                'error': str(e)
            }


def get_default_feature_extractor() -> AudioFeatureExtractor:
    """Get default audio feature extractor for Epic 2."""
    return AudioFeatureExtractor()