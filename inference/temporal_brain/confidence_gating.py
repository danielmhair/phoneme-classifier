"""
Confidence gating algorithm for temporal brain implementation.

Implements adaptive confidence thresholds with temporal persistence
to ensure only high-confidence, stable phonemes are emitted.
"""
from typing import List, Tuple, Optional, Dict
import numpy as np


class ConfidenceGating:
    """Adaptive confidence gating with temporal persistence.
    
    Only emits phonemes that maintain confidence above threshold
    for a minimum number of consecutive frames.
    """
    
    def __init__(self, 
                 default_threshold: float = 0.6,
                 persistence_frames: int = 3,
                 phoneme_thresholds: Optional[Dict[str, float]] = None,
                 phoneme_persistence: Optional[Dict[str, int]] = None):
        """Initialize confidence gating.
        
        Args:
            default_threshold: Default confidence threshold for all phonemes
            persistence_frames: Default number of frames phoneme must persist above threshold
            phoneme_thresholds: Per-phoneme confidence thresholds
            phoneme_persistence: Per-phoneme persistence frame requirements
        """
        self.default_threshold = default_threshold
        self.persistence_frames = persistence_frames
        self.phoneme_thresholds = phoneme_thresholds or {}
        self.phoneme_persistence = phoneme_persistence or {}
        
        # State variables for candidate tracking
        self.candidate_phoneme: Optional[str] = None
        self.candidate_count = 0
        self.candidate_confidence_sum = 0.0
    
    def get_threshold(self, phoneme: str) -> float:
        """Get confidence threshold for specific phoneme.
        
        Args:
            phoneme: Phoneme label
            
        Returns:
            Confidence threshold for this phoneme
        """
        return self.phoneme_thresholds.get(phoneme, self.default_threshold)
    
    def get_persistence(self, phoneme: str) -> int:
        """Get persistence requirement for specific phoneme.
        
        Args:
            phoneme: Phoneme label
            
        Returns:
            Number of frames this phoneme must persist
        """
        return self.phoneme_persistence.get(phoneme, self.persistence_frames)
    
    def process(self, probabilities: np.ndarray, phoneme_labels: List[str]) -> Tuple[Optional[str], float]:
        """Process probabilities through confidence gating.
        
        Args:
            probabilities: Current frame probability distribution
            phoneme_labels: List of phoneme labels corresponding to probabilities
            
        Returns:
            Tuple of (phoneme, confidence) or (None, 0.0) if gated
        """
        max_idx = np.argmax(probabilities)
        predicted_phoneme = phoneme_labels[max_idx]
        confidence = probabilities[max_idx]
        threshold = self.get_threshold(predicted_phoneme)
        
        if confidence < threshold:
            # Below threshold - reset candidate
            self._reset_candidate()
            return None, 0.0
            
        if predicted_phoneme == self.candidate_phoneme:
            # Same candidate - accumulate
            self.candidate_count += 1
            self.candidate_confidence_sum += confidence
        else:
            # New candidate - start tracking
            self.candidate_phoneme = predicted_phoneme
            self.candidate_count = 1
            self.candidate_confidence_sum = confidence
        
        # Check if candidate has persisted long enough (phoneme-specific)
        required_persistence = self.get_persistence(predicted_phoneme)
        if self.candidate_count >= required_persistence:
            # Candidate has persisted long enough
            avg_confidence = self.candidate_confidence_sum / self.candidate_count
            self._reset_candidate()
            return predicted_phoneme, avg_confidence
        
        return None, 0.0
    
    def _reset_candidate(self):
        """Reset candidate tracking state."""
        self.candidate_phoneme = None
        self.candidate_count = 0
        self.candidate_confidence_sum = 0.0