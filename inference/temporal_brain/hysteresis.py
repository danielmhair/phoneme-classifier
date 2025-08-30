"""
Hysteresis control algorithm for temporal brain implementation.

Prevents rapid flip-flopping between phoneme predictions using 
dual-threshold hysteresis with minimum lock duration.
"""
from typing import List, Optional
import numpy as np


class HysteresisControl:
    """Dual-threshold hysteresis control to prevent phoneme flip-flopping.
    
    Uses different thresholds for locking onto and unlocking from phonemes,
    with minimum lock duration to ensure stability.
    """
    
    def __init__(self, 
                 lock_threshold: float = 0.7,    # Higher threshold to lock
                 unlock_threshold: float = 0.3,  # Lower threshold to unlock
                 min_lock_duration: int = 3):    # Minimum frames to stay locked
        """Initialize hysteresis control.
        
        Args:
            lock_threshold: Confidence threshold to lock onto a phoneme
            unlock_threshold: Confidence threshold to unlock from phoneme
            min_lock_duration: Minimum frames to stay locked before unlocking
        """
        self.lock_threshold = lock_threshold
        self.unlock_threshold = unlock_threshold
        self.min_lock_duration = min_lock_duration
        
        # State variables
        self.current_phoneme: Optional[str] = None
        self.lock_count = 0
        self.is_locked = False
    
    def process(self, probabilities: np.ndarray, phoneme_labels: List[str]) -> Optional[str]:
        """Process probabilities through hysteresis control.
        
        Args:
            probabilities: Current frame probability distribution
            phoneme_labels: List of phoneme labels corresponding to probabilities
            
        Returns:
            Current locked phoneme or None if unlocked
            
        Raises:
            ValueError: If current_phoneme is not in phoneme_labels when locked
        """
        max_prob = np.max(probabilities)
        predicted_phoneme = phoneme_labels[np.argmax(probabilities)]
        
        if not self.is_locked:
            # Not locked - check if we should lock onto a phoneme
            if max_prob >= self.lock_threshold:
                self.current_phoneme = predicted_phoneme
                self.is_locked = True
                self.lock_count = 1
        else:
            # Currently locked - check if we should unlock
            if self.current_phoneme not in phoneme_labels:
                raise ValueError(f"Current phoneme '{self.current_phoneme}' not in labels {phoneme_labels}")
            
            current_prob = probabilities[phoneme_labels.index(self.current_phoneme)]
            
            if (current_prob < self.unlock_threshold and 
                self.lock_count >= self.min_lock_duration):
                # Unlock - confidence too low and minimum duration reached
                self.is_locked = False
                self.lock_count = 0
                self.current_phoneme = None
            else:
                # Remain locked - increment counter
                self.lock_count += 1
        
        return self.current_phoneme if self.is_locked else None