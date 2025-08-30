"""
Smoothing algorithms for temporal brain implementation.

Reduces high-frequency noise in probability sequences to provide
more stable phoneme detection.
"""
from collections import deque
from typing import Optional
import numpy as np


class MovingAverageSmoothing:
    """Moving average smoothing for probability sequences.
    
    Reduces high-frequency noise by averaging probabilities over
    a fixed time window.
    """
    
    def __init__(self, window_size: int = 5):
        """Initialize moving average smoother.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def smooth(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing to probability vector.
        
        Args:
            probabilities: Current frame probabilities
            
        Returns:
            Smoothed probabilities (average over window)
        """
        self.history.append(probabilities)
        
        if len(self.history) < self.window_size:
            return probabilities  # Not enough history
        
        # Average across time window
        return np.mean(self.history, axis=0)


class ExponentialSmoothing:
    """Exponential smoothing for probability sequences.
    
    Weights recent observations more heavily while maintaining
    influence from historical data.
    """
    
    def __init__(self, alpha: float = 0.3):
        """Initialize exponential smoother.
        
        Args:
            alpha: Smoothing factor (0=no smoothing, 1=no history)
        """
        self.alpha = alpha
        self.smoothed: Optional[np.ndarray] = None
    
    def smooth(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to probability vector.
        
        Args:
            probabilities: Current frame probabilities
            
        Returns:
            Exponentially smoothed probabilities
        """
        if self.smoothed is None:
            self.smoothed = probabilities.copy()
        else:
            self.smoothed = (self.alpha * probabilities + 
                           (1 - self.alpha) * self.smoothed)
        return self.smoothed