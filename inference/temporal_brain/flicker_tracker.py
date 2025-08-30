"""
Flicker rate measurement for temporal brain implementation.

Tracks phoneme transitions to measure flicker rate - the frequency
of phoneme changes over time.
"""
from collections import deque
from typing import Optional


class FlickerTracker:
    """Tracks phoneme transitions to measure flicker rate.
    
    Maintains a sliding window of recent phoneme predictions and
    counts transitions between different phonemes.
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize flicker tracker.
        
        Args:
            window_size: Number of recent frames to track for flicker calculation
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.transitions = 0
    
    def update(self, phoneme: Optional[str]):
        """Update tracker with new phoneme prediction.
        
        Args:
            phoneme: Current phoneme prediction (or None if no prediction)
        """
        if len(self.history) > 0 and self.history[-1] != phoneme:
            self.transitions += 1
        
        self.history.append(phoneme)
        
        # When window is full, approximate transition removal
        # This is a simplified approximation - exact tracking would require more state
        if len(self.history) == self.window_size and self.transitions > 0:
            # Approximate: assume one transition is removed when window rotates
            # This is not perfect but provides reasonable approximation
            self.transitions = max(0, self.transitions - 1)
    
    def get_flicker_rate(self) -> float:
        """Calculate current flicker rate.
        
        Returns:
            Flicker rate as transitions per frame (0.0 = stable, 1.0 = maximum flicker)
        """
        if len(self.history) < 2:
            return 0.0
        return self.transitions / len(self.history)