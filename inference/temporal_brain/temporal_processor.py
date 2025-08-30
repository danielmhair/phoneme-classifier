"""
Complete temporal processor combining all temporal brain algorithms.

Integrates smoothing, hysteresis control, and confidence gating to
provide stable real-time phoneme detection with flicker measurement.
"""
import json
import time
from typing import Dict, Any, Optional, List
import numpy as np

from .smoothing import MovingAverageSmoothing, ExponentialSmoothing
from .hysteresis import HysteresisControl
from .confidence_gating import ConfidenceGating
from .flicker_tracker import FlickerTracker


class TemporalProcessor:
    """Main temporal brain processor combining all stabilization algorithms.
    
    Provides a complete pipeline for stable phoneme recognition:
    Raw probabilities → Smoothing → Hysteresis → Confidence Gating → Stable output
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize temporal processor with configuration.
        
        Args:
            config: Configuration dictionary containing all algorithm parameters
        """
        self.config = config
        
        # Initialize components
        self.smoother = self._create_smoother(config)
        self.hysteresis = HysteresisControl(
            lock_threshold=config['hysteresis']['lock_threshold'],
            unlock_threshold=config['hysteresis']['unlock_threshold'],
            min_lock_duration=config['hysteresis']['min_lock_duration']
        )
        self.confidence_gate = ConfidenceGating(
            default_threshold=config['confidence']['default_threshold'],
            persistence_frames=config['confidence']['persistence_frames'],
            phoneme_thresholds=config['confidence']['phoneme_thresholds']
        )
        
        self.phoneme_labels = config['phoneme_labels']
        self.flicker_tracker = FlickerTracker(
            window_size=config['metrics']['flicker_window_size']
        )
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'TemporalProcessor':
        """Create temporal processor from JSON configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configured TemporalProcessor instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)
    
    def process_frame(self, raw_probabilities: np.ndarray) -> Dict[str, Any]:
        """Process a single frame through the complete temporal brain pipeline.
        
        Args:
            raw_probabilities: Raw probability distribution from model
            
        Returns:
            Dictionary containing processed results and metadata
        """
        # Step 1: Smooth probabilities to reduce noise
        smoothed_probs = self.smoother.smooth(raw_probabilities)
        
        # Step 2: Apply hysteresis control for stability
        hysteresis_result = self.hysteresis.process(smoothed_probs, self.phoneme_labels)
        
        # Step 3: Apply confidence gating for reliability
        if hysteresis_result:
            gated_phoneme, confidence = self.confidence_gate.process(
                smoothed_probs, self.phoneme_labels
            )
            # Only emit if both hysteresis and confidence gate agree
            final_phoneme = gated_phoneme if gated_phoneme == hysteresis_result else None
            final_confidence = confidence if final_phoneme else 0.0
        else:
            final_phoneme, final_confidence = None, 0.0
        
        # Step 4: Track flicker for metrics
        self.flicker_tracker.update(final_phoneme)
        
        return {
            'phoneme': final_phoneme,
            'confidence': final_confidence,
            'raw_probabilities': raw_probabilities,
            'smoothed_probabilities': smoothed_probs,
            'hysteresis_result': hysteresis_result,
            'is_stable': final_phoneme is not None,
            'flicker_rate': self.flicker_tracker.get_flicker_rate(),
            'timestamp': time.time()
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current temporal brain metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            'flicker_rate': self.flicker_tracker.get_flicker_rate(),
            'target_flicker_rate': self.config['metrics']['target_flicker_rate'],
            'flicker_performance': max(0.0, 1.0 - (self.flicker_tracker.get_flicker_rate() / 
                                                   self.config['metrics']['target_flicker_rate'])),
            'hysteresis_lock_count': self.hysteresis.lock_count,
            'confidence_candidate_count': self.confidence_gate.candidate_count
        }
    
    def reset(self):
        """Reset all temporal brain components to initial state."""
        # Reset smoother (create new instance to clear history)
        self.smoother = self._create_smoother(self.config)
        
        # Reset hysteresis
        self.hysteresis.current_phoneme = None
        self.hysteresis.lock_count = 0
        self.hysteresis.is_locked = False
        
        # Reset confidence gating
        self.confidence_gate._reset_candidate()
        
        # Reset flicker tracker
        self.flicker_tracker = FlickerTracker(
            window_size=self.config['metrics']['flicker_window_size']
        )
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration and reinitialize components.
        
        Args:
            new_config: New configuration dictionary
        """
        self.config = new_config
        
        # Reinitialize all components with new config
        self.smoother = self._create_smoother(new_config)
        self.hysteresis = HysteresisControl(
            lock_threshold=new_config['hysteresis']['lock_threshold'],
            unlock_threshold=new_config['hysteresis']['unlock_threshold'],
            min_lock_duration=new_config['hysteresis']['min_lock_duration']
        )
        self.confidence_gate = ConfidenceGating(
            default_threshold=new_config['confidence']['default_threshold'],
            persistence_frames=new_config['confidence']['persistence_frames'],
            phoneme_thresholds=new_config['confidence']['phoneme_thresholds']
        )
        self.phoneme_labels = new_config['phoneme_labels']
        
        # Keep existing flicker tracker but update window if changed
        if new_config['metrics']['flicker_window_size'] != self.flicker_tracker.window_size:
            self.flicker_tracker = FlickerTracker(
                window_size=new_config['metrics']['flicker_window_size']
            )
    
    def _create_smoother(self, config: Dict[str, Any]):
        """Create smoother instance based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured smoother instance
            
        Raises:
            ValueError: If smoother type is unknown
        """
        smoother_type = config['smoothing']['type']
        if smoother_type == 'moving_average':
            return MovingAverageSmoothing(config['smoothing']['window_size'])
        elif smoother_type == 'exponential':
            return ExponentialSmoothing(config['smoothing']['alpha'])
        else:
            raise ValueError(f"Unknown smoother type: {smoother_type}")
    
    def __repr__(self) -> str:
        """String representation of the temporal processor."""
        return (f"TemporalProcessor("
                f"smoother={type(self.smoother).__name__}, "
                f"flicker_rate={self.flicker_tracker.get_flicker_rate():.3f})")