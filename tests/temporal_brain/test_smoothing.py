import pytest
import numpy as np
from inference.temporal_brain.smoothing import MovingAverageSmoothing, ExponentialSmoothing


class TestMovingAverageSmoothing:
    def test_initialization(self):
        """Test MovingAverageSmoothing initialization"""
        smoother = MovingAverageSmoothing(window_size=5)
        assert smoother.window_size == 5
        assert len(smoother.history) == 0
    
    def test_smooth_single_frame(self):
        """Test smoothing with single frame returns original probabilities"""
        smoother = MovingAverageSmoothing(window_size=3)
        probs = np.array([0.1, 0.2, 0.7])
        result = smoother.smooth(probs)
        np.testing.assert_array_equal(result, probs)
    
    def test_smooth_insufficient_history(self):
        """Test smoothing with insufficient history returns current frame"""
        smoother = MovingAverageSmoothing(window_size=5)
        probs1 = np.array([0.1, 0.2, 0.7])
        probs2 = np.array([0.2, 0.3, 0.5])
        
        result1 = smoother.smooth(probs1)
        np.testing.assert_array_equal(result1, probs1)
        
        result2 = smoother.smooth(probs2)  
        np.testing.assert_array_equal(result2, probs2)
    
    def test_smooth_with_sufficient_history(self):
        """Test smoothing with sufficient history performs averaging"""
        smoother = MovingAverageSmoothing(window_size=3)
        
        # Add frames to build history
        probs1 = np.array([0.0, 0.0, 1.0])
        probs2 = np.array([0.0, 0.0, 1.0])
        probs3 = np.array([0.0, 0.0, 1.0])
        
        smoother.smooth(probs1)
        smoother.smooth(probs2)
        result = smoother.smooth(probs3)
        
        expected = np.array([0.0, 0.0, 1.0])  # Average of identical frames
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_smooth_averaging_calculation(self):
        """Test correct averaging calculation"""
        smoother = MovingAverageSmoothing(window_size=3)
        
        probs1 = np.array([1.0, 0.0, 0.0])
        probs2 = np.array([0.0, 1.0, 0.0])
        probs3 = np.array([0.0, 0.0, 1.0])
        
        smoother.smooth(probs1)
        smoother.smooth(probs2) 
        result = smoother.smooth(probs3)
        
        # Should be average: (1,0,0) + (0,1,0) + (0,0,1) / 3 = (0.33, 0.33, 0.33)
        expected = np.array([1.0/3, 1.0/3, 1.0/3])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_window_size_enforcement(self):
        """Test that history is limited by window size"""
        smoother = MovingAverageSmoothing(window_size=2)
        
        probs1 = np.array([1.0, 0.0])
        probs2 = np.array([0.0, 1.0])
        probs3 = np.array([0.5, 0.5])
        
        smoother.smooth(probs1)
        smoother.smooth(probs2)
        result = smoother.smooth(probs3)
        
        # Should only use last 2 frames: (0,1) + (0.5,0.5) / 2 = (0.25, 0.75)
        expected = np.array([0.25, 0.75])
        np.testing.assert_array_almost_equal(result, expected)


class TestExponentialSmoothing:
    def test_initialization(self):
        """Test ExponentialSmoothing initialization"""
        smoother = ExponentialSmoothing(alpha=0.3)
        assert smoother.alpha == 0.3
        assert smoother.smoothed is None
    
    def test_smooth_first_frame(self):
        """Test first frame smoothing initializes state"""
        smoother = ExponentialSmoothing(alpha=0.3)
        probs = np.array([0.1, 0.2, 0.7])
        result = smoother.smooth(probs)
        
        np.testing.assert_array_equal(result, probs)
        np.testing.assert_array_equal(smoother.smoothed, probs)
    
    def test_smooth_exponential_calculation(self):
        """Test exponential smoothing calculation"""
        smoother = ExponentialSmoothing(alpha=0.5)
        
        probs1 = np.array([1.0, 0.0, 0.0])
        probs2 = np.array([0.0, 1.0, 0.0])
        
        result1 = smoother.smooth(probs1)
        np.testing.assert_array_equal(result1, probs1)
        
        result2 = smoother.smooth(probs2)
        # Should be: 0.5 * probs2 + 0.5 * probs1 = [0.5, 0.5, 0.0]
        expected = np.array([0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result2, expected)
    
    def test_smooth_alpha_effect(self):
        """Test alpha parameter effect on smoothing"""
        # High alpha (0.9) - more responsive to current frame
        smoother_high = ExponentialSmoothing(alpha=0.9)
        # Low alpha (0.1) - more emphasis on history
        smoother_low = ExponentialSmoothing(alpha=0.1)
        
        probs1 = np.array([1.0, 0.0])
        probs2 = np.array([0.0, 1.0])
        
        # Initialize both smoothers
        smoother_high.smooth(probs1)
        smoother_low.smooth(probs1)
        
        # Apply second frame
        result_high = smoother_high.smooth(probs2)
        result_low = smoother_low.smooth(probs2)
        
        # High alpha should be closer to new frame
        assert result_high[1] > result_low[1]
        assert result_high[0] < result_low[0]
    
    def test_smooth_convergence(self):
        """Test smoothing converges to repeated input"""
        smoother = ExponentialSmoothing(alpha=0.3)
        target = np.array([0.2, 0.8])
        
        # Apply same frame multiple times
        result = target.copy()
        for _ in range(20):  # Enough iterations to converge
            result = smoother.smooth(target)
        
        # Should converge close to target
        np.testing.assert_array_almost_equal(result, target, decimal=3)