import pytest
import numpy as np
from inference.temporal_brain.hysteresis import HysteresisControl


class TestHysteresisControl:
    def test_initialization(self):
        """Test HysteresisControl initialization with defaults"""
        hysteresis = HysteresisControl()
        assert hysteresis.lock_threshold == 0.7
        assert hysteresis.unlock_threshold == 0.3
        assert hysteresis.min_lock_duration == 3
        assert hysteresis.current_phoneme is None
        assert hysteresis.lock_count == 0
        assert hysteresis.is_locked == False
    
    def test_initialization_with_custom_params(self):
        """Test HysteresisControl initialization with custom parameters"""
        hysteresis = HysteresisControl(
            lock_threshold=0.8,
            unlock_threshold=0.2,
            min_lock_duration=5
        )
        assert hysteresis.lock_threshold == 0.8
        assert hysteresis.unlock_threshold == 0.2
        assert hysteresis.min_lock_duration == 5
    
    def test_no_lock_low_confidence(self):
        """Test no locking with low confidence scores"""
        hysteresis = HysteresisControl(lock_threshold=0.7)
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Low confidence - should not lock
        probabilities = np.array([0.5, 0.3, 0.2])
        result = hysteresis.process(probabilities, phoneme_labels)
        
        assert result is None
        assert not hysteresis.is_locked
        assert hysteresis.current_phoneme is None
    
    def test_lock_on_high_confidence(self):
        """Test locking when confidence exceeds threshold"""
        hysteresis = HysteresisControl(lock_threshold=0.7)
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # High confidence - should lock
        probabilities = np.array([0.8, 0.1, 0.1])
        result = hysteresis.process(probabilities, phoneme_labels)
        
        assert result == 'AA'
        assert hysteresis.is_locked
        assert hysteresis.current_phoneme == 'AA'
        assert hysteresis.lock_count == 1
    
    def test_remain_locked_during_confidence_drop(self):
        """Test remaining locked when confidence drops but above unlock threshold"""
        hysteresis = HysteresisControl(
            lock_threshold=0.7,
            unlock_threshold=0.3,
            min_lock_duration=3
        )
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # First lock
        probs1 = np.array([0.8, 0.1, 0.1])
        result1 = hysteresis.process(probs1, phoneme_labels)
        assert result1 == 'AA'
        
        # Confidence drops but still above unlock threshold
        probs2 = np.array([0.4, 0.3, 0.3])
        result2 = hysteresis.process(probs2, phoneme_labels)
        assert result2 == 'AA'
        assert hysteresis.is_locked
        assert hysteresis.lock_count == 2
    
    def test_unlock_on_confidence_drop(self):
        """Test unlocking when confidence drops below unlock threshold"""
        hysteresis = HysteresisControl(
            lock_threshold=0.7,
            unlock_threshold=0.3,
            min_lock_duration=3
        )
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Lock first
        probs1 = np.array([0.8, 0.1, 0.1])
        hysteresis.process(probs1, phoneme_labels)
        probs2 = np.array([0.5, 0.2, 0.3])
        hysteresis.process(probs2, phoneme_labels) 
        probs3 = np.array([0.4, 0.3, 0.3])
        hysteresis.process(probs3, phoneme_labels)
        
        # Now confidence drops below unlock threshold
        probs4 = np.array([0.2, 0.4, 0.4])
        result = hysteresis.process(probs4, phoneme_labels)
        
        assert result is None
        assert not hysteresis.is_locked
        assert hysteresis.current_phoneme is None
        assert hysteresis.lock_count == 0
    
    def test_min_lock_duration_enforcement(self):
        """Test that minimum lock duration is enforced"""
        hysteresis = HysteresisControl(
            lock_threshold=0.7,
            unlock_threshold=0.3,
            min_lock_duration=5
        )
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Lock first
        probs1 = np.array([0.8, 0.1, 0.1])
        hysteresis.process(probs1, phoneme_labels)
        
        # Try to unlock before min duration (should stay locked)
        for _ in range(4):  # 4 more frames, total count will be 5, equals min_duration
            probs_low = np.array([0.1, 0.4, 0.5])
            result = hysteresis.process(probs_low, phoneme_labels)
            assert result == 'AA'  # Should remain locked
            assert hysteresis.is_locked
        
        # One more frame - now should unlock since count >= min_duration
        probs_low = np.array([0.1, 0.4, 0.5])
        result = hysteresis.process(probs_low, phoneme_labels)
        assert result is None
        assert not hysteresis.is_locked
    
    def test_relock_after_unlock(self):
        """Test relocking on a new phoneme after unlocking"""
        hysteresis = HysteresisControl(
            lock_threshold=0.7,
            unlock_threshold=0.3,
            min_lock_duration=2
        )
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Initial lock on AA
        probs1 = np.array([0.8, 0.1, 0.1])
        result1 = hysteresis.process(probs1, phoneme_labels)
        assert result1 == 'AA'
        
        # Build up lock count
        probs2 = np.array([0.5, 0.2, 0.3])
        hysteresis.process(probs2, phoneme_labels)
        
        # Unlock
        probs3 = np.array([0.2, 0.4, 0.4])
        result3 = hysteresis.process(probs3, phoneme_labels)
        assert result3 is None
        
        # Relock on different phoneme
        probs4 = np.array([0.1, 0.8, 0.1])
        result4 = hysteresis.process(probs4, phoneme_labels)
        assert result4 == 'BB'
        assert hysteresis.current_phoneme == 'BB'
    
    def test_phoneme_not_in_labels_error(self):
        """Test error handling when locked phoneme is not in current labels"""
        hysteresis = HysteresisControl()
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Lock on AA
        probs1 = np.array([0.8, 0.1, 0.1])
        hysteresis.process(probs1, phoneme_labels)
        
        # Change labels without AA - should handle gracefully
        new_labels = ['DD', 'EE', 'FF']
        probs2 = np.array([0.3, 0.3, 0.4])
        
        # This should not crash but might unlock
        with pytest.raises(ValueError):
            hysteresis.process(probs2, new_labels)
    
    def test_edge_case_exact_thresholds(self):
        """Test behavior at exact threshold values"""
        hysteresis = HysteresisControl(
            lock_threshold=0.7,
            unlock_threshold=0.3
        )
        phoneme_labels = ['AA', 'BB']
        
        # Exactly at lock threshold
        probs1 = np.array([0.7, 0.3])
        result1 = hysteresis.process(probs1, phoneme_labels)
        assert result1 == 'AA'  # Should lock
        
        # Exactly at unlock threshold
        probs2 = np.array([0.3, 0.7])
        result2 = hysteresis.process(probs2, phoneme_labels)
        assert result2 == 'AA'  # Should remain locked (>= unlock threshold)
        
        # Slightly below unlock threshold
        probs3 = np.array([0.29, 0.71])
        result3 = hysteresis.process(probs3, phoneme_labels)
        probs4 = np.array([0.29, 0.71])
        result4 = hysteresis.process(probs4, phoneme_labels)
        assert result4 is None  # Should unlock