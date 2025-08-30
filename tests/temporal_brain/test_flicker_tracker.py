import pytest
from inference.temporal_brain.flicker_tracker import FlickerTracker


class TestFlickerTracker:
    def test_initialization(self):
        """Test FlickerTracker initialization"""
        tracker = FlickerTracker()
        assert tracker.window_size == 100
        assert len(tracker.history) == 0
        assert tracker.transitions == 0
    
    def test_initialization_with_custom_window(self):
        """Test FlickerTracker initialization with custom window size"""
        tracker = FlickerTracker(window_size=50)
        assert tracker.window_size == 50
    
    def test_first_update_no_transition(self):
        """Test first update doesn't count as transition"""
        tracker = FlickerTracker()
        tracker.update('AA')
        
        assert len(tracker.history) == 1
        assert tracker.transitions == 0
        assert tracker.get_flicker_rate() == 0.0
    
    def test_same_phoneme_no_transition(self):
        """Test repeated same phoneme doesn't count as transition"""
        tracker = FlickerTracker()
        tracker.update('AA')
        tracker.update('AA')
        tracker.update('AA')
        
        assert len(tracker.history) == 3
        assert tracker.transitions == 0
        assert tracker.get_flicker_rate() == 0.0
    
    def test_phoneme_change_counts_transition(self):
        """Test phoneme change counts as transition"""
        tracker = FlickerTracker()
        tracker.update('AA')
        tracker.update('BB')
        
        assert len(tracker.history) == 2
        assert tracker.transitions == 1
        assert tracker.get_flicker_rate() == 1.0 / 2  # 1 transition out of 2 frames
    
    def test_multiple_transitions(self):
        """Test multiple transitions are counted correctly"""
        tracker = FlickerTracker()
        phonemes = ['AA', 'BB', 'AA', 'CC', 'CC']
        
        for phoneme in phonemes:
            tracker.update(phoneme)
        
        # Transitions: AA->BB, BB->AA, AA->CC (3 transitions)
        assert tracker.transitions == 3
        assert tracker.get_flicker_rate() == 3.0 / 5  # 3 transitions out of 5 frames
    
    def test_none_phoneme_handling(self):
        """Test None phonemes are handled correctly"""
        tracker = FlickerTracker()
        tracker.update('AA')
        tracker.update(None)
        tracker.update('BB')
        tracker.update(None)
        
        # Transitions: AA->None, None->BB, BB->None (3 transitions)
        assert tracker.transitions == 3
        assert tracker.get_flicker_rate() == 3.0 / 4
    
    def test_window_size_enforcement(self):
        """Test that history is limited by window size"""
        tracker = FlickerTracker(window_size=3)
        
        # Add more than window size
        phonemes = ['AA', 'BB', 'CC', 'DD', 'EE']
        for phoneme in phonemes:
            tracker.update(phoneme)
        
        # Should only keep last 3
        assert len(tracker.history) == 3
        assert list(tracker.history) == ['CC', 'DD', 'EE']
        # Should have approximate transition count (transitions are estimated when window full)
        assert tracker.transitions <= 4  # Should be reduced when old transitions removed
    
    def test_flicker_rate_calculation(self):
        """Test flicker rate calculation with various scenarios"""
        tracker = FlickerTracker()
        
        # No history
        assert tracker.get_flicker_rate() == 0.0
        
        # Single frame
        tracker.update('AA')
        assert tracker.get_flicker_rate() == 0.0
        
        # Two frames, same phoneme
        tracker.update('AA')
        assert tracker.get_flicker_rate() == 0.0
        
        # Two frames, different phoneme
        tracker.history.clear()
        tracker.transitions = 0
        tracker.update('AA')
        tracker.update('BB')
        assert tracker.get_flicker_rate() == 0.5  # 1 transition / 2 frames
    
    def test_high_flicker_scenario(self):
        """Test high flicker scenario (alternating phonemes)"""
        tracker = FlickerTracker()
        
        # Alternating pattern - maximum flicker
        for i in range(10):
            phoneme = 'AA' if i % 2 == 0 else 'BB'
            tracker.update(phoneme)
        
        # Should have 9 transitions (alternating every frame)
        assert tracker.transitions == 9
        assert tracker.get_flicker_rate() == 0.9  # 9 transitions / 10 frames
    
    def test_low_flicker_scenario(self):
        """Test low flicker scenario (stable phonemes with occasional changes)"""
        tracker = FlickerTracker()
        
        # Stable segments with occasional changes
        phonemes = ['AA'] * 5 + ['BB'] * 5 + ['CC'] * 5  # Only 2 transitions total
        for phoneme in phonemes:
            tracker.update(phoneme)
        
        assert tracker.transitions == 2
        assert abs(tracker.get_flicker_rate() - 2.0/15) < 0.001  # ~0.133
    
    def test_transition_approximation_with_window(self):
        """Test transition count approximation when window is full"""
        tracker = FlickerTracker(window_size=5)
        
        # Fill beyond window size with alternating pattern
        phonemes = ['AA', 'BB'] * 10  # Many transitions
        for phoneme in phonemes:
            tracker.update(phoneme)
        
        # Should have approximate transition count (not exact due to approximation)
        # But should be reasonable for a high-flicker pattern
        assert tracker.get_flicker_rate() > 0.5  # Should be high flicker
        assert len(tracker.history) == 5  # Window size enforced
    
    def test_edge_case_empty_and_none(self):
        """Test edge cases with empty updates and None values"""
        tracker = FlickerTracker()
        
        # Multiple None values
        tracker.update(None)
        tracker.update(None)
        assert tracker.transitions == 0
        assert tracker.get_flicker_rate() == 0.0
        
        # None to phoneme
        tracker.update('AA')
        assert tracker.transitions == 1
        
        # Phoneme to None
        tracker.update(None)
        assert tracker.transitions == 2