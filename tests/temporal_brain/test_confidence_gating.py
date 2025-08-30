import pytest
import numpy as np
from inference.temporal_brain.confidence_gating import ConfidenceGating


class TestConfidenceGating:
    def test_initialization(self):
        """Test ConfidenceGating initialization with defaults"""
        gate = ConfidenceGating()
        assert gate.default_threshold == 0.6
        assert gate.persistence_frames == 3
        assert gate.phoneme_thresholds == {}
        assert gate.candidate_phoneme is None
        assert gate.candidate_count == 0
        assert gate.candidate_confidence_sum == 0.0
    
    def test_initialization_with_custom_params(self):
        """Test ConfidenceGating initialization with custom parameters"""
        custom_thresholds = {'AA': 0.8, 'BB': 0.5}
        gate = ConfidenceGating(
            default_threshold=0.7,
            persistence_frames=5,
            phoneme_thresholds=custom_thresholds
        )
        assert gate.default_threshold == 0.7
        assert gate.persistence_frames == 5
        assert gate.phoneme_thresholds == custom_thresholds
    
    def test_get_threshold_default(self):
        """Test getting default threshold for unknown phoneme"""
        gate = ConfidenceGating(default_threshold=0.6)
        assert gate.get_threshold('UNKNOWN') == 0.6
    
    def test_get_threshold_custom(self):
        """Test getting custom threshold for specific phoneme"""
        gate = ConfidenceGating(
            default_threshold=0.6,
            phoneme_thresholds={'AA': 0.8, 'BB': 0.4}
        )
        assert gate.get_threshold('AA') == 0.8
        assert gate.get_threshold('BB') == 0.4
        assert gate.get_threshold('CC') == 0.6  # Default
    
    def test_below_threshold_returns_none(self):
        """Test that confidence below threshold returns None"""
        gate = ConfidenceGating(default_threshold=0.6)
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Low confidence - should return None
        probabilities = np.array([0.5, 0.3, 0.2])
        result_phoneme, result_confidence = gate.process(probabilities, phoneme_labels)
        
        assert result_phoneme is None
        assert result_confidence == 0.0
        assert gate.candidate_phoneme is None
    
    def test_above_threshold_starts_candidate_tracking(self):
        """Test that confidence above threshold starts tracking candidate"""
        gate = ConfidenceGating(default_threshold=0.6, persistence_frames=3)
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # High confidence - should start tracking
        probabilities = np.array([0.8, 0.1, 0.1])
        result_phoneme, result_confidence = gate.process(probabilities, phoneme_labels)
        
        assert result_phoneme is None  # Not persisted enough yet
        assert result_confidence == 0.0
        assert gate.candidate_phoneme == 'AA'
        assert gate.candidate_count == 1
        assert gate.candidate_confidence_sum == 0.8
    
    def test_candidate_persistence_required(self):
        """Test that candidate must persist for required frames"""
        gate = ConfidenceGating(default_threshold=0.6, persistence_frames=3)
        phoneme_labels = ['AA', 'BB', 'CC']
        
        probabilities = np.array([0.8, 0.1, 0.1])
        
        # Frame 1
        result1 = gate.process(probabilities, phoneme_labels)
        assert result1 == (None, 0.0)
        assert gate.candidate_count == 1
        
        # Frame 2
        result2 = gate.process(probabilities, phoneme_labels)
        assert result2 == (None, 0.0)
        assert gate.candidate_count == 2
        
        # Frame 3 - should return result now
        result3 = gate.process(probabilities, phoneme_labels)
        assert result3[0] == 'AA'
        assert abs(result3[1] - 0.8) < 0.001  # Average confidence (handle floating point precision)
        assert gate.candidate_phoneme is None  # Reset after emission
    
    def test_candidate_change_resets_tracking(self):
        """Test that changing candidate resets tracking"""
        gate = ConfidenceGating(default_threshold=0.6, persistence_frames=3)
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Start with AA
        probs1 = np.array([0.8, 0.1, 0.1])
        gate.process(probs1, phoneme_labels)
        assert gate.candidate_phoneme == 'AA'
        assert gate.candidate_count == 1
        
        # Change to BB - should reset
        probs2 = np.array([0.1, 0.9, 0.0])
        result = gate.process(probs2, phoneme_labels)
        assert result == (None, 0.0)
        assert gate.candidate_phoneme == 'BB'
        assert gate.candidate_count == 1
        assert gate.candidate_confidence_sum == 0.9
    
    def test_confidence_averaging(self):
        """Test correct averaging of confidence values"""
        gate = ConfidenceGating(default_threshold=0.6, persistence_frames=3)
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Different confidence values for same phoneme
        probs1 = np.array([0.7, 0.2, 0.1])  # 0.7
        probs2 = np.array([0.8, 0.1, 0.1])  # 0.8
        probs3 = np.array([0.9, 0.05, 0.05])  # 0.9
        
        gate.process(probs1, phoneme_labels)
        gate.process(probs2, phoneme_labels)
        result_phoneme, result_confidence = gate.process(probs3, phoneme_labels)
        
        expected_avg = (0.7 + 0.8 + 0.9) / 3
        assert result_phoneme == 'AA'
        assert abs(result_confidence - expected_avg) < 0.001
    
    def test_per_phoneme_thresholds(self):
        """Test that per-phoneme thresholds are respected"""
        gate = ConfidenceGating(
            default_threshold=0.6,
            persistence_frames=1,  # Make it easy to test
            phoneme_thresholds={'AA': 0.9, 'BB': 0.3}  # AA requires higher confidence
        )
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Test AA with confidence 0.7 (above default but below AA threshold)
        probs_aa = np.array([0.7, 0.2, 0.1])
        result_aa = gate.process(probs_aa, phoneme_labels)
        assert result_aa == (None, 0.0)  # Should be rejected
        
        # Test BB with confidence 0.5 (above BB threshold)
        probs_bb = np.array([0.2, 0.5, 0.3])
        result_bb = gate.process(probs_bb, phoneme_labels)
        assert result_bb[0] == 'BB'  # Should be accepted
        
        # Test CC with default threshold
        probs_cc = np.array([0.2, 0.1, 0.7])
        result_cc = gate.process(probs_cc, phoneme_labels)
        assert result_cc[0] == 'CC'  # Should be accepted with default threshold
    
    def test_confidence_drop_resets_candidate(self):
        """Test that confidence drop below threshold resets candidate"""
        gate = ConfidenceGating(default_threshold=0.6, persistence_frames=3)
        phoneme_labels = ['AA', 'BB', 'CC']
        
        # Start building candidate
        probs1 = np.array([0.8, 0.1, 0.1])
        gate.process(probs1, phoneme_labels)
        assert gate.candidate_count == 1
        
        probs2 = np.array([0.7, 0.2, 0.1])
        gate.process(probs2, phoneme_labels)
        assert gate.candidate_count == 2
        
        # Confidence drops - should reset
        probs3 = np.array([0.5, 0.3, 0.2])
        result = gate.process(probs3, phoneme_labels)
        assert result == (None, 0.0)
        assert gate.candidate_phoneme is None
        assert gate.candidate_count == 0
    
    def test_reset_after_emission(self):
        """Test that state is reset after successful emission"""
        gate = ConfidenceGating(default_threshold=0.6, persistence_frames=2)
        phoneme_labels = ['AA', 'BB']
        
        probabilities = np.array([0.8, 0.2])
        
        # Build up to emission
        gate.process(probabilities, phoneme_labels)
        result = gate.process(probabilities, phoneme_labels)
        
        # Should emit and reset
        assert result[0] == 'AA'
        assert gate.candidate_phoneme is None
        assert gate.candidate_count == 0
        assert gate.candidate_confidence_sum == 0.0
    
    def test_edge_case_exact_threshold(self):
        """Test behavior at exact threshold values"""
        gate = ConfidenceGating(default_threshold=0.6, persistence_frames=1)
        phoneme_labels = ['AA', 'BB']
        
        # Exactly at threshold
        probs_exact = np.array([0.6, 0.4])
        result = gate.process(probs_exact, phoneme_labels)
        assert result[0] == 'AA'  # Should be accepted
        
        # Just below threshold
        probs_below = np.array([0.59, 0.41])
        result = gate.process(probs_below, phoneme_labels)
        assert result == (None, 0.0)  # Should be rejected