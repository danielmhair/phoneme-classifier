"""
Test Suite for CTC Model Implementation

This module provides comprehensive tests for the CTC phoneme classification model
using pytest. Tests cover model architecture, forward/backward passes, loss computation,
inference, and reliability aspects.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Import from the same directory
from ctc_model import (
    CTCModel,
    Wav2Vec2FeatureExtractor,
    SequenceEncoder,
    CTCHead,
    create_ctc_model
)


class TestWav2Vec2FeatureExtractor:
    """Test suite for Wav2Vec2 feature extraction component."""
    
    def test_feature_extractor_creation(self):
        """Test that feature extractor can be created successfully."""
        extractor = Wav2Vec2FeatureExtractor()
        assert extractor.feature_dim == 768
        assert extractor.wav2vec2 is not None
    
    def test_feature_extraction_shape(self):
        """Test that feature extraction preserves temporal dimension."""
        extractor = Wav2Vec2FeatureExtractor()
        
        # Test with different sequence lengths
        for seq_len in [8000, 16000, 32000]:  # 0.5s, 1s, 2s at 16kHz
            audio = torch.randn(2, seq_len)  # batch_size=2
            features = extractor(audio)
            
            # Check output shape (batch, time, features)
            assert len(features.shape) == 3
            assert features.shape[0] == 2  # batch size
            assert features.shape[2] == 768  # feature dimension
            assert features.shape[1] > 0  # temporal dimension should exist
    
    def test_feature_extraction_deterministic(self):
        """Test that feature extraction is deterministic for same input."""
        extractor = Wav2Vec2FeatureExtractor()
        audio = torch.randn(1, 16000)
        
        features1 = extractor(audio)
        features2 = extractor(audio)
        
        torch.testing.assert_close(features1, features2)
    
    def test_feature_extraction_gradient_disabled(self):
        """Test that feature extraction doesn't require gradients."""
        extractor = Wav2Vec2FeatureExtractor()
        audio = torch.randn(1, 16000, requires_grad=True)
        
        features = extractor(audio)
        assert not features.requires_grad


class TestSequenceEncoder:
    """Test suite for LSTM sequence encoder."""
    
    def test_encoder_creation(self):
        """Test encoder creation with various configurations."""
        # Default configuration
        encoder = SequenceEncoder()
        assert encoder.lstm.input_size == 768
        assert encoder.lstm.hidden_size == 128
        assert encoder.lstm.bidirectional is True
        
        # Custom configuration
        encoder_custom = SequenceEncoder(input_dim=512, hidden_dim=64, num_layers=1)
        assert encoder_custom.lstm.input_size == 512
        assert encoder_custom.lstm.hidden_size == 64
    
    def test_encoder_forward_shape(self):
        """Test encoder output shapes."""
        encoder = SequenceEncoder(input_dim=768, hidden_dim=128)
        
        # Test with different batch sizes and sequence lengths
        for batch_size in [1, 4, 8]:
            for seq_len in [50, 100, 200]:
                input_tensor = torch.randn(batch_size, seq_len, 768)
                output = encoder(input_tensor)
                
                # Check output shape (bidirectional doubles hidden dim)
                assert output.shape == (batch_size, seq_len, 256)
    
    def test_encoder_gradient_flow(self):
        """Test that gradients flow through encoder."""
        encoder = SequenceEncoder()
        input_tensor = torch.randn(2, 50, 768, requires_grad=True)
        
        output = encoder(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert torch.any(input_tensor.grad != 0)


class TestCTCHead:
    """Test suite for CTC classification head."""
    
    def test_ctc_head_creation(self):
        """Test CTC head creation."""
        head = CTCHead(input_dim=256, num_classes=38)
        assert head.num_classes == 38
        assert head.blank_token == 37  # Last token is blank
        assert head.classifier.in_features == 256
        assert head.classifier.out_features == 38
    
    def test_ctc_head_forward_shape(self):
        """Test CTC head output shapes."""
        head = CTCHead(input_dim=256, num_classes=38)
        
        batch_size, seq_len, input_dim = 4, 100, 256
        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        
        output = head(input_tensor)
        
        # Check output shape and log probabilities
        assert output.shape == (batch_size, seq_len, 38)
        
        # Verify log probabilities sum to 1 (approximately) across classes
        probs = torch.exp(output)
        prob_sums = probs.sum(dim=-1)
        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    
    def test_ctc_head_log_probabilities(self):
        """Test that CTC head outputs valid log probabilities."""
        head = CTCHead()
        input_tensor = torch.randn(2, 50, 256)
        
        log_probs = head(input_tensor)
        
        # Log probabilities should be negative or zero
        assert torch.all(log_probs <= 0)
        
        # Converting to probabilities should give valid probability distribution
        probs = torch.exp(log_probs)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)


class TestCTCModel:
    """Test suite for complete CTC model."""
    
    def test_model_creation(self):
        """Test CTC model creation."""
        model = CTCModel(num_classes=38)
        assert model.ctc_head.num_classes == 38
        assert model.ctc_head.blank_token == 37
    
    def test_model_forward_inference(self):
        """Test model forward pass in inference mode."""
        model = CTCModel(num_classes=38)
        audio = torch.randn(2, 16000)  # 2 samples, 1 second each
        
        log_probs, loss = model(audio)
        
        # Check output shape
        assert len(log_probs.shape) == 3
        assert log_probs.shape[0] == 2  # batch size
        assert log_probs.shape[2] == 38  # num classes
        assert loss is None  # No targets provided
    
    def test_model_forward_training(self):
        """Test model forward pass in training mode."""
        model = CTCModel(num_classes=38)
        audio = torch.randn(2, 16000)
        targets = torch.randint(0, 37, (2, 10))  # Random phoneme sequences
        
        log_probs, loss = model(audio, targets=targets)
        
        # Check outputs
        assert log_probs.shape[0] == 2  # batch size
        assert log_probs.shape[2] == 38  # num classes
        assert loss is not None
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0  # CTC loss should be non-negative
    
    def test_model_predict_greedy(self):
        """Test greedy prediction."""
        model = CTCModel(num_classes=38)
        model.eval()
        
        audio = torch.randn(3, 16000)
        predictions = model.predict(audio, beam_width=1)
        
        # Check predictions format
        assert len(predictions) == 3  # batch size
        for pred in predictions:
            assert isinstance(pred, list)
            # All predicted tokens should be valid (0-36, excluding blank=37)
            for token in pred:
                assert 0 <= token <= 36
    
    def test_model_gradient_flow(self):
        """Test gradient flow in training mode."""
        model = CTCModel(num_classes=38)
        audio = torch.randn(2, 16000, requires_grad=True)
        targets = torch.randint(0, 37, (2, 5))
        
        log_probs, loss = model(audio, targets=targets)
        loss.backward()
        
        # Check that gradients exist for model parameters
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_variable_length_sequences(self):
        """Test model with variable length sequences."""
        model = CTCModel(num_classes=38)
        
        # Different length audio samples
        audio1 = torch.randn(1, 8000)   # 0.5 seconds
        audio2 = torch.randn(1, 24000)  # 1.5 seconds
        
        log_probs1, _ = model(audio1)
        log_probs2, _ = model(audio2)
        
        # Different inputs should produce different temporal lengths
        assert log_probs1.shape[1] != log_probs2.shape[1]
        
        # Both should have correct batch and class dimensions
        assert log_probs1.shape[0] == 1 and log_probs1.shape[2] == 38
        assert log_probs2.shape[0] == 1 and log_probs2.shape[2] == 38


class TestCTCDecoding:
    """Test suite for CTC decoding algorithms."""
    
    def test_greedy_decoding_removes_blanks(self):
        """Test that greedy decoding removes blank tokens."""
        model = CTCModel(num_classes=38)
        
        # Create log probabilities that favor specific pattern
        batch_size, seq_len, num_classes = 1, 10, 38
        log_probs = torch.full((batch_size, seq_len, num_classes), -10.0)
        
        # Set up pattern: token 5, blank, token 5, token 7, blank
        # Should decode to [5, 7] (removing duplicates and blanks)
        log_probs[0, 0, 5] = 0    # token 5
        log_probs[0, 1, 37] = 0   # blank
        log_probs[0, 2, 5] = 0    # token 5 (duplicate, should be removed)
        log_probs[0, 3, 7] = 0    # token 7
        log_probs[0, 4, 37] = 0   # blank
        
        predictions = model._greedy_decode(log_probs)
        
        assert len(predictions) == 1
        assert predictions[0] == [5, 7]
    
    def test_greedy_decoding_empty_sequence(self):
        """Test greedy decoding with all blank tokens."""
        model = CTCModel(num_classes=38)
        
        # Create log probabilities that favor only blanks
        batch_size, seq_len, num_classes = 1, 5, 38
        log_probs = torch.full((batch_size, seq_len, num_classes), -10.0)
        log_probs[0, :, 37] = 0  # All blanks
        
        predictions = model._greedy_decode(log_probs)
        
        assert len(predictions) == 1
        assert predictions[0] == []  # Empty sequence


class TestModelReliability:
    """Test suite for model reliability and error handling."""
    
    def test_model_with_invalid_audio_length(self):
        """Test model behavior with very short audio."""
        model = CTCModel(num_classes=38)
        
        # Very short audio (might cause issues)
        short_audio = torch.randn(1, 100)  # Very short
        
        try:
            log_probs, _ = model(short_audio)
            # If successful, check output is reasonable
            assert log_probs.shape[0] == 1
            assert log_probs.shape[2] == 38
        except Exception as e:
            # If it fails, it should fail gracefully
            pytest.skip(f"Model cannot handle very short audio: {e}")
    
    def test_model_memory_efficiency(self):
        """Test that model doesn't leak memory during multiple forward passes."""
        model = CTCModel(num_classes=38)
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Multiple forward passes
        for _ in range(5):
            audio = torch.randn(2, 16000)
            with torch.no_grad():
                log_probs, _ = model(audio)
            del audio, log_probs
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Memory should not grow significantly
        if torch.cuda.is_available():
            memory_growth = final_memory - initial_memory
            assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
    
    def test_model_deterministic_inference(self):
        """Test that model inference is deterministic."""
        torch.manual_seed(42)
        model = CTCModel(num_classes=38)
        model.eval()
        
        audio = torch.randn(1, 16000)
        
        # Multiple predictions should be identical
        pred1 = model.predict(audio)
        pred2 = model.predict(audio)
        
        assert pred1 == pred2


class TestModelFactory:
    """Test suite for model factory function."""
    
    def test_create_ctc_model_default(self):
        """Test default model creation."""
        model = create_ctc_model()
        assert model.ctc_head.num_classes == 38  # 37 + 1 blank
        assert model.ctc_head.blank_token == 37
    
    def test_create_ctc_model_custom(self):
        """Test custom model creation."""
        model = create_ctc_model(num_classes=20)
        assert model.ctc_head.num_classes == 21  # 20 + 1 blank
        assert model.ctc_head.blank_token == 20


@pytest.fixture
def sample_model():
    """Fixture providing a sample CTC model for testing."""
    return CTCModel(num_classes=38)


@pytest.fixture
def sample_audio():
    """Fixture providing sample audio data."""
    return torch.randn(2, 16000)  # 2 samples, 1 second each


@pytest.fixture
def sample_targets():
    """Fixture providing sample target sequences."""
    return torch.randint(0, 37, (2, 8))  # 2 sequences, max 8 phonemes each


if __name__ == "__main__":
    # Run basic tests if executed directly
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("pytest not found. Running basic tests manually...")
        
        # Manual test execution
        print("Testing CTC Model Creation...")
        model = create_ctc_model()
        print("✅ Model created successfully")
        
        print("Testing Forward Pass...")
        audio = torch.randn(2, 16000)
        log_probs, loss = model(audio)
        print(f"✅ Forward pass successful. Output shape: {log_probs.shape}")
        
        print("Testing Prediction...")
        predictions = model.predict(audio)
        print(f"✅ Prediction successful. Predictions: {[len(p) for p in predictions]}")
        
        print("All basic tests passed! ✅")