"""
CTC Model Architecture for Phoneme Classification

This module implements a Connectionist Temporal Classification (CTC) model
for phoneme recognition using PyTorch. The model combines Wav2Vec2 feature
extraction with LSTM sequence modeling and CTC loss for alignment-free
phoneme sequence prediction.
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from typing import Tuple, Optional, List


class Wav2Vec2FeatureExtractor(nn.Module):
    """Wrapper for Wav2Vec2 feature extraction that preserves temporal sequences."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.feature_dim = self.wav2vec2.config.hidden_size  # 768

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features from audio.

        Args:
            input_values: Audio input tensor (batch_size, sequence_length)

        Returns:
            Temporal features (batch_size, time_steps, feature_dim)
        """
        with torch.no_grad():
            outputs = self.wav2vec2(input_values)
        # Return full temporal sequence, no mean pooling
        return outputs.last_hidden_state


class SequenceEncoder(nn.Module):
    """LSTM-based sequence encoder for temporal modeling."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal sequences.

        Args:
            x: Input features (batch_size, time_steps, input_dim)

        Returns:
            Encoded sequences (batch_size, time_steps, hidden_dim * 2)
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        return self.dropout(lstm_out)


class CTCHead(nn.Module):
    """CTC classification head with blank token."""

    def __init__(self, input_dim: int = 256, num_classes: int = 38):  # 37 phonemes + 1 blank
        super().__init__()
        self.num_classes = num_classes
        self.blank_token = num_classes - 1  # Last token is blank
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate CTC logits.

        Args:
            x: Encoded sequences (batch_size, time_steps, input_dim)

        Returns:
            CTC logits (batch_size, time_steps, num_classes)
        """
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)


class CTCModel(nn.Module):
    """Complete CTC model for phoneme classification."""

    def __init__(
        self,
        embedding_dim: int = 768,  # Wav2Vec2 base embedding dimension
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 38,
        dropout: float = 0.1
    ):
        super().__init__()
        # Use pre-extracted embeddings instead of feature extractor
        self.embedding_dim = embedding_dim
        self.sequence_encoder = SequenceEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.ctc_head = CTCHead(
            input_dim=hidden_dim * 2,  # bidirectional
            num_classes=num_classes
        )
        self.ctc_loss = nn.CTCLoss(blank=num_classes-1, zero_infinity=True)

    def forward(
        self, 
        input_values: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of CTC model.

        Args:
            input_values: Pre-extracted embeddings (batch_size, seq_len, embedding_dim)
            targets: Target phoneme sequences (batch_size, target_length) - optional for inference
            input_lengths: Actual lengths of input sequences - optional
            target_lengths: Actual lengths of target sequences - optional

        Returns:
            Tuple of (log_probs, loss)
            - log_probs: CTC log probabilities (batch_size, time_steps, num_classes)
            - loss: CTC loss if targets provided, else None
        """
        # Skip feature extraction - input_values are already embeddings
        # input_values shape: (batch_size, seq_len, embedding_dim)
        features = input_values

        # Encode sequences
        encoded = self.sequence_encoder(features)

        # Generate CTC logits
        log_probs = self.ctc_head(encoded)

        loss = None
        if targets is not None:
            # Prepare for CTC loss computation
            log_probs_transposed = log_probs.transpose(0, 1)  # (time_steps, batch_size, num_classes)

            # Use provided lengths or compute them
            if input_lengths is None:
                input_lengths = torch.full((log_probs.size(0),), log_probs.size(1), dtype=torch.long)
            if target_lengths is None:
                target_lengths = torch.full((targets.size(0),), targets.size(1), dtype=torch.long)

            # Compute CTC loss
            loss = self.ctc_loss(log_probs_transposed, targets, input_lengths, target_lengths)

        return log_probs, loss

    def predict(self, input_values: torch.Tensor, beam_width: int = 1) -> List[List[int]]:
        """
        Predict phoneme sequences using CTC decoding.

        Args:
            input_values: Audio input (batch_size, sequence_length)
            beam_width: Beam search width (1 for greedy decoding)

        Returns:
            List of predicted phoneme sequences for each sample in batch
        """
        self.eval()
        with torch.no_grad():
            log_probs, _ = self.forward(input_values)

            if beam_width == 1:
                return self._greedy_decode(log_probs)
            else:
                return self._beam_search_decode(log_probs, beam_width)

    def _greedy_decode(self, log_probs: torch.Tensor) -> List[List[int]]:
        """Greedy CTC decoding."""
        predictions = []

        for log_prob in log_probs:  # Iterate over batch
            # Get most likely token at each timestep
            tokens = torch.argmax(log_prob, dim=-1)

            # Remove consecutive duplicates and blank tokens
            decoded = []
            prev_token = None

            for token in tokens:
                token = token.item()
                if token != self.ctc_head.blank_token and token != prev_token:
                    decoded.append(token)
                prev_token = token

            predictions.append(decoded)

        return predictions

    def _beam_search_decode(self, log_probs: torch.Tensor, beam_width: int) -> List[List[int]]:
        """Beam search CTC decoding - simplified implementation."""
        # For now, fallback to greedy decoding
        # TODO: Implement proper beam search
        return self._greedy_decode(log_probs)


def create_ctc_model(num_classes: int = 37) -> CTCModel:
    """
    Factory function to create CTC model with default configuration.

    Args:
        num_classes: Number of phoneme classes (excluding blank token)

    Returns:
        Configured CTCModel instance
    """
    return CTCModel(
        embedding_dim=768,  # Wav2Vec2 base embedding dimension
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes + 1,  # +1 for blank token
        dropout=0.1
    )


if __name__ == "__main__":
    # Basic model test
    model = create_ctc_model(num_classes=37)

    # Test forward pass
    batch_size = 2
    sequence_length = 16000  # 1 second at 16kHz
    dummy_audio = torch.randn(batch_size, sequence_length)

    # Test inference
    predictions = model.predict(dummy_audio)
    print(f"Model created successfully. Predictions shape: {[len(p) for p in predictions]}")

    # Test training mode
    dummy_targets = torch.randint(0, 37, (batch_size, 10))  # Random phoneme sequences
    log_probs, loss = model(dummy_audio, targets=dummy_targets)
    print(f"Training mode - Log probs shape: {log_probs.shape}, Loss: {loss.item():.4f}")
