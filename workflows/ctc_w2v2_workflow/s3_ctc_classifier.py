"""
CTC Classifier Training Script

This script trains a CTC (Connectionist Temporal Classification) model for phoneme recognition.
It replaces the traditional MLP classifier with a sequence-to-sequence CTC model that can handle
variable-length audio inputs and outputs without requiring alignment.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.ctc_model import create_ctc_model


class PhonemeDataset(Dataset):
    """Dataset for CTC training with phoneme sequences."""

    def __init__(self, embeddings_dir: str, phoneme_labels: List[str], max_sequence_length: int = 1000):
        """
        Initialize dataset.

        Args:
            embeddings_dir: Directory containing embedding files
            phoneme_labels: List of phoneme labels
            max_sequence_length: Maximum sequence length for memory management
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.phoneme_labels = phoneme_labels
        self.max_sequence_length = max_sequence_length
        self.label_to_idx = {label: idx for idx, label in enumerate(phoneme_labels)}

        # Load metadata
        metadata_path = self.embeddings_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.metadata = pd.read_csv(metadata_path)
        print(f"📊 Loaded {len(self.metadata)} samples from {embeddings_dir}")

        # Filter out samples that don't exist
        self.valid_samples = []
        for _, row in self.metadata.iterrows():
            emb_path = self.embeddings_dir / row["embedding_filename"]
            if emb_path.exists():
                self.valid_samples.append(row)

        print(f"✅ Found {len(self.valid_samples)} valid embedding files")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.valid_samples[idx]

        # Load embedding
        emb_path = self.embeddings_dir / row["embedding_filename"]
        embedding = np.load(emb_path)

        # Handle different embedding shapes
        if len(embedding.shape) == 1:
            # Single vector (current format) - convert to sequence of length 1
            # This is a compatibility layer for existing embeddings
            embedding = embedding.reshape(1, -1)
        elif len(embedding.shape) == 2:
            # Already temporal sequence (new CTC format)
            pass
        else:
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")

        # Limit sequence length for memory management
        if embedding.shape[0] > self.max_sequence_length:
            embedding = embedding[:self.max_sequence_length]

        # Convert phoneme label to index
        phoneme = row["phoneme"]
        if phoneme not in self.label_to_idx:
            raise ValueError(f"Unknown phoneme: {phoneme}")

        phoneme_idx = self.label_to_idx[phoneme]

        return {
            'embedding': torch.FloatTensor(embedding),
            'phoneme': torch.LongTensor([phoneme_idx]),  # Single phoneme for now
            'phoneme_length': torch.LongTensor([1]),     # Length of target sequence
            'embedding_length': torch.LongTensor([embedding.shape[0]])  # Length of input sequence
        }


def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable-length sequences.
    """
    embeddings = [item['embedding'] for item in batch]
    phonemes = [item['phoneme'] for item in batch]
    phoneme_lengths = torch.stack([item['phoneme_length'] for item in batch])
    embedding_lengths = torch.stack([item['embedding_length'] for item in batch])

    # Pad sequences
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)

    return {
        'embeddings': embeddings_padded,
        'phonemes': phonemes_padded,
        'phoneme_lengths': phoneme_lengths.squeeze(),
        'embedding_lengths': embedding_lengths.squeeze()
    }


class CTCTrainer:
    """Trainer class for CTC model."""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose="True"
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            embeddings = batch['embeddings'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            embedding_lengths = batch['embedding_lengths'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Note: For now we use embeddings directly. In full implementation,
            # we would use audio and extract features in the model
            _, loss = self.model(
                input_values=embeddings.mean(dim=2),  # Dummy audio input
                targets=phonemes,
                input_lengths=embedding_lengths,
                target_lengths=phoneme_lengths
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embeddings'].to(self.device)
                phonemes = batch['phonemes'].to(self.device)
                phoneme_lengths = batch['phoneme_lengths'].to(self.device)
                embedding_lengths = batch['embedding_lengths'].to(self.device)

                # Forward pass
                log_probs, loss = self.model(
                    input_values=embeddings.mean(dim=2),  # Dummy audio input
                    targets=phonemes,
                    input_lengths=embedding_lengths,
                    target_lengths=phoneme_lengths
                )

                total_loss += loss.item()

                # Calculate accuracy (simplified for single phoneme)
                predictions = self.model.predict(embeddings.mean(dim=2))  # type:ignore
                for i, pred in enumerate(predictions):
                    if len(pred) > 0 and pred[0] == phonemes[i, 0].item():
                        correct_predictions += 1
                    total_predictions += 1

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return avg_loss, accuracy


def train_ctc_model(
    embeddings_dir: str,
    output_dir: str,
    phoneme_labels_path: str,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Dict[str, str]:
    """
    Train CTC model and save artifacts.

    Args:
        embeddings_dir: Directory containing embedding files
        output_dir: Output directory for model artifacts
        phoneme_labels_path: Path to phoneme labels JSON
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Dictionary with paths to saved artifacts
    """
    print("🚀 Starting CTC model training...")
    print(f"   Embeddings: {embeddings_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Device: {device}")

    # Load phoneme labels
    with open(phoneme_labels_path, 'r') as f:
        phoneme_labels = json.load(f)

    print(f"📝 Loaded {len(phoneme_labels)} phoneme classes: {phoneme_labels[:10]}...")

    # Create dataset
    dataset = PhonemeDataset(embeddings_dir, phoneme_labels)

    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"📊 Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create model
    model = create_ctc_model(num_classes=len(phoneme_labels))
    print(f"🧠 Created CTC model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    trainer = CTCTrainer(model, device)

    # Training loop
    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_loss, val_accuracy = trainer.evaluate(val_loader)

        # Update learning rate
        trainer.scheduler.step(val_loss)

        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, "ctc_model_best.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'phoneme_labels': phoneme_labels
            }, best_model_path)
            print(f"   💾 Saved best model to {best_model_path}")

    # Save final model
    final_model_path = os.path.join(output_dir, "ctc_model_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'epoch': num_epochs,
        'phoneme_labels': phoneme_labels
    }, final_model_path)

    # Save label encoder for compatibility
    label_encoder_path = os.path.join(output_dir, "ctc_label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(phoneme_labels, f)

    # Ensure best_model_path is always a string for the return type
    if best_model_path is None:
        best_model_path = final_model_path

    print("\n✅ Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Final model: {final_model_path}")
    print(f"   Best model: {best_model_path}")

    return {
        'final_model': final_model_path,
        'best_model': best_model_path,
        'label_encoder': label_encoder_path
    }


def ctc_classifier_training(
    input_dir_str: str,
    output_dir: str = "dist",
    num_epochs: int = 20,
    batch_size: int = 32
):
    """
    Main function for CTC classifier training.

    Args:
        input_dir_str: Input directory containing embeddings
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        batch_size: Batch size
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Paths
    input_dir = Path(input_dir_str)
    phoneme_labels_path = Path(output_dir) / "phoneme_labels.json"

    # Check if phoneme labels exist
    if not phoneme_labels_path.exists():
        print(f"❌ Phoneme labels not found: {phoneme_labels_path}")
        print("   Please run embedding extraction (s2) first.")
        return

    # Check if embeddings directory exists
    if not input_dir.exists():
        print(f"❌ Embeddings directory not found: {input_dir}")
        print("   Please run embedding extraction (s2) first.")
        return

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")

    try:
        # Train model
        artifacts = train_ctc_model(
            embeddings_dir=str(input_dir),
            output_dir=output_dir,
            phoneme_labels_path=str(phoneme_labels_path),
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )

        print("\n🎉 CTC training completed successfully!")
        for name, path in artifacts.items():
            print(f"   {name}: {path}")

    except Exception as e:
        print(f"❌ CTC training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTC classifier for phoneme recognition")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dist/phoneme_embeddings",
        help="Input directory containing embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dist",
        help="Output directory for models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )

    args = parser.parse_args()

    ctc_classifier_training(
        input_dir_str=args.input_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
