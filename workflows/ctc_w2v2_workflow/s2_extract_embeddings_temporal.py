"""
Temporal Embeddings Extraction for CTC Training

This script extracts Wav2Vec2 embeddings while preserving the temporal dimension
for CTC (Connectionist Temporal Classification) training. Unlike the original
s2_extract_embeddings_for_phonemes.py which uses mean pooling, this version
keeps the full temporal sequence for alignment-free sequence modeling.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os

# Try to import dependencies with fallback
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import torch
    import soundfile as sf
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"⚠️ Dependencies not available: {e}")
    print("To enable temporal embedding extraction, install:")
    print("pip install torch transformers soundfile")

# Augmentation (if available)
try:
    from audiomentations import Compose, AddGaussianNoise, HighPassFilter, LowPassFilter, Normalize
    
    # Define augmentation pipeline for simulating bad mic conditions
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
        HighPassFilter(min_cutoff_freq=100.0, max_cutoff_freq=300.0, p=0.4),
        LowPassFilter(min_cutoff_freq=3000.0, max_cutoff_freq=8000.0, p=0.4),
        Normalize(p=1.0),
    ])
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    augment = None


def extract_temporal_embeddings(
    input_dir: str = "dist/organized_recordings",
    output_dir: str = "dist/phoneme_embeddings_temporal",
    phoneme_label_json_path: str = "dist/phoneme_labels.json",
    preserve_temporal: bool = True,
    max_length: int = 1000
):
    """
    Extract embeddings with temporal information preserved for CTC training.
    
    Args:
        input_dir: Directory containing organized recordings
        output_dir: Output directory for temporal embeddings
        phoneme_label_json_path: Path to save phoneme labels
        preserve_temporal: Whether to preserve temporal dimension (True for CTC)
        max_length: Maximum temporal length to avoid memory issues
    """
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Required dependencies not available. Skipping temporal embedding extraction.")
        return
    
    print(f"🚀 Starting temporal embedding extraction...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Preserve temporal: {preserve_temporal}")
    print(f"   Max temporal length: {max_length}")
    
    # Setup paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Load metadata
    metadata_csv = input_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    
    meta = pd.read_csv(metadata_csv)
    print(f"📊 Loaded metadata with {len(meta)} rows")
    
    # Load Wav2Vec2 processor and model
    print("🧠 Loading Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
    
    # Get phoneme directories and create label mapping
    phoneme_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    phoneme_labels = sorted([d.name for d in phoneme_dirs])
    
    # Save phoneme labels
    with open(phoneme_label_json_path, "w") as f:
        json.dump(phoneme_labels, f, indent=2)
    print(f"✅ Saved {len(phoneme_labels)} phoneme labels to {phoneme_label_json_path}")
    
    # Process embeddings
    embeddings = []
    labels = []
    processed_count = 0
    skipped_count = 0
    
    print(f"🔄 Processing {len(meta)} audio files...")
    
    for i, row in meta.iterrows():
        if processed_count % 100 == 0:
            print(f"   Progress: {processed_count}/{len(meta)} processed, {skipped_count} skipped")
        
        # Construct file path
        wav_path = input_dir / row["phoneme"] / row["new_filename"]
        
        if not wav_path.exists():
            print(f"   [SKIP] File not found: {wav_path}")
            skipped_count += 1
            continue
        
        try:
            # Load audio
            audio, sr = sf.read(str(wav_path))
            
            if sr != 16000:
                print(f"   [SKIP] Wrong sample rate {sr} for {wav_path}")
                skipped_count += 1
                continue
            
            # Apply augmentation if available
            if AUGMENTATION_AVAILABLE and augment is not None:
                try:
                    audio = augment(samples=audio, sample_rate=sr)
                except Exception as e:
                    print(f"   [WARN] Augmentation failed for {wav_path}: {e}")
            
            # Preprocess for Wav2Vec2
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get temporal embeddings
            if preserve_temporal:
                # Keep full temporal sequence (T, D) where T=time, D=features
                emb = outputs.last_hidden_state.squeeze(0).numpy()  # Remove batch dim
                
                # Limit temporal length for memory management
                if emb.shape[0] > max_length:
                    emb = emb[:max_length]
                    
            else:
                # Use mean pooling for backward compatibility
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Save embedding
            emb_filename = str(row["new_filename"])
            if emb_filename.endswith(".wav"):
                emb_filename = emb_filename[:-4] + ".npy"
            
            embedding_path = output_dir / emb_filename
            np.save(embedding_path, emb)
            
            # Store for batch processing
            embeddings.append(emb)
            labels.append(row["phoneme"])
            processed_count += 1
            
        except Exception as e:
            print(f"   [ERROR] Failed to process {wav_path}: {e}")
            skipped_count += 1
            continue
    
    print(f"✅ Temporal embedding extraction completed!")
    print(f"   Processed: {processed_count} files")
    print(f"   Skipped: {skipped_count} files")
    print(f"   Output directory: {output_dir}")
    
    # Save batch embeddings if any were processed
    if embeddings:
        if preserve_temporal:
            # For temporal embeddings, we can't easily stack due to variable lengths
            # Save metadata about embedding shapes instead
            embedding_info = []
            for i, (emb, label) in enumerate(zip(embeddings, labels)):
                embedding_info.append({
                    'filename': meta.iloc[i]['new_filename'].replace('.wav', '.npy'),
                    'phoneme': label,
                    'temporal_shape': emb.shape,
                    'temporal_length': emb.shape[0] if len(emb.shape) > 1 else 1,
                    'feature_dim': emb.shape[1] if len(emb.shape) > 1 else emb.shape[0]
                })
            
            # Save embedding info
            embedding_info_df = pd.DataFrame(embedding_info)
            embedding_info_path = output_dir / "temporal_embedding_info.csv"
            embedding_info_df.to_csv(embedding_info_path, index=False)
            print(f"📊 Saved temporal embedding info to {embedding_info_path}")
            
            # Print statistics
            temporal_lengths = [info['temporal_length'] for info in embedding_info]
            print(f"📈 Temporal length statistics:")
            print(f"   Min: {min(temporal_lengths)}, Max: {max(temporal_lengths)}")
            print(f"   Mean: {np.mean(temporal_lengths):.1f}, Std: {np.std(temporal_lengths):.1f}")
            
        else:
            # For non-temporal embeddings, save as before
            embeddings_array = np.array(embeddings)
            np.save(output_dir / "embeddings.npy", embeddings_array)
            print(f"📦 Saved batch embeddings to {output_dir / 'embeddings.npy'}")
    
    # Update metadata with embedding filenames
    meta_out = meta.copy()
    meta_out["embedding_filename"] = meta_out["new_filename"].astype(str).str.replace(".wav", ".npy")
    meta_out.to_csv(output_dir / "metadata.csv", index=False)
    print(f"📝 Saved updated metadata to {output_dir / 'metadata.csv'}")


def extract_embeddings_for_phonemes_temporal(
    input_dir: str = "dist/organized_recordings",
    output_dir: str = "dist/phoneme_embeddings",
    phoneme_label_json_path: str = "dist/phoneme_labels.json",
    enable_ctc: bool = False
):
    """
    Wrapper function for backward compatibility with existing pipeline.
    
    Args:
        input_dir: Input directory with organized recordings
        output_dir: Output directory for embeddings
        phoneme_label_json_path: Path for phoneme labels JSON
        enable_ctc: Whether to enable CTC mode (preserve temporal sequences)
    """
    if enable_ctc:
        print("🔄 CTC mode enabled - preserving temporal sequences")
        # Use temporal output directory for CTC
        ctc_output_dir = output_dir.replace("phoneme_embeddings", "phoneme_embeddings_temporal")
        extract_temporal_embeddings(
            input_dir=input_dir,
            output_dir=ctc_output_dir,
            phoneme_label_json_path=phoneme_label_json_path,
            preserve_temporal=True
        )
    else:
        print("🔄 Standard mode - using mean pooling for MLP compatibility")
        extract_temporal_embeddings(
            input_dir=input_dir,
            output_dir=output_dir,
            phoneme_label_json_path=phoneme_label_json_path,
            preserve_temporal=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract temporal embeddings for CTC training")
    parser.add_argument("--input_dir", type=str, default="dist/organized_recordings",
                       help="Input directory with organized recordings")
    parser.add_argument("--output_dir", type=str, default="dist/phoneme_embeddings_temporal",
                       help="Output directory for temporal embeddings")
    parser.add_argument("--phoneme_labels", type=str, default="dist/phoneme_labels.json",
                       help="Path to save phoneme labels JSON")
    parser.add_argument("--max_length", type=int, default=1000,
                       help="Maximum temporal length to avoid memory issues")
    parser.add_argument("--no_temporal", action="store_true",
                       help="Disable temporal preservation (use mean pooling)")
    
    args = parser.parse_args()
    
    preserve_temporal = not args.no_temporal
    
    extract_temporal_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        phoneme_label_json_path=args.phoneme_labels,
        preserve_temporal=preserve_temporal,
        max_length=args.max_length
    )