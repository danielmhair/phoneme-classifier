#!/usr/bin/env python3
"""
Create temporary metadata.csv for CTC analysis from existing temporal embeddings
"""

import pandas as pd
from pathlib import Path
import re

def create_metadata_from_embeddings():
    """Create metadata.csv from existing temporal embedding filenames"""
    
    embeddings_dir = Path("./dist/phoneme_embeddings_temporal")
    
    if not embeddings_dir.exists():
        print(f"❌ Embeddings directory not found: {embeddings_dir}")
        return
    
    print(f"🔍 Scanning embeddings in {embeddings_dir}...")
    
    # Get all .npy files
    embedding_files = list(embeddings_dir.glob("*.npy"))
    print(f"📁 Found {len(embedding_files)} embedding files")
    
    if len(embedding_files) == 0:
        print("❌ No embedding files found")
        return
    
    # Extract metadata from filenames
    metadata_rows = []
    
    for emb_file in embedding_files:
        filename = emb_file.name
        
        # Parse filename: speaker_phoneme_number.npy
        # e.g., callie_a_æ_001.npy
        match = re.match(r'([^_]+)_(.+)_(\d+)\.npy', filename)
        
        if match:
            speaker, phoneme, number = match.groups()
            
            metadata_rows.append({
                'embedding_filename': filename,
                'phoneme': phoneme,
                'speaker': speaker,
                'file_number': int(number),
                'source_file': f"{speaker}_{phoneme}_{number}.wav"
            })
        else:
            print(f"⚠️ Could not parse filename: {filename}")
    
    print(f"📊 Parsed {len(metadata_rows)} valid embedding files")
    
    # Create DataFrame
    metadata_df = pd.DataFrame(metadata_rows)
    
    # Save metadata
    metadata_path = embeddings_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"💾 Created metadata file: {metadata_path}")
    print(f"📈 Metadata shape: {metadata_df.shape}")
    
    # Show phoneme distribution
    phoneme_counts = metadata_df['phoneme'].value_counts()
    print(f"\n📊 Phoneme distribution (top 10):")
    for phoneme, count in phoneme_counts.head(10).items():
        print(f"   {phoneme}: {count}")
    
    print(f"\n✅ Metadata created successfully!")
    return metadata_path

if __name__ == "__main__":
    create_metadata_from_embeddings()