#!/usr/bin/env python3
"""
CTC Confusion Analysis

Analyzes most confused phoneme pairs for CTC model.
Adapted from s6_confusion_pairs.py for CTC workflow.
"""

import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from workflows.ctc_w2v2_workflow.models.ctc_model import create_ctc_model


def analyze_ctc_confusion(
    model_path: str = "./dist/ctc_model_best.pt",
    embeddings_dir: str = "./dist/phoneme_embeddings_temporal",
    labels_path: str = "./dist/phoneme_labels.json",
    output_dir: str = "./dist",
    max_samples: int = 1000
):
    """
    Analyze most confused phoneme pairs for CTC model.
    
    Args:
        model_path: Path to trained CTC model
        embeddings_dir: Directory containing temporal embeddings
        labels_path: Path to phoneme labels JSON
        output_dir: Directory to save confusion analysis
        max_samples: Maximum samples for analysis (for performance)
    """
    print("🧩 Starting CTC confusion analysis...")
    
    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {device}")
        
        # Load phoneme labels
        with open(labels_path, 'r') as f:
            phoneme_labels = json.load(f)
        print(f"📝 Loaded {len(phoneme_labels)} phoneme classes")
        
        # Load CTC model
        model = create_ctc_model(num_classes=len(phoneme_labels))
        
        # Load checkpoint (may contain training state)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(device)
        
        # Load test data
        embeddings_path = Path(embeddings_dir)
        metadata_path = embeddings_path / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)
        print(f"📊 Loaded metadata with {len(metadata)} samples")
        
        # Sample for analysis
        test_samples = min(max_samples, len(metadata))
        test_metadata = metadata.sample(n=test_samples, random_state=42)
        print(f"🎯 Analyzing {test_samples} samples")
        
        # Collect predictions
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for i, (_, row) in enumerate(test_metadata.iterrows()):
                if i % 100 == 0:
                    print(f"   Progress: {i}/{test_samples} ({i/test_samples*100:.1f}%)")
                
                emb_path = embeddings_path / row["embedding_filename"]
                if not emb_path.exists():
                    continue
                
                try:
                    # Load temporal embedding
                    embedding = np.load(emb_path)
                    if len(embedding.shape) != 2:
                        continue
                    
                    # Convert to tensor
                    embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(device)
                    
                    # Get model prediction
                    prediction_sequence = model.predict(embedding_tensor)[0]
                    
                    # Get true label
                    true_phoneme = row["phoneme"]
                    if true_phoneme not in phoneme_labels:
                        continue
                    
                    true_label_idx = phoneme_labels.index(true_phoneme)
                    
                    # For single phoneme classification, take first prediction
                    predicted_label_idx = prediction_sequence[0] if len(prediction_sequence) > 0 else len(phoneme_labels)
                    
                    # Only include valid predictions
                    if predicted_label_idx < len(phoneme_labels):
                        predictions.append(predicted_label_idx)
                        true_labels.append(true_label_idx)
                    
                except Exception as e:
                    continue
        
        if len(predictions) == 0:
            print("❌ No valid predictions for confusion analysis")
            return
        
        print(f"📊 Analyzing {len(predictions)} valid predictions")
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Find most confused pairs (excluding diagonal)
        confusion_pairs = []
        for i in range(len(phoneme_labels)):
            for j in range(len(phoneme_labels)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_phoneme': phoneme_labels[i],
                        'predicted_phoneme': phoneme_labels[j],
                        'confusion_count': cm[i, j],
                        'true_total': cm[i, :].sum(),
                        'confusion_rate': cm[i, j] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                    })
        
        # Sort by confusion count
        confusion_pairs.sort(key=lambda x: x['confusion_count'], reverse=True)
        
        print(f"\\n🔍 Top 10 Most Confused Phoneme Pairs:")
        print(f"{'True':<8} {'Predicted':<10} {'Count':<6} {'Rate':<6}")
        print("-" * 35)
        
        for pair in confusion_pairs[:10]:
            print(f"{pair['true_phoneme']:<8} {pair['predicted_phoneme']:<10} "
                  f"{pair['confusion_count']:<6} {pair['confusion_rate']:.3f}")
        
        # Save confusion analysis
        confusion_df = pd.DataFrame(confusion_pairs)
        confusion_path = Path(output_dir) / "ctc_confusion_analysis.csv"
        confusion_df.to_csv(confusion_path, index=False)
        print(f"\\n💾 Confusion analysis saved to: {confusion_path}")
        
        # Generate summary statistics
        total_errors = sum(pair['confusion_count'] for pair in confusion_pairs)
        total_predictions = len(predictions)
        error_rate = total_errors / total_predictions if total_predictions > 0 else 0
        
        print(f"\\n📈 Confusion Analysis Summary:")
        print(f"   Total Predictions: {total_predictions}")
        print(f"   Total Errors: {total_errors}")
        print(f"   Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
        print(f"   Accuracy: {1-error_rate:.4f} ({(1-error_rate)*100:.2f}%)")
        print(f"   Unique Confusion Pairs: {len(confusion_pairs)}")
        
        # Save confusion matrix
        cm_data = {
            'confusion_matrix': cm.tolist(),
            'phoneme_labels': phoneme_labels,
            'total_predictions': int(total_predictions),
            'total_errors': int(total_errors),
            'error_rate': float(error_rate),
            'accuracy': float(1 - error_rate)
        }
        
        cm_path = Path(output_dir) / "ctc_confusion_matrix_data.json"
        with open(cm_path, 'w') as f:
            json.dump(cm_data, f, indent=2)
        print(f"📊 Confusion matrix data saved to: {cm_path}")
        
        print("\\n🎉 CTC confusion analysis completed!")
        
        return {
            'confusion_pairs': confusion_pairs[:10],
            'total_predictions': total_predictions,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'accuracy': 1 - error_rate
        }
        
    except Exception as e:
        print(f"❌ CTC confusion analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = analyze_ctc_confusion()
    if result:
        print(f"✅ Analysis completed with {result['accuracy']:.2%} accuracy")
        print(f"🔍 Found {len(result['confusion_pairs'])} top confusion pairs")
    else:
        print("❌ Analysis failed")