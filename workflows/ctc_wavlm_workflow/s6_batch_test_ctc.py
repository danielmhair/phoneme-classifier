#!/usr/bin/env python3
"""
CTC Batch Testing

Comprehensive batch testing of CTC model on validation set.
Adapted from s7_batch_test_phonemes.py for CTC workflow.
"""

import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from workflows.ctc_w2v2_workflow.models.ctc_model import create_ctc_model

def batch_test_ctc_phonemes(
    model_path: str = "./dist/ctc_model_best.pt",
    embeddings_dir: str = "./dist/phoneme_embeddings_temporal", 
    labels_path: str = "./dist/phoneme_labels.json",
    output_dir: str = "./dist",
    max_samples: int = 2000
):
    """
    Perform comprehensive batch testing of CTC model.
    
    Args:
        model_path: Path to trained CTC model
        embeddings_dir: Directory containing temporal embeddings
        labels_path: Path to phoneme labels JSON
        output_dir: Directory to save test results
        max_samples: Maximum samples to test (for performance)
    """
    print("🧪 Starting CTC batch testing...")
    
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
        
        print(f"🧠 Loaded CTC model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Load test data
        embeddings_path = Path(embeddings_dir)
        metadata_path = embeddings_path / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)
        print(f"📊 Loaded metadata with {len(metadata)} samples")
        
        # Sample for testing (to avoid memory issues)
        test_samples = min(max_samples, len(metadata))
        test_metadata = metadata.sample(n=test_samples, random_state=42)
        print(f"🎯 Testing on {test_samples} samples")
        
        # Initialize results tracking
        results = {
            'predictions': [],
            'true_labels': [],
            'phoneme_names': [],
            'confidence_scores': [],
            'sequence_lengths': [],
            'filenames': []
        }
        
        per_phoneme_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []})
        
        print("🔄 Running batch predictions...")
        
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
                    log_probs, _ = model(embedding_tensor)
                    
                    # Decode CTC output (greedy decoding)
                    prediction_sequence = model.predict(embedding_tensor)[0]
                    
                    # Get true label
                    true_phoneme = row["phoneme"]
                    if true_phoneme not in phoneme_labels:
                        continue
                    
                    true_label_idx = phoneme_labels.index(true_phoneme)
                    
                    # For single phoneme classification, take first prediction or default to unknown
                    predicted_label_idx = prediction_sequence[0] if len(prediction_sequence) > 0 else len(phoneme_labels)
                    
                    # Calculate confidence (max log probability for predicted token)
                    if predicted_label_idx < log_probs.shape[2]:
                        confidence = torch.exp(log_probs[0, :, predicted_label_idx].max()).item()
                    else:
                        confidence = 0.0
                    
                    # Store results
                    results['predictions'].append(predicted_label_idx)
                    results['true_labels'].append(true_label_idx)
                    results['phoneme_names'].append(true_phoneme)
                    results['confidence_scores'].append(confidence)
                    results['sequence_lengths'].append(embedding.shape[0])
                    results['filenames'].append(row["embedding_filename"])
                    
                    # Update per-phoneme stats
                    is_correct = (predicted_label_idx == true_label_idx)
                    per_phoneme_stats[true_phoneme]['correct'] += int(is_correct)
                    per_phoneme_stats[true_phoneme]['total'] += 1
                    per_phoneme_stats[true_phoneme]['predictions'].append(predicted_label_idx)
                    
                except Exception as e:
                    print(f"⚠️ Error processing {emb_path}: {e}")
                    continue
        
        # Calculate overall metrics
        valid_predictions = [p for p in results['predictions'] if p < len(phoneme_labels)]
        valid_true_labels = [results['true_labels'][i] for i, p in enumerate(results['predictions']) if p < len(phoneme_labels)]
        
        if len(valid_predictions) == 0:
            print("❌ No valid predictions made")
            return
        
        overall_accuracy = accuracy_score(valid_true_labels, valid_predictions)
        print(f"🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # Per-phoneme analysis
        print("\n📊 Per-Phoneme Analysis:")
        phoneme_accuracies = {}
        
        for phoneme in sorted(per_phoneme_stats.keys()):
            stats = per_phoneme_stats[phoneme]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                phoneme_accuracies[phoneme] = accuracy
                print(f"   {phoneme:>6}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        # Find best and worst performing phonemes
        if phoneme_accuracies:
            best_phoneme = max(phoneme_accuracies.items(), key=lambda x: x[1])
            worst_phoneme = min(phoneme_accuracies.items(), key=lambda x: x[1])
            
            print(f"\n🏆 Best: {best_phoneme[0]} ({best_phoneme[1]:.3f})")
            print(f"🚨 Worst: {worst_phoneme[0]} ({worst_phoneme[1]:.3f})")
        
        # Confidence analysis
        avg_confidence = np.mean(results['confidence_scores'])
        print(f"📈 Average Confidence: {avg_confidence:.4f}")
        
        # Sequence length analysis
        avg_seq_len = np.mean(results['sequence_lengths'])
        print(f"📏 Average Sequence Length: {avg_seq_len:.1f} frames")
        
        # Save detailed results
        results_df = pd.DataFrame({
            'filename': results['filenames'],
            'true_phoneme': results['phoneme_names'],
            'true_label_idx': results['true_labels'],
            'predicted_label_idx': results['predictions'],
            'predicted_phoneme': [phoneme_labels[p] if p < len(phoneme_labels) else 'UNK' for p in results['predictions']],
            'confidence': results['confidence_scores'],
            'sequence_length': results['sequence_lengths'],
            'correct': [t == p for t, p in zip(results['true_labels'], results['predictions'])]
        })
        
        results_path = Path(output_dir) / "ctc_batch_test_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"💾 Detailed results saved to: {results_path}")
        
        # Save per-phoneme summary
        phoneme_summary = []
        for phoneme in sorted(phoneme_accuracies.keys()):
            stats = per_phoneme_stats[phoneme]
            phoneme_summary.append({
                'phoneme': phoneme,
                'accuracy': phoneme_accuracies[phoneme],
                'correct': stats['correct'],
                'total': stats['total'],
                'error_rate': 1 - phoneme_accuracies[phoneme]
            })
        
        summary_df = pd.DataFrame(phoneme_summary)
        summary_path = Path(output_dir) / "ctc_phoneme_accuracy_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Per-phoneme summary saved to: {summary_path}")
        
        # Generate classification report
        if len(valid_predictions) > 0:
            class_report = classification_report(
                valid_true_labels,
                valid_predictions,
                target_names=phoneme_labels,
                output_dict=True
            )
            
            report_path = Path(output_dir) / "ctc_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(class_report, f, indent=2)
            print(f"📋 Classification report saved to: {report_path}")
        
        print("\n🎉 CTC batch testing completed!")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_tested': len(results['true_labels']),
            'phoneme_accuracies': phoneme_accuracies,
            'avg_confidence': avg_confidence,
            'avg_sequence_length': avg_seq_len
        }
        
    except Exception as e:
        print(f"❌ CTC batch testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = batch_test_ctc_phonemes()
    if result:
        print(f"✅ Testing completed with {result['overall_accuracy']:.2%} accuracy")
    else:
        print("❌ Testing failed")