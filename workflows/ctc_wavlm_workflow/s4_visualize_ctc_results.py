#!/usr/bin/env python3
"""
CTC Results Visualization

Generates UMAP plots and confusion matrices for CTC model evaluation.
Adapted from s5_visualize_results.py for CTC workflow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not available. Install with: pip install umap-learn")

def load_ctc_model_predictions(model_path: str, embeddings_dir: str, labels_path: str):
    """Load CTC model and generate predictions for visualization."""
    try:
        import torch
        from workflows.ctc_w2v2_workflow.models.ctc_model import create_ctc_model
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_ctc_model(num_classes=37)
        
        # Load checkpoint (may contain training state)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(device)
        
        # Load phoneme labels
        with open(labels_path, 'r') as f:
            phoneme_labels = json.load(f)
        
        # Load test embeddings and make predictions
        embeddings_path = Path(embeddings_dir)
        metadata_path = embeddings_path / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)
        
        embeddings = []
        true_labels = []
        predictions = []
        
        print(f"📊 Loading embeddings from {embeddings_dir}")
        
        # Sample subset for visualization (to avoid memory issues)
        sample_size = min(5000, len(metadata))
        sample_metadata = metadata.sample(n=sample_size, random_state=42)
        
        with torch.no_grad():
            for _, row in sample_metadata.iterrows():
                emb_path = embeddings_path / row["embedding_filename"]
                if emb_path.exists():
                    # Load embedding
                    embedding = np.load(emb_path)
                    if len(embedding.shape) == 2:
                        # Use mean pooling for visualization (like MLP)
                        pooled_embedding = np.mean(embedding, axis=0)
                        embeddings.append(pooled_embedding)
                        
                        # Get true label
                        phoneme = row["phoneme"]
                        if phoneme in phoneme_labels:
                            true_labels.append(phoneme_labels.index(phoneme))
                            
                            # Get prediction
                            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(device)
                            pred_sequence = model.predict(embedding_tensor)[0]
                            # Use first prediction if available, otherwise predict unknown
                            pred_label = pred_sequence[0] if len(pred_sequence) > 0 else len(phoneme_labels)
                            predictions.append(pred_label)
        
        return np.array(embeddings), np.array(true_labels), np.array(predictions), phoneme_labels
        
    except Exception as e:
        print(f"❌ Error loading CTC model predictions: {e}")
        return None, None, None, None


def visualize_ctc_results(
    model_path: str = "./dist/ctc_model_best.pt",
    embeddings_dir: str = "./dist/phoneme_embeddings_temporal",
    labels_path: str = "./dist/phoneme_labels.json",
    output_dir: str = "./dist"
):
    """
    Generate CTC visualization plots and confusion matrix.
    
    Args:
        model_path: Path to trained CTC model
        embeddings_dir: Directory containing temporal embeddings
        labels_path: Path to phoneme labels JSON
        output_dir: Directory to save plots
    """
    print("🎨 Generating CTC visualization plots...")
    
    # Load model predictions
    embeddings, true_labels, predictions, phoneme_labels = load_ctc_model_predictions(
        model_path, embeddings_dir, labels_path
    )
    
    if embeddings is None:
        print("❌ Could not load CTC model predictions for visualization")
        return
    
    print(f"📊 Visualizing {len(embeddings)} samples")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Generate UMAP plot if available
    if UMAP_AVAILABLE and len(embeddings) > 10:
        print("🗺️ Generating UMAP plot...")
        
        # Create UMAP embedding
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Create UMAP plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1], 
            c=true_labels, 
            cmap='tab20', 
            alpha=0.7,
            s=20
        )
        plt.colorbar(scatter, label='Phoneme Class')
        plt.title('CTC Model: UMAP Visualization of Phoneme Embeddings')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        
        # Save plot
        umap_path = Path(output_dir) / "ctc_umap_plot.png"
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ UMAP plot saved to: {umap_path}")
    
    # 2. Generate confusion matrix
    print("📊 Generating confusion matrix...")
    
    # Filter valid predictions (within label range)
    valid_mask = predictions < len(phoneme_labels)
    valid_true = true_labels[valid_mask]
    valid_pred = predictions[valid_mask]
    
    if len(valid_true) > 0:
        # Create confusion matrix
        cm = confusion_matrix(valid_true, valid_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            cm,
            annot=False,  # Too many labels to show numbers
            cmap='Blues',
            xticklabels=[phoneme_labels[i] if i < len(phoneme_labels) else f'unk_{i}' for i in range(cm.shape[1])],
            yticklabels=[phoneme_labels[i] if i < len(phoneme_labels) else f'unk_{i}' for i in range(cm.shape[0])]
        )
        plt.title('CTC Model: Confusion Matrix')
        plt.xlabel('Predicted Phoneme')
        plt.ylabel('True Phoneme')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save confusion matrix
        cm_path = Path(output_dir) / "ctc_confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved to: {cm_path}")
        
        # Calculate and print accuracy
        accuracy = np.sum(valid_true == valid_pred) / len(valid_true)
        print(f"📈 CTC Model Accuracy on sample: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Save classification report
        report = classification_report(
            valid_true, 
            valid_pred,
            target_names=[phoneme_labels[i] for i in range(min(len(phoneme_labels), cm.shape[0]))],
            output_dict=True
        )
        
        report_path = Path(output_dir) / "ctc_classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Classification report saved to: {report_path}")
    
    print("🎨 CTC visualization complete!")


if __name__ == "__main__":
    visualize_ctc_results()