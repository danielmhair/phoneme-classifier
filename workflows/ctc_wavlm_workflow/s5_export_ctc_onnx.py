#!/usr/bin/env python3
"""
CTC Model ONNX Export

Exports trained CTC model to ONNX format for deployment.
Adapted from s10_onnx_export.py for CTC workflow.
"""

import torch
import numpy as np
import json
from pathlib import Path
from workflows.ctc_w2v2_workflow.models.ctc_model import create_ctc_model

def export_ctc_to_onnx(
    model_path: str = "./dist/ctc_model_best.pt",
    labels_path: str = "./dist/phoneme_labels.json",
    output_dir: str = "./dist",
    model_name: str = "phoneme_ctc"
):
    """
    Export CTC model to ONNX format.
    
    Args:
        model_path: Path to trained CTC model
        labels_path: Path to phoneme labels JSON
        output_dir: Directory to save ONNX files
        model_name: Base name for exported files
    """
    print("📦 Exporting CTC model to ONNX...")
    
    try:
        # Check if ONNX is available
        try:
            import onnx
            import onnxruntime
            print("✅ ONNX dependencies available")
        except ImportError:
            print("❌ ONNX not available. Install with: pip install onnx onnxruntime")
            return
        
        # Load phoneme labels
        with open(labels_path, 'r') as f:
            phoneme_labels = json.load(f)
        
        print(f"📝 Loaded {len(phoneme_labels)} phoneme labels")
        
        # Load trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        # Create example input (batch_size=1, max_seq_len=100, embedding_dim=768)
        example_seq_len = 100
        embedding_dim = 768
        example_input = torch.randn(1, example_seq_len, embedding_dim).to(device)
        
        print(f"📏 Example input shape: {example_input.shape}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Export to ONNX
        onnx_path = Path(output_dir) / f"{model_name}.onnx"
        
        print(f"🔄 Exporting to ONNX: {onnx_path}")
        
        # Custom forward pass for ONNX (inference only, no loss computation)
        class CTCInferenceModel(torch.nn.Module):
            def __init__(self, ctc_model):
                super().__init__()
                self.ctc_model = ctc_model
            
            def forward(self, input_values):
                # Only return log probabilities for inference
                log_probs, _ = self.ctc_model(input_values)
                return log_probs
        
        inference_model = CTCInferenceModel(model)
        inference_model.eval()
        
        # Export with dynamic axes for variable sequence length
        torch.onnx.export(
            inference_model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['embeddings'],
            output_names=['log_probabilities'],
            dynamic_axes={
                'embeddings': {0: 'batch_size', 1: 'sequence_length'},
                'log_probabilities': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        print(f"✅ ONNX model exported to: {onnx_path}")
        
        # Verify the exported model
        print("🔍 Verifying ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        
        # Test ONNX runtime inference
        print("🧪 Testing ONNX runtime inference...")
        ort_session = onnxruntime.InferenceSession(str(onnx_path))
        
        # Prepare test input
        test_input = np.random.randn(1, 50, 768).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"📊 ONNX output shape: {ort_outputs[0].shape}")
        print(f"📊 ONNX output range: [{ort_outputs[0].min():.4f}, {ort_outputs[0].max():.4f}]")
        
        # Save model metadata
        metadata = {
            "model_type": "ctc_phoneme_classifier",
            "input_shape": [1, "sequence_length", 768],
            "output_shape": [1, "sequence_length", len(phoneme_labels) + 1],  # +1 for blank
            "phoneme_labels": phoneme_labels,
            "num_classes": len(phoneme_labels) + 1,
            "embedding_dim": embedding_dim,
            "blank_token_id": len(phoneme_labels),
            "model_path": str(model_path),
            "onnx_path": str(onnx_path)
        }
        
        metadata_path = Path(output_dir) / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Model metadata saved to: {metadata_path}")
        
        # Get file sizes
        onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"📦 ONNX model size: {onnx_size_mb:.2f} MB")
        
        print("🎉 CTC ONNX export completed successfully!")
        
        return {
            "onnx_path": str(onnx_path),
            "metadata_path": str(metadata_path),
            "model_size_mb": onnx_size_mb,
            "num_classes": len(phoneme_labels) + 1,
            "embedding_dim": embedding_dim
        }
        
    except Exception as e:
        print(f"❌ CTC ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = export_ctc_to_onnx()
    if result:
        print(f"✅ Export successful: {result['onnx_path']}")
    else:
        print("❌ Export failed")