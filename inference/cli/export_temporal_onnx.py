#!/usr/bin/env python3
"""
Export Temporal Feature Extraction Models to ONNX

Creates ONNX models for WavLM and Wav2Vec2 that preserve temporal sequences
for proper CTC inference in Epic 2, replacing the averaged embeddings approach.
"""
import torch
import numpy as np
import json
import sys
import os
from pathlib import Path
from transformers import Wav2Vec2Processor, WavLMModel, Wav2Vec2Model

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TemporalFeatureExtractor(torch.nn.Module):
    """Wrapper for temporal feature extraction without averaging."""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        
        if "wavlm" in model_name.lower():
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        else:
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze the model for feature extraction
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def forward(self, input_values):
        """Extract temporal features preserving sequence dimension.
        
        Args:
            input_values: Raw audio tensor [batch, audio_length]
            
        Returns:
            temporal_features: [batch, sequence_length, 768]
        """
        with torch.no_grad():
            outputs = self.model(input_values)
            # Return temporal features (not averaged)
            return outputs.last_hidden_state  # [batch, seq_len, 768]


def export_temporal_onnx_model(
    model_name: str, 
    output_path: str,
    sample_rate: int = 16000,
    chunk_duration_ms: int = 64
):
    """Export temporal feature extractor to ONNX.
    
    Args:
        model_name: 'wavlm' or 'wav2vec2'
        output_path: Output path for ONNX model
        sample_rate: Audio sample rate
        chunk_duration_ms: Chunk duration in milliseconds (Epic 2 uses 64ms)
    """
    print(f"🚀 Exporting {model_name} temporal feature extractor to ONNX...")
    
    # Load processor for audio preprocessing
    if "wavlm" in model_name.lower():
        # WavLM uses Wav2Vec2 processor (they share the same preprocessing)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model_display_name = "WavLM"
    else:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model_display_name = "Wav2Vec2"
    
    # Create temporal feature extractor
    feature_extractor = TemporalFeatureExtractor(model_name)
    
    # Calculate chunk size in samples
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    print(f"📏 Chunk size: {chunk_duration_ms}ms = {chunk_samples} samples")
    
    # Create example input for ONNX export (64ms chunk)
    example_audio = torch.randn(1, chunk_samples)  # [batch=1, audio_length]
    
    print(f"📝 Example input shape: {example_audio.shape}")
    
    # Test the model first
    with torch.no_grad():
        test_output = feature_extractor(example_audio)
        print(f"✅ Test output shape: {test_output.shape}")
        print(f"📊 Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        feature_extractor,
        example_audio,
        output_path,
        export_params=True,
        opset_version=14,  # Use newer opset for transformer support
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['temporal_features'],
        dynamic_axes={
            'audio': {0: 'batch_size', 1: 'audio_length'},
            'temporal_features': {0: 'batch_size', 1: 'sequence_length'}
        },
        verbose=False
    )
    
    print(f"✅ {model_display_name} temporal ONNX exported to: {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        import onnxruntime as ort
        
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        
        # Test ONNX runtime
        session = ort.InferenceSession(str(output_path))
        test_input = np.random.randn(1, chunk_samples).astype(np.float32)
        ort_outputs = session.run(None, {'audio': test_input})
        
        print(f"🧪 ONNX runtime test: {ort_outputs[0].shape}")
        
    except ImportError:
        print("⚠️ ONNX verification skipped (onnx/onnxruntime not available)")
    except Exception as e:
        print(f"⚠️ ONNX verification failed: {e}")
    
    # Create metadata
    metadata = {
        "model_type": f"{model_name}_temporal_features",
        "model_name": model_display_name,
        "input_shape": [1, "audio_length"],
        "output_shape": [1, "sequence_length", 768],
        "sample_rate": sample_rate,
        "chunk_duration_ms": chunk_duration_ms,
        "chunk_samples": chunk_samples,
        "feature_dim": 768,
        "export_date": str(torch.utils.data.Dataset.__dict__.get('__doc__', 'unknown')),  # Quick timestamp
        "purpose": "Epic 2 temporal feature extraction for CTC models"
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Metadata saved to: {metadata_path}")
    
    # Get model size
    model_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"📦 Model size: {model_size_mb:.2f} MB")
    
    return {
        "onnx_path": str(output_path),
        "metadata_path": str(metadata_path),
        "model_size_mb": model_size_mb,
        "input_shape": [1, chunk_samples],
        "output_shape": list(test_output.shape),
        "feature_dim": 768
    }


def main():
    """Export both WavLM and Wav2Vec2 temporal feature extractors."""
    print("🏭 Epic 2: Temporal Feature Extractor ONNX Export")
    print("=" * 60)
    print("Creating ONNX models for proper CTC temporal inference")
    print()
    
    dist_dir = Path("/home/danie/Workspaces/fast-api-phoneme-python/dist")
    
    models_to_export = [
        ("wavlm", dist_dir / "wavlm_temporal.onnx"),
        ("wav2vec2", dist_dir / "wav2vec2_temporal.onnx")
    ]
    
    results = {}
    
    for model_name, output_path in models_to_export:
        try:
            print(f"📦 Exporting {model_name.upper()} temporal features...")
            result = export_temporal_onnx_model(model_name, str(output_path))
            results[model_name] = result
            print("✅ Export successful!")
            print()
            
        except Exception as e:
            print(f"❌ Failed to export {model_name}: {e}")
            print()
            continue
    
    # Summary
    print("📊 Export Summary")
    print("=" * 30)
    for model_name, result in results.items():
        print(f"✅ {model_name.upper()}: {result['model_size_mb']:.1f}MB")
        print(f"   Input: {result['input_shape']}")
        print(f"   Output: {result['output_shape']}")
        print()
    
    print(f"🎯 Next steps:")
    print("1. Update AudioFeatureExtractor to use temporal models for CTC")
    print("2. Fix CTC probability distribution extraction") 
    print("3. Test improved accuracy with temporal brain")


if __name__ == "__main__":
    main()