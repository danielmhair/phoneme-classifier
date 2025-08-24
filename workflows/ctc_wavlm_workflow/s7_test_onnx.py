#!/usr/bin/env python3
"""
CTC ONNX Model Testing

Tests the exported CTC ONNX model functionality.
Adapted from MLP ONNX testing for CTC workflow.
"""

import numpy as np
import pandas as pd
import json
import onnxruntime as ort
from pathlib import Path


def test_ctc_onnx_model(
    onnx_path: str = "./dist/phoneme_ctc.onnx",
    embeddings_dir: str = "./dist/phoneme_embeddings_temporal",
    labels_path: str = "./dist/phoneme_labels.json",
    metadata_path: str = "./dist/phoneme_ctc_metadata.json",
    test_samples: int = 100
):
    """
    Test CTC ONNX model functionality.
    
    Args:
        onnx_path: Path to ONNX model file
        embeddings_dir: Directory containing temporal embeddings
        labels_path: Path to phoneme labels JSON
        metadata_path: Path to model metadata JSON
        test_samples: Number of samples to test
    """
    print("🧪 Testing CTC ONNX model...")
    
    try:
        # Load ONNX model
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        session = ort.InferenceSession(onnx_path)
        print(f"✅ ONNX model loaded: {onnx_path}")
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        print(f"📊 Input shape: {input_info.shape} (name: {input_info.name})")
        print(f"📊 Output shape: {output_info.shape} (name: {output_info.name})")
        
        # Load metadata
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            print(f"📝 Model metadata loaded: {model_metadata['model_type']}")
        else:
            print("⚠️ No metadata file found")
            model_metadata = {}
        
        # Load phoneme labels
        with open(labels_path, 'r') as f:
            phoneme_labels = json.load(f)
        print(f"📝 Loaded {len(phoneme_labels)} phoneme classes")
        
        # Load test embeddings
        embeddings_path = Path(embeddings_dir)
        csv_path = embeddings_path / "metadata.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Test metadata not found: {csv_path}")
        
        metadata_df = pd.read_csv(csv_path)
        print(f"📊 Available test samples: {len(metadata_df)}")
        
        # Sample test data
        test_data = metadata_df.sample(n=min(test_samples, len(metadata_df)), random_state=42)
        print(f"🎯 Testing on {len(test_data)} samples")
        
        # Test predictions
        successful_tests = 0
        total_tests = 0
        predictions = []
        true_labels = []
        
        print("🔄 Running ONNX predictions...")
        
        for i, (_, row) in enumerate(test_data.iterrows()):
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
            
            emb_path = embeddings_path / row["embedding_filename"]
            if not emb_path.exists():
                continue
            
            try:
                # Load temporal embedding
                embedding = np.load(emb_path)
                if len(embedding.shape) != 2:
                    continue
                
                # Prepare input (add batch dimension)
                input_data = embedding.astype(np.float32)[np.newaxis, :]  # Shape: (1, seq_len, 768)
                
                # Run ONNX inference
                result = session.run(None, {input_info.name: input_data})
                log_probs = result[0]  # Shape: (1, seq_len, num_classes)
                
                # Greedy decoding (take argmax at each timestep)
                predictions_seq = np.argmax(log_probs[0], axis=1)  # Shape: (seq_len,)
                
                # Simple decoding: take first non-blank prediction
                predicted_class = None
                blank_token_id = model_metadata.get('blank_token_id', len(phoneme_labels))
                
                for pred in predictions_seq:
                    if pred != blank_token_id and pred < len(phoneme_labels):
                        predicted_class = pred
                        break
                
                if predicted_class is not None:
                    # Get true label
                    true_phoneme = row["phoneme"]
                    if true_phoneme in phoneme_labels:
                        true_class = phoneme_labels.index(true_phoneme)
                        
                        predictions.append(predicted_class)
                        true_labels.append(true_class)
                        
                        successful_tests += 1
                
                total_tests += 1
                
            except Exception as e:
                print(f"⚠️ Error testing {emb_path}: {e}")
                continue
        
        if successful_tests == 0:
            print("❌ No successful ONNX predictions")
            return None
        
        # Calculate accuracy
        correct_predictions = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct_predictions / successful_tests if successful_tests > 0 else 0
        
        print(f"\\n📊 ONNX Model Test Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful Predictions: {successful_tests}")
        print(f"   Correct Predictions: {correct_predictions}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Test edge cases
        print("\\n🧪 Testing edge cases...")
        
        # Test minimum sequence length
        try:
            min_seq = np.random.randn(1, 10, 768).astype(np.float32)  # Very short sequence
            result = session.run(None, {input_info.name: min_seq})
            print(f"   ✅ Short sequence test passed: {result[0].shape}")
        except Exception as e:
            print(f"   ❌ Short sequence test failed: {e}")
        
        # Test longer sequence
        try:
            long_seq = np.random.randn(1, 200, 768).astype(np.float32)  # Longer sequence
            result = session.run(None, {input_info.name: long_seq})
            print(f"   ✅ Long sequence test passed: {result[0].shape}")
        except Exception as e:
            print(f"   ❌ Long sequence test failed: {e}")
        
        # Model size info
        model_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"\\n📦 Model Size: {model_size_mb:.2f} MB")
        
        # Save test results
        test_results = {
            'onnx_path': onnx_path,
            'total_tests': total_tests,
            'successful_predictions': successful_tests,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'model_size_mb': model_size_mb,
            'input_shape': input_info.shape,
            'output_shape': output_info.shape
        }
        
        results_path = Path("./dist/ctc_onnx_test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"💾 Test results saved to: {results_path}")
        
        print("\\n🎉 CTC ONNX testing completed!")
        
        return test_results
        
    except Exception as e:
        print(f"❌ CTC ONNX testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_ctc_onnx_model()
    if result:
        print(f"✅ ONNX testing completed with {result['accuracy']:.2%} accuracy")
        print(f"📦 Model size: {result['model_size_mb']:.2f} MB")
    else:
        print("❌ ONNX testing failed")