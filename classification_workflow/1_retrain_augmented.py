from w12_augment_audio import main as augment_audio
from w13_extract_augmented_embeddings import extract_augmented_embeddings
from w14_train_only import train_only
from w7_trace_mlp_classifier import trace_mlp_classifier
from w8_onnx_export import onnx_export
from w9_onnx_test import onnx_test

# TODO: w12->w14 haven't been tested yet, but here for good measure if we need it later
def main():
    print("\n🏁 Step 1: Augment audio files 🏁")
    augment_audio()

    print("\n🏁 Step 2: Extract embeddings for augmented data 🏁")
    extract_augmented_embeddings()

    print("\n🏁 Step 3: Retrain classifier with original + augmented data 🏁")
    train_only()

    print("\n🏁 Step 4: Trace MLP classifier for ONNX export 🏁")
    trace_mlp_classifier()

    print("\n🏁 Step 5: Export ONNX models 🏁")
    onnx_export()

    print("\n🏁 Step 6: Test ONNX models 🏁")
    onnx_test()

    print("\n✅ All augmentation-based retraining steps completed successfully! ✅")


if __name__ == "__main__":
    main()
