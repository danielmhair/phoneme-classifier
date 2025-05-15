from w1_prepare_wav_files import prepare_wav_files
from w2_extract_embeddings_for_phonemes import extract_embeddings_for_phonemes
from w3_embeddings import embeddings
from w4_batch_test_phonemes import batch_test_phonemes
from w6_benchmark_inference_and_save import benchmark_and_save
from w7_trace_mlp_classifier import trace_mlp_classifier
from w8_onnx_export import onnx_export
from w9_onnx_test import onnx_test
from w10_visualize_results import visualize_results
from w11_confusion_pairs import analyze_confusion

def main():
    print("🏁 Prepare the dataset 🏁")
    prepare_wav_files()

    print("🏁 Extract embeddings for phonemes 🏁")
    extract_embeddings_for_phonemes()

    print("🏁 Extract embeddings 🏁")
    embeddings()

    print("🏁 Batch test phonemes 🏁")
    batch_test_phonemes()

    input("Test the phoneme in another command line with w5_record_voice_cli.py. Press Enter to continue...")


    print("🏁 Benchmark inference and save 🏁")
    benchmark_and_save()

    print("🏁 Trace MLP classifier 🏁")
    trace_mlp_classifier()

    print("🏁 Visualize Results 🏁")
    visualize_results()

    print("🏁 Export to ONNX 🏁")
    onnx_export()

    print("🏁 Test ONNX model 🏁")
    onnx_test()

    print("🏁 Analyze confusion matrix 🏁")
    analyze_confusion()

    print("🏁 All steps completed successfully. 🏁")

if __name__ == "__main__":
    main()
    