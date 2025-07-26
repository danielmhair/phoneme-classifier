import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ✅ Wrapper that applies Wav2Vec2 Feature Extractor logic + model + mean pooling
class Wav2Vec2WithPreprocessing(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval()

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        # raw_audio: [B, T] in range [-1, 1]
        # → manually apply the same preprocessing that HuggingFace does in processor(audio)

        # Normalize to mean 0, std 1 (per waveform)
        mean = raw_audio.mean(dim=1, keepdim=True)
        std = raw_audio.std(dim=1, keepdim=True).clamp(min=1e-9)
        normed = (raw_audio - mean) / std  # [B, T]

        # Apply wav2vec2
        out = self.model(normed)
        hidden = out.last_hidden_state      # [B, L, H]
        return hidden.mean(dim=1)           # → [B, H]

def onnx_export():
    # 1. Create the model
    model = Wav2Vec2WithPreprocessing()

    # 2. Dummy input (1 second of audio @ 16kHz)
    dummy_input = torch.randn(1, 16000, dtype=torch.float32)  # [B, T] waveform

    # 3. Load your existing MLP classifier
    mlp = torch.jit.load("dist/phoneme_clf_traced.pt").eval()

    # 4. Export wav2vec2 full model with preprocessing
    torch.onnx.export(
        model,
        (dummy_input,),
        "dist/wav2vec2.onnx",
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={
            "audio": {1: "num_samples"},
            "embedding": {1: "hidden_len"}
        },
        opset_version=14,
    )

    # 5. Run dummy audio through feature extractor to get input shape for MLP
    with torch.no_grad():
        dummy_embedding = model(dummy_input)  # [B, H]

    # 6. Export phoneme classifier (same as before)
    torch.onnx.export(
        mlp,
        (dummy_embedding,),
        "dist/phoneme_mlp.onnx",
        input_names=["embedding"],
        output_names=["phoneme_probs"],
        dynamic_axes={
            "embedding": {1: "hidden_len"},
            "phoneme_probs": {1: "num_classes"}
        },
        opset_version=14,
    )

    # ✅ Final check
    missing = [f for f in ["dist/wav2vec2.onnx", "dist/phoneme_mlp.onnx"] if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("❌ Export failed: " + ", ".join(missing))
    print("✅ Exported wav2vec2.onnx (with preprocessing) + phoneme_mlp.onnx")

