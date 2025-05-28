import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib
from pathlib import Path

# --- OUTPUT SHAPE CHECK BEFORE EXPORT ---
CLASSIFIER_PATH = 'dist/phoneme_classifier_finetuned.pkl' if Path('dist/phoneme_classifier_finetuned.pkl').exists() else 'dist/phoneme_classifier.pkl'
LABEL_ENCODER_PATH = 'dist/label_encoder.pkl'
EXPECTED_CLASSES = 37

try:
    clf = joblib.load(CLASSIFIER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(label_encoder.classes_)
    if hasattr(clf, 'coefs_') and len(clf.coefs_) > 0:
        output_size = clf.coefs_[-1].shape[1]
        print(f"[DEBUG] Classifier output layer shape: {output_size}, Label encoder classes: {n_classes}")
        if output_size != EXPECTED_CLASSES or n_classes != EXPECTED_CLASSES:
            raise ValueError(f"[ERROR] Output shape mismatch: classifier output={output_size}, label encoder={n_classes}, expected={EXPECTED_CLASSES}. Aborting ONNX export.")
    else:
        print("[WARNING] Could not determine classifier output shape. Skipping shape check.")
except Exception as e:
    print(f"[FATAL] Error during output shape check: {e}")
    exit(1)

# 1️⃣ Define a small wrapper that mean-pools the last_hidden_state
class Wav2Vec2Pooled(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model.eval()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, T]
        outputs = self.base(audio)               # BaseModelOutput(last_hidden_state=[B, L, H], …)
        hidden = outputs.last_hidden_state       # [B, L, H]
        return hidden.mean(dim=1)                # → [B, H]

def onnx_export():    
    # 2️⃣ Load HF processor & base model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    base_w2v  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
    pooled_w2v = Wav2Vec2Pooled(base_w2v)

    # 3️⃣ Load your traced MLP (expects [1 × H])
    # Use fine-tuned traced model if available
    traced_pt_path = "dist/phoneme_clf_traced_finetuned.pt"
    if os.path.exists(traced_pt_path):
        print("Using fine-tuned traced MLP: phoneme_clf_traced_finetuned.pt")
    else:
        traced_pt_path = "dist/phoneme_clf_traced.pt"
        print("Using original traced MLP: phoneme_clf_traced.pt")
    mlp = torch.jit.load(traced_pt_path).eval()

    # 4️⃣ Create a dummy 2s audio vector → run through pooled_w2v to get dummy_emb
    dummy_audio = torch.randn(1, 16000 * 2, dtype=torch.float32)  # [1, T]
    with torch.no_grad():
        dummy_emb = pooled_w2v(dummy_audio)                       # [1, H]


    # 5️⃣ Export pooled wav2vec2 → embedding ([B, H])
    torch.onnx.export(
        pooled_w2v,
        (dummy_audio,),
        "dist/wav2vec2.onnx",
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={
            "audio":     {1: "num_samples"},   # allow varying T
            "embedding": {1: "hidden_len"},    # allow varying H if you change the model
        },
        opset_version=14,
    )

    # 6️⃣ Export embedding → phoneme_probs ([B, num_classes])
    torch.onnx.export(
        mlp,
        (dummy_emb,),
        "dist/phoneme_mlp.onnx",
        input_names=["embedding"],
        output_names=["phoneme_probs"],
        dynamic_axes={
            "embedding":     {1: "hidden_len"},
            "phoneme_probs": {1: "num_classes"},
        },
        opset_version=14,
    )

    # After exporting ONNX models
    missing_files = [file for file in ["dist/phoneme_mlp.onnx", "dist/wav2vec2.onnx"] if not os.path.exists(file)]
    if len(missing_files) > 0:
        raise FileNotFoundError("❌ Failed to save: " + ", ".join(missing_files))
    print("✅ Exported wav2vec2.onnx (pooled) and phoneme_mlp.onnx")
