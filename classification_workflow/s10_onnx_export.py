import os
import time
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib
from pathlib import Path

# --- 1️⃣ Define mean-pooling wrapper for wav2vec2 ---
class Wav2Vec2Pooled(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model.eval()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        outputs = self.base(audio)               # BaseModelOutput(last_hidden_state=[B, L, H])
        hidden = outputs.last_hidden_state       # [B, L, H]
        return hidden.mean(dim=1)                # → [B, H]

def onnx_export():
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
    
    # --- 2️⃣ Load base wav2vec2 ---
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    base_w2v  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
    pooled_w2v = Wav2Vec2Pooled(base_w2v)

    # --- DEBUG: Print traced model file info before loading ---
    TRACED_MODEL_PATH = 'dist/phoneme_clf_traced_finetuned.pt' if os.path.exists('dist/phoneme_clf_traced_finetuned.pt') else 'dist/phoneme_clf_traced.pt'
    print(f"[DEBUG] Loading traced model: {TRACED_MODEL_PATH}")
    if os.path.exists(TRACED_MODEL_PATH):
        mtime = os.path.getmtime(TRACED_MODEL_PATH)
        print(f"[DEBUG] Traced model mtime: {time.ctime(mtime)}")
    else:
        print(f"[ERROR] Traced model not found: {TRACED_MODEL_PATH}")

    # --- 3️⃣ Load traced MLP classifier ---
    traced_pt_path = "dist/phoneme_clf_traced_finetuned.pt"
    if os.path.exists(traced_pt_path):
        print("Using fine-tuned traced MLP: phoneme_clf_traced_finetuned.pt")
    else:
        traced_pt_path = "dist/phoneme_clf_traced.pt"
        print("Using original traced MLP: phoneme_clf_traced.pt")
    mlp = torch.jit.load(traced_pt_path).eval()

    # --- 4️⃣ Run dummy data through to get embedding ---
    dummy_audio = torch.randn(1, 16000 * 2, dtype=torch.float32)  # [1, T]
    with torch.no_grad():
        dummy_emb = pooled_w2v(dummy_audio)                       # [1, H]

        # 🔎 Sanity check: does MLP produce 37 output classes?
        test_probs = mlp(dummy_emb)
        if test_probs.shape[1] != EXPECTED_CLASSES:
            raise ValueError(f"[FATAL] ONNX export failed: MLP output {test_probs.shape[1]} ≠ expected {EXPECTED_CLASSES}. Re-train & re-trace MLP.")

    # --- 5️⃣ Export wav2vec2 pooled model ---
    torch.onnx.export(
        pooled_w2v,
        (dummy_audio,),
        "dist/wav2vec2.onnx",
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={
            "audio": {1: "num_samples"},
            "embedding": {1: "hidden_len"}
        },
        opset_version=14,
    )

    # --- 6️⃣ Export MLP phoneme classifier ---
    torch.onnx.export(
        mlp,
        (dummy_emb,),
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
        raise FileNotFoundError(f"[❌] Missing export(s): {', '.join(missing)}")
    print("✅ Successfully exported: wav2vec2.onnx + phoneme_mlp.onnx")
