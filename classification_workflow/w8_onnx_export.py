import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

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
    mlp = torch.jit.load("src/phoneme_clf_traced.pt").eval()

    # 4️⃣ Create a dummy 2s audio vector → run through pooled_w2v to get dummy_emb
    dummy_audio = torch.randn(1, 16000 * 2, dtype=torch.float32)  # [1, T]
    with torch.no_grad():
        dummy_emb = pooled_w2v(dummy_audio)                       # [1, H]

    # 5️⃣ Export pooled wav2vec2 → embedding ([B, H])
    torch.onnx.export(
        pooled_w2v,
        (dummy_audio,),
        "src/wav2vec2.onnx",
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
        "src/phoneme_mlp.onnx",
        input_names=["embedding"],
        output_names=["phoneme_probs"],
        dynamic_axes={
            "embedding":     {1: "hidden_len"},
            "phoneme_probs": {1: "num_classes"},
        },
        opset_version=14,
    )

    print("✅ Exported wav2vec2.onnx (pooled) and phoneme_mlp.onnx")
