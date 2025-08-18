import os
import time
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import torch.nn as nn

# === Wrapper for tracing ===
class Wav2Vec2EmbeddingWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_values):
        output = self.model(input_values)
        return output.last_hidden_state.mean(dim=1)


def benchmark(processor, audio, model_instance, label="Base", use_padding=True, save=False, warmup=3, runs=10):
    print(f"\n🧪 Benchmark: {label}")
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=use_padding)
    model_instance.eval()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model_instance(**inputs)

    # Timed runs
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model_instance(**inputs)
    end = time.time()
    avg_time = (end - start) / runs
    print(f"⏱️ Average Inference Time: {avg_time:.4f} seconds")
    if save:
        print("🔒 Saving traced model...")
        model_instance.save("dist/wav2vec2_traced_mean.pt")

def benchmark_and_save():
    # === Load audio sample ===
    audio, sr = sf.read("recordings/dan/sh/dan_sh_ep-ʃ_20250329_100822_5.wav")
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        raise ValueError("Audio must be 16kHz mono")

    # === Load Processor and Model ===
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    base_model.eval()

    # Base model benchmarks
    benchmark(processor, audio, base_model, "Base Model (with padding)", use_padding=True)
    benchmark(processor, audio, base_model, "Base Model (no padding)", use_padding=False)

    # TorchScript benchmark using wrapped model
    wrapped_model = Wav2Vec2EmbeddingWrapper(base_model)
    try:
        example_input = processor(audio, sampling_rate=16000, return_tensors="pt")["input_values"] # type: ignore
        traced_model = torch.jit.trace(wrapped_model, example_input)
        benchmark(processor, audio, traced_model, "TorchScript (mean pooled)", use_padding=False, save=True)
    except Exception as e:
        print(f"⚠️ TorchScript failed: {e}")
        raise e

    # Torch Compile benchmark
    try:
        compiled_model = torch.compile(wrapped_model, mode="reduce-overhead")
        benchmark(processor, audio, compiled_model, "Torch Compile (mean pooled)", use_padding=False)
    except Exception as e:
        print(f"⚠️ Torch Compile failed: {e}")
        raise e
    
    # Verify files created
    if not os.path.exists("dist/wav2vec2_traced_mean.pt"):
        raise FileNotFoundError("❌ Failed to save dist/wav2vec2_traced_mean.pt!")

# === Main ===
if __name__ == "__main__":
    benchmark_and_save()

