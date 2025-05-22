import os
import torch, soundfile as sf, numpy as np
import onnxruntime as ort

def onnx_test():
    # Load your TorchScript MLP and wav2vec2
    wav2vec = torch.jit.load("dist/wav2vec2_traced_mean.pt").eval()
    mlp = torch.jit.load("dist/phoneme_clf_traced.pt").eval()

    # Create ONNX Runtime sessions
    sess_w2v = ort.InferenceSession("dist/wav2vec2.onnx")
    sess_mlp = ort.InferenceSession("dist/phoneme_mlp.onnx")

    def to_onx(audio: np.ndarray):
        # run wav2vec2 → embedding
        emb = sess_w2v.run(
            ["embedding"], {"audio": audio.astype(np.float32)[None,:]}
        )[0]
        # run embedding → phoneme probs
        return sess_mlp.run(
            ["phoneme_probs"], {"embedding": emb}
        )[0]

    # for each folder in recordings/dan folder, Test a random clip per phoneme
    folders = os.listdir("recordings/dan")
    for folder in folders:
        files = os.listdir(f"recordings/dan/{folder}")
        # get a random file
        if len(files) == 0:
            continue
        # get a random index from files
        idx = np.random.randint(0, len(files))
        file = files[idx]
        audio, sr = sf.read(f"recordings/dan/{folder}/{file}")
        assert sr == 16000

        # TorchScript pipeline
        with torch.no_grad():
            input_tensor = torch.tensor(audio, dtype=torch.float32)[None, :]
            torch_emb = wav2vec(input_tensor)
            ts_probs  = mlp(torch_emb).numpy()

        # ONNX pipeline
        onnx_probs = to_onx(audio)

        # Compare
        print("TorchScript phoneme:", ts_probs.argmax())
        print("ONNX       phoneme:", onnx_probs.argmax())
