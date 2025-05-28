import torch
import pickle
import os
import hashlib

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def trace_mlp_classifier():
    # Save both original and fine-tuned traced MLPs if available
    for clf_name, out_path in [
        ("dist/phoneme_classifier.pkl", "dist/phoneme_clf_traced.pt"),
        ("dist/phoneme_classifier_finetuned.pkl", "dist/phoneme_clf_traced_finetuned.pt"),
    ]:
        if not os.path.exists(clf_name):
            print(f"Skipping {clf_name} (not found)")
            continue
        with open(clf_name, "rb") as f:
            sk_clf = pickle.load(f)
        # --- DEBUG: Print output layer shape ---
        if hasattr(sk_clf, 'coefs_') and len(sk_clf.coefs_) > 0:
            out_size = sk_clf.coefs_[-1].shape[1]
            print(f"[DEBUG] Tracing model: {clf_name}, output layer shape: {out_size}")
        else:
            print(f"[WARNING] Could not determine output shape for {clf_name}")
        class SklearnMLPWrapper(torch.nn.Module):
            def __init__(self, sk_clf):
                super().__init__()
                # Dynamically build layers to match sklearn MLP
                self.layers = torch.nn.ModuleList()
                self.activations = torch.nn.ModuleList()
                layer_sizes = [w.shape[0] for w in sk_clf.coefs_] + [sk_clf.coefs_[-1].shape[1]]
                for i in range(len(sk_clf.coefs_)):
                    self.layers.append(torch.nn.Linear(sk_clf.coefs_[i].shape[0], sk_clf.coefs_[i].shape[1]))
                    if i < len(sk_clf.coefs_) - 1:
                        self.activations.append(torch.nn.ReLU())
                # Assign weights and biases
                for i, layer in enumerate(self.layers):
                    layer.weight.data = torch.tensor(sk_clf.coefs_[i].T, dtype=torch.float32)
                    layer.bias.data = torch.tensor(sk_clf.intercepts_[i], dtype=torch.float32)
            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i < len(self.activations):
                        x = self.activations[i](x)
                return x
        pt_clf = SklearnMLPWrapper(sk_clf).eval()
        # Use the input size from the first sklearn layer
        dummy_emb = torch.randn(1, sk_clf.coefs_[0].shape[0])
        traced_clf = torch.jit.trace(pt_clf, dummy_emb)
        traced_clf.save(out_path) # type: ignore
        print(f"✅ Saved traced classifier: {out_path}")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"❌ Failed to save {out_path}!")
        # --- DEBUG: Print hash and output shape after reload ---
        sha = file_sha256(out_path)
        print(f"[DEBUG] Traced file hash: {sha}")
        reloaded = torch.jit.load(out_path)
        try:
            dummy = torch.randn(1, sk_clf.coefs_[0].shape[0])
            test_probs = reloaded(dummy)
            print(f"[DEBUG] Reloaded traced model output shape: {test_probs.shape}")
        except Exception as e:
            print(f"[ERROR] Could not infer output shape after reload: {e}")

if __name__ == "__main__":
    trace_mlp_classifier()