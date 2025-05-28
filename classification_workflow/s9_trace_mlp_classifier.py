import torch
import pickle
import os

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
        class SklearnMLPWrapper(torch.nn.Module):
            def __init__(self, sk_clf):
                super().__init__()
                in_size = sk_clf.coefs_[0].shape[0]
                H       = sk_clf.coefs_[0].shape[1]
                out_size = sk_clf.coefs_[-1].shape[1]
                self.fc1 = torch.nn.Linear(in_size,  H)
                self.act = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(H, out_size)
                self.fc1.weight.data = torch.tensor(sk_clf.coefs_[0].T, dtype=torch.float32)
                self.fc1.bias.data   = torch.tensor(sk_clf.intercepts_[0], dtype=torch.float32)
                self.fc2.weight.data = torch.tensor(sk_clf.coefs_[1].T, dtype=torch.float32)
                self.fc2.bias.data   = torch.tensor(sk_clf.intercepts_[1], dtype=torch.float32)
            def forward(self, x):
                x = self.fc1(x)
                x = self.act(x)
                x = self.fc2(x)
                return x
        pt_clf = SklearnMLPWrapper(sk_clf).eval()
        dummy_emb = torch.randn(1, sk_clf.coefs_[0].shape[0])
        traced_clf = torch.jit.trace(pt_clf, dummy_emb)
        traced_clf.save(out_path) # type: ignore
        print(f"✅ Saved traced classifier: {out_path}")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"❌ Failed to save {out_path}!")

if __name__ == "__main__":
    trace_mlp_classifier()