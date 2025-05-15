from os.path import dirname
import torch
import pickle
import os

def trace_mlp_classifier():
    # Load your trained MLP classifier
    full_path = "src/phoneme_classifier.pkl"
    with open(full_path, "rb") as f:
        sk_clf = pickle.load(f)


    # 2. Build equivalent PyTorch module
    class SklearnMLPWrapper(torch.nn.Module):
        def __init__(self, sk_clf):
            super().__init__()
            # assume sk_clf.hidden_layer_sizes = (H,)
            in_size = sk_clf.coefs_[0].shape[0]
            H       = sk_clf.coefs_[0].shape[1]
            out_size = sk_clf.coefs_[-1].shape[1]
            # define layers
            self.fc1 = torch.nn.Linear(in_size,  H)
            self.act = torch.nn.ReLU()               # sklearn uses ReLU by default
            self.fc2 = torch.nn.Linear(H, out_size)
            # copy weights & biases
            self.fc1.weight.data = torch.tensor(sk_clf.coefs_[0].T, dtype=torch.float32)
            self.fc1.bias.data   = torch.tensor(sk_clf.intercepts_[0], dtype=torch.float32)
            self.fc2.weight.data = torch.tensor(sk_clf.coefs_[1].T, dtype=torch.float32)
            self.fc2.bias.data   = torch.tensor(sk_clf.intercepts_[1], dtype=torch.float32)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            return x

    # 3. Instantiate & trace
    pt_clf = SklearnMLPWrapper(sk_clf).eval()
    dummy_emb = torch.randn(1, sk_clf.coefs_[0].shape[0])
    traced_clf = torch.jit.trace(pt_clf, dummy_emb)
    traced_clf.save("src/phoneme_clf_traced.pt")
    print("Saved!")

if __name__ == "__main__":
    trace_mlp_classifier()