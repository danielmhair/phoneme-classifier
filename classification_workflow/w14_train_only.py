import csv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle

# Paths
EMBEDDINGS_DIR = Path("phoneme_embeddings")
METADATA_PATH = EMBEDDINGS_DIR / "metadata.csv"


def train_only():
    embeddings = []
    labels = []

    with open(METADATA_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emb_file = EMBEDDINGS_DIR / row["embedding_filename"]
            if emb_file.exists():
                emb = np.load(emb_file)
                embeddings.append(emb)
                labels.append(row["phoneme"])

    # Filter rare classes
    class_counts = Counter(labels)
    filtered_data = [(e, l) for e, l in zip(embeddings, labels) if class_counts[l] >= 2]
    if not filtered_data:
        raise ValueError("No classes have at least 2 samples!")

    X, y_raw = zip(*filtered_data)
    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Split for training only (don't care about test here)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    with open("dist/phoneme_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("dist/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Classifier retrained and saved without evaluation.")


if __name__ == "__main__":
    train_only()
