import os
import pickle
import csv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import argparse
import pandas as pd
from collections import Counter

def classifier_encoder(input_dir_str: str, classifier_out: str, encoder_out: str):
    # Load metadata and embeddings
    embeddings = []
    labels = []
    input_dir = Path(input_dir_str)
    metadata_csv = input_dir / "metadata.csv"
    meta = pd.read_csv(metadata_csv)
    
    for _, row in meta.iterrows():
        emb_path = input_dir / row["embedding_filename"]
        if emb_path.exists():
            emb = np.load(emb_path)
            embeddings.append(emb)
            labels.append(row["phoneme"])
        else:
            print(f"Missing embedding file: {row['embedding_filename']}")

    embeddings = np.stack(embeddings)
    labels = np.array(labels)
    print(f"Embeddings shape: {embeddings.shape}")

    # Check class distribution
    class_counts = Counter(labels)
    print("Class distribution before filtering:", class_counts)

    # Filter out classes with fewer than 2 samples
    filtered_data = [(emb, label) for emb, label in zip(embeddings, labels) if class_counts[label] >= 2]
    if not filtered_data:
        raise ValueError("No classes have at least 2 samples!")
    filtered_embeddings, filtered_labels = zip(*filtered_data)
    filtered_embeddings = np.array(filtered_embeddings)
    print("Filtered embeddings shape:", filtered_embeddings.shape)
    print("Filtered class distribution:", Counter(filtered_labels))

    print("Encode the phoneme labels into integers")
    le = LabelEncoder()
    y = le.fit_transform(filtered_labels)

    print("Split into training and testing sets (80/20 split)")
    X_train, X_test, y_train, y_test = train_test_split(
        filtered_embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Create and train an MLP classifier")
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    print("Evaluate the classifier")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Use only the classes present in the test set")
    unique_test_labels = np.unique(y_test)
    target_names = [le.inverse_transform([label])[0] for label in unique_test_labels]

    report = classification_report(y_test, y_pred, labels=unique_test_labels, target_names=target_names)

    print("Test Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:")
    print(report)

    # --- CHECK OUTPUT LAYER SHAPE ---
    n_classes = len(le.classes_) # type:ignore
    output_shape = clf.coefs_[-1].shape[1]
    if output_shape != n_classes or n_classes != 37:
        raise ValueError(f"Classifier output shape mismatch: got {output_shape} outputs, label encoder has {n_classes} classes, expected 37. Aborting training to prevent ONNX/class mismatch.")
    print(f"[Shape check] Classifier output: {output_shape}, Label encoder classes: {n_classes}")

    print("Save the trained classifier and label encoder")
    with open(classifier_out, "wb") as f:
        pickle.dump(clf, f)
    with open(encoder_out, "wb") as f:
        pickle.dump(le, f)

    # Check that files exist
    missing_files = [f for f in [classifier_out, encoder_out] if not os.path.exists(f)]
    if len(missing_files) > 0:
        raise Exception("❌ Failed to save: " + ", ".join(missing_files))
    print(f"✅ Classifier and label encoder saved to {classifier_out} and {encoder_out}.")
    print("✅ Embedding extraction and classification complete!")
