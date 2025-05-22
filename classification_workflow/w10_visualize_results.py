import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
import csv
from pathlib import Path
from collections import Counter

EMBEDDINGS_DIR = Path("phoneme_embeddings")
METADATA_PATH = EMBEDDINGS_DIR / "metadata.csv"

def load_embeddings_and_labels():
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
    filtered_data = [(emb, label) for emb, label in zip(embeddings, labels) if class_counts[label] >= 2]
    filtered_embeddings, filtered_labels = zip(*filtered_data)
    return np.array(filtered_embeddings), np.array(filtered_labels)

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues") # type:ignore
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_per_class_accuracy(y_true, y_pred, label_encoder):
    report = classification_report(y_true, y_pred, output_dict=True, labels=np.unique(y_true), target_names=label_encoder.inverse_transform(np.unique(y_true)))
    accuracies = [report[cls]["precision"] for cls in report if cls in label_encoder.classes_] # type:ignore
    labels = [cls for cls in report if cls in label_encoder.classes_]

    plt.figure(figsize=(14, 6))
    sns.barplot(x=labels, y=accuracies)
    plt.xticks(rotation=45)
    plt.ylabel("Precision")
    plt.title("Per-Class Precision")
    plt.tight_layout()
    plt.show()

def plot_tsne_umap(X, y, label_encoder):
    print("Fitting t-SNE...")
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
    print("Fitting UMAP...")
    X_umap = umap.UMAP(random_state=42).fit_transform(X)

    def plot_2d(data, title):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=label_encoder.inverse_transform(y), palette="tab20", s=40, alpha=0.7, edgecolor="k")
        plt.title(title)
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    plot_2d(X_tsne, "t-SNE of Phoneme Embeddings")
    plot_2d(X_umap, "UMAP of Phoneme Embeddings")

def visualize_results():
    # Load classifier & encoder
    with open("dist/phoneme_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("dist/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Load data
    X, labels = load_embeddings_and_labels()
    y = le.transform(labels)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Predict
    y_pred = clf.predict(X_test)

    # Visualizations
    plot_confusion_matrix(y_test, y_pred, le)
    plot_per_class_accuracy(y_test, y_pred, le)
    plot_tsne_umap(X_test, y_test, le)

if __name__ == "__main__":
    visualize_results()
