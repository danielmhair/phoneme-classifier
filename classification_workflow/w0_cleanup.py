import shutil
from pathlib import Path

def cleanup():
    # List of folders to remove
    folders_to_remove = [
        "organized_recordings",
        "phoneme_embeddings",
    ]

    # List of files to remove (in project root or src/)
    files_to_remove = [
        "label_encoder.pkl",
        "phoneme_classifier.pkl",
        "phoneme_clf_traced.pt",
        "wav2vec2_traced_mean.pt",
        "phoneme_mlp.onnx",
        "wav2vec2.onnx",
    ]

    # Remove folders
    for folder in folders_to_remove:
        p = Path(folder)
        if p.exists() and p.is_dir():
            print(f"Removing folder: {p}")
            shutil.rmtree(p)

    # Remove files from root and src/
    for file in files_to_remove:
        for base in [Path("."), Path("dist")]:
            f = base / file
            if f.exists():
                print(f"Removing file: {f}")
                f.unlink()

    print("✅ Cleanup complete.")

if __name__ == "__main__":
    cleanup()