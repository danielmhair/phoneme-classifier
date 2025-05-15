import os

AUDIO_DIR = "recordings"  # 🔁 Replace with your folder path
ALLOWED_EXTS = (".wav")

def review_files():
    for root, _, files in os.walk(AUDIO_DIR):
        for fname in sorted(files):
            if not fname.endswith(ALLOWED_EXTS):
                continue
            if fname.startswith("n_"):
                # delete the file
                os.remove(os.path.join(root, fname))
                print(f"❌ Deleted: {fname}")
                continue

if __name__ == "__main__":
    review_files()