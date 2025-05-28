from pathlib import Path
import shutil
from tqdm import tqdm

def copy_with_progress(src: Path, dest: Path, desc="Copying"):
    total_size = src.stat().st_size
    with src.open("rb") as fsrc, dest.open("wb") as fdst, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=desc, ncols=80
    ) as pbar:
        while True:
            buf = fsrc.read(1024 * 1024)  # 1 MB chunks
            if not buf:
                break
            fdst.write(buf)
            pbar.update(len(buf))

def overwrite_onnx_unreal():
    phoneme_mlp_onnx_dest = Path("/mnt/c/Users/danie/Documents/MyProject/Content/Models/phoneme_mlp.onnx")
    wav2vec2_onnx_dest = Path("/mnt/c/Users/danie/Documents/MyProject/Content/Models/wav2vec2.onnx")
    phoneme_labels_dest = Path("/mnt/c/Users/danie/Documents/MyProject/Content/Models/phoneme_labels.json")

    # Delete dest files first
    if phoneme_mlp_onnx_dest.exists():
        print(f"Deleting {phoneme_mlp_onnx_dest}")
        phoneme_mlp_onnx_dest.unlink()
    if wav2vec2_onnx_dest.exists():
        print(f"Deleting {wav2vec2_onnx_dest}")
        wav2vec2_onnx_dest.unlink()
    if phoneme_labels_dest.exists():
        print(f"Deleting {phoneme_labels_dest}")
        phoneme_labels_dest.unlink()

    phoneme_mlp_onnx_src = Path("dist/phoneme_mlp.onnx")
    wav2vec2_onnx_src = Path("dist/wav2vec2.onnx")
    phoneme_labels_src = Path("dist/phoneme_labels.json")

    # Copy the files to the Unreal Engine project directory
    copy_with_progress(phoneme_mlp_onnx_src, phoneme_mlp_onnx_dest, desc="Copy MLP ONNX")
    copy_with_progress(wav2vec2_onnx_src, wav2vec2_onnx_dest, desc="Copy Wav2Vec2 ONNX")
    copy_with_progress(phoneme_labels_src, phoneme_labels_dest, desc="Copy Labels JSON")

    # Ensure all files exist
    if not phoneme_mlp_onnx_dest.exists():
        print(f"❌ Error: {phoneme_mlp_onnx_dest} does not exist after copying.")
    else:
        print(f"✅ Successfully copied {phoneme_mlp_onnx_src} to {phoneme_mlp_onnx_dest}")

    if not wav2vec2_onnx_dest.exists():
        print(f"❌ Error: {wav2vec2_onnx_dest} does not exist after copying.")
    else:
        print(f"✅ Successfully copied {wav2vec2_onnx_src} to {wav2vec2_onnx_dest}")

    if not phoneme_labels_dest.exists():
        print(f"❌ Error: {phoneme_labels_dest} does not exist after copying.")
    else:
        print(f"✅ Successfully copied {phoneme_labels_src} to {phoneme_labels_dest}")

    print("✅ All files copied successfully.")


if __name__ == "__main__":
    overwrite_onnx_unreal()