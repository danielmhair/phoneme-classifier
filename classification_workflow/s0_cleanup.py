import shutil
from pathlib import Path

def cleanup(folders=None):
    if folders is not None:
        for file in folders:
            f = Path(".") / file
            if f.exists():
                print(f"Removing directory: {f}")
                shutil.rmtree(f)

        # Create directories
        for folder in folders:
            p = Path(folder)
            if not p.exists():
                print(f"Creating directory: {p}")
                p.mkdir(parents=True, exist_ok=True)
        print("✅ Cleanup complete.")

    else:
        print("No folders specified for cleanup.")