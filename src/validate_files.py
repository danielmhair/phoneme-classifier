import os
import pygame

# -------------------------
# CONFIGURATION
# -------------------------
AUDIO_DIR = "recordings"  # 🔁 Replace with your folder path
ALLOWED_EXTS = (".wav")

# -------------------------
# INIT AUDIO
# -------------------------
pygame.mixer.init()

def play_audio(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    print(f"🔊 {path}: {os.path.basename(path)}")
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# -------------------------
# MAIN LOOP
# -------------------------
def review_files():
    for root, _, files in os.walk(AUDIO_DIR):
        for fname in sorted(files):
            if not fname.endswith(ALLOWED_EXTS):
                continue
            if fname.startswith("n_"):
                continue  # already marked bad
            # if folder we are in is g, dh, I_aɪ, ir_ɝ, y_j, l, then skip
            if os.path.basename(os.path.dirname(root)) in ["chloe"]:
                continue  # skip chloe folder
            if os.path.basename(os.path.dirname(root)) in ["callie"]:
                if os.path.basename(root) in ["f", "ch", "sh", "th"]:
                    print(f"Skipping file: {fname} in {os.path.basename(root)}")
                    continue

            fpath = os.path.join(root, fname)
            play_audio(fpath)

            while True:
                action = input("✅ [y] keep / ❌ [n] mark bad / 🔁 [r] replay: ").strip().lower()
                if action == "y":
                    break  # good, move to next
                elif action == "n":
                    bad_name = os.path.join(root, "n_" + fname)
                    os.rename(fpath, bad_name)
                    print(f"❌ Marked as bad: {bad_name}")
                    break
                elif action == "r":
                    play_audio(fpath)
                else:
                    print("Please press 'y', 'n', or 'r'.")

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    review_files()
