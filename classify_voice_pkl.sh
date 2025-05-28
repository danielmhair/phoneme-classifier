set -ex

# Copy files over
mkdir -p C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\phoneme_labels.json C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\phoneme_labels.json
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\phoneme_classifier.pkl C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\phoneme_classifier.pkl
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\label_encoder.pkl C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\label_encoder.pkl

c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/.venv/Scripts/python.exe c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/classification_workflow/s103_classify_voice_pkl.py
