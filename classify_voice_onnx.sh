set -ex

# Copy files over
mkdir -p C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\phoneme_labels.json C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\phoneme_labels.json
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\wav2vec2.onnx C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\wav2vec2.onnx
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\phoneme_mlp.onnx C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\phoneme_mlp.onnx

./.venv/Scripts/activate

c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/.venv/Scripts/python.exe c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/control_mlp_workflow/s103_classify_voice_onnx.py
