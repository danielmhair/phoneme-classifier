set -ex

# Epic 1: Three-Model ONNX Deployment Script
# Updated to include all three model types: MLP Control, Wav2Vec2 CTC, WavLM CTC

# Create target directory
mkdir -p C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist

# Copy shared metadata
echo "Copying shared model metadata..."
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\phoneme_labels.json C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\phoneme_labels.json

# MLP Control Model (Original baseline)
echo "Copying MLP Control ONNX model..."
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\phoneme_mlp.onnx C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\phoneme_mlp.onnx

# Wav2Vec2 CTC Model (Sequence modeling)
echo "Copying Wav2Vec2 CTC ONNX model..."
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\wav2vec2_ctc.onnx C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\wav2vec2_ctc.onnx

# WavLM CTC Model (Best performance - 85.35% accuracy)
echo "Copying WavLM CTC ONNX model..."
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\wavlm_ctc.onnx C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\wavlm_ctc.onnx

# Copy additional metadata files if available
echo "Copying additional model metadata..."
cp \\\\wsl.localhost\\Ubuntu\\home\\danie\\Workspaces\\fast-api-phoneme-python\\dist\\*_metadata.json C:\\Users\\danie\\Documents\\MyProject\\AiServer\\fast-api-python\\dist\\ 2>/dev/null || echo "No additional metadata files found"

# Activate Python environment
./.venv/Scripts/activate

# Run inference with WavLM CTC (best performing model)
echo "Running inference with WavLM CTC model (85.35% accuracy)..."
c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/.venv/Scripts/python.exe c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/control_mlp_workflow/s103_classify_voice_onnx.py --model wavlm_ctc

# Optional: Test all three models for comparison
# Uncomment the lines below for comprehensive testing
# echo "Testing MLP Control model..."
# c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/.venv/Scripts/python.exe c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/control_mlp_workflow/s103_classify_voice_onnx.py --model mlp
# echo "Testing Wav2Vec2 CTC model..."  
# c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/.venv/Scripts/python.exe c:/Users/danie/Documents/MyProject/AiServer/fast-api-python/control_mlp_workflow/s103_classify_voice_onnx.py --model wav2vec2_ctc

echo "Epic 1 three-model deployment complete!"
