# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a phoneme classification system that trains AI models to classify speech phonemes from children's voices.
This project is defined completely by this theme defined in Notion:

<https://www.notion.so/Theme-Automatic-Phoneme-Recognition-APR-22b502b4855680cba080ea07b68a3127>

Epics:

1. [Epic - Live Phoneme CTCs](https://www.notion.so/Epic-Live-Phoneme-CTCs-22b502b4855680149d70eec42adf84d3?pvs=21)
2. [Epic - Live Streaming Improvements (Temporal Brain)](https://www.notion.so/Epic-Live-Streaming-Improvements-Temporal-Brain-22b502b48556801c86f0f3f5a7036010?pvs=21)
3. [Epic - Whisper Teacher & Distillation](https://www.notion.so/Epic-Whisper-Teacher-Distillation-22b502b4855680da8047e51acd13ef1e?pvs=21)
4. [Epic - Multi-Model Bake-off Harness](https://www.notion.so/Epic-Multi-Model-Bake-off-Harness-22b502b485568092ab79fe7ec3901b36?pvs=21)
5. [Epic - Game for Data Collection](https://www.notion.so/Epic-Game-for-Data-Collection-22b502b4855680dfa7d6f0c8ea071806?pvs=21)
6. [Epic - Offline Rescoring & Parent Summaries](https://www.notion.so/Epic-Offline-Rescoring-Parent-Summaries-22b502b485568051b73efd500dd632f3?pvs=21)
7. [Epic - Personalization & Practice Packs](https://www.notion.so/Epic-Personalization-Practice-Packs-22b502b48556807a9d2ade60a605d358?pvs=21)
8. [Epic - Data Enrichment & Noise Robustness](https://www.notion.so/Epic-Data-Enrichment-Noise-Robustness-22b502b485568027a789c927a85a096b?pvs=21)
9. [Epic - Evaluation & Progress Gates](https://www.notion.so/Epic-Evaluation-Progress-Gates-22b502b4855680dcb4f3e071691c4957?pvs=21)
10. [Epic - Model Update & Export Pipeline](https://www.notion.so/Epic-Model-Update-Export-Pipeline-22b502b485568049af1fe48dcff0d011?pvs=21)

### Current Epic & Task

1. [Epic - Live Phoneme CTCs](https://www.notion.so/Epic-Live-Phoneme-CTCs-22b502b4855680149d70eec42adf84d3?pvs=21)

- <https://www.notion.so/Task-001-Setup-Global-Workflow-for-all-the-encoders-251502b4855680829126d97b8f7e20aa>

## Development Commands

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Additional PyTorch installation for CUDA (if needed)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Data Preparation and Training

```bash
# Run the complete phoneme classification workflow
cd mlp_control_workflow
python 0_workflow.py
```

### Model Testing

```bash
# Test with ONNX model
./classify_voice_onnx.sh

# Test with pickle model  
./classify_voice_pkl.sh

# Manual testing with specific scripts
python mlp_control_workflow/validations/classify_voice_onnx.py
python mlp_control_workflow/validations/classify_voice_pkl.py
python mlp_control_workflow/validations/record_phonemes_cli.py
```

### Code Quality

```bash
# Lint check (configured in .flake8)
flake8 mlp_control_workflow/
```

## Architecture Overview

### Core Workflow Pipeline

The main workflow (`0_workflow.py`) orchestrates 13 sequential steps:

1. **Data Cleanup** - Cleans previous dist/ outputs
2. **Data Preparation** - Organizes recordings from multiple sources and applies audio augmentation
3. **Embedding Extraction** - Uses Wav2Vec2 to extract embeddings from organized phoneme recordings
4. **Classifier Training** - Trains MLP classifier on embeddings with label encoding
5. **Visualization** - Generates UMAP plots and confusion matrices
6. **Analysis** - Analyzes confusion pairs and batch tests
7. **Benchmarking** - Performance testing and model saving
8. **Model Export** - Traces PyTorch models and exports to ONNX
9. **Unreal Integration** - Copies models to Unreal Engine project

### Key Components

- **Audio Processing**: Uses `soundfile`, `sounddevice`, `librosa` for audio I/O and processing
- **Feature Extraction**: Wav2Vec2 (`facebook/wav2vec2-base`) for speech embeddings  
- **Classification**: scikit-learn MLPClassifier for phoneme prediction
- **Model Export**: PyTorch tracing and ONNX export for deployment
- **Augmentation**: `audiomentations` for synthetic data generation with noise injection

### Data Flow

```text
recordings/ → organized_recordings/ → phoneme_embeddings/ → classifier.pkl
                ↓                           ↓                      ↓
        augmented_recordings/     embeddings.npy files     label_encoder.pkl
                                                                   ↓
                                                            traced_model.pt
                                                                   ↓
                                                              model.onnx
```

### Directory Structure

- `mlp_control_workflow/` - Main pipeline scripts (s0-s12)
- `mlp_control_workflow/utils/` - Audio processing utilities
- `mlp_control_workflow/validations/` - Testing and validation scripts
- `recordings/` - Source audio data (phoneme recordings by speaker)
- `dist/` - Generated outputs (models, embeddings, visualizations)
- `logs/` - Workflow execution logs with timestamps

## Model Formats

The system generates multiple model formats:

- `.pkl` files - Scikit-learn models for Python inference
- `.pt` files - PyTorch traced models
- `.onnx` files - ONNX models for Unreal Engine and browser games (with onnx.js) integration
- `phoneme_labels.json` - Label mapping for consistent inference

## Integration Notes

- Unreal Engine integration via ONNX models copied to Windows paths
- Cross-platform development (WSL/Linux training, Windows deployment)
- Real-time audio recording and classification capabilities
- Support for multiple microphone quality levels in training data
