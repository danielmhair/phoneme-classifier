# Phoneme Classifier

AI-powered phoneme classification system for children's speech recognition using dual approaches: traditional MLP and modern CTC sequence modeling with Wav2Vec2 features and Teacher-student distillation with Whisper.

This project is defined completely by this theme in Notion:

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

## Current Epic & Task

Currently on - [Epic - Live Phoneme CTCs](https://www.notion.so/Epic-Live-Phoneme-CTCs-22b502b4855680149d70eec42adf84d3?pvs=21).
Just added ctc. Haven't ran anything or tested. Just solidifying the initial [workflows/ctc_w2v2_workflow](./workflows/ctc_w2v2_workflow).

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# Install Poetry (if needed)
curl -sSL https://install.python-poetry.org | python3 -

# Complete setup: install dependencies + CUDA support
poe setup

# Or step by step:
poetry install
poetry add --group dev poethepoet
poe setup-cuda
```

### Train Your First Model

```bash
# Train MLP classifier (fast, traditional approach)
poe train-mlp

# Test the trained model
poe test-pkl

# Interactive phoneme recording
poe record-cli
```

## 🎯 Available Workflows

### **MLP Control Workflow** (Traditional)

- **Fast training** with scikit-learn
- **Single phoneme classification**
- **Low memory usage**
- **Quick prototyping**

```bash
poe train-mlp     # Train MLP classifier
poe test-pkl      # Test pickle model
```

### **CTC Wav2Vec2 Workflow** (Advanced)

- **Sequence modeling** with PyTorch + LSTM
- **Temporal phoneme sequences**
- **Alignment-free training**
- **Real-time capable**

```bash
poe train-ctc     # Train CTC model
poe test-ctc      # Test CTC inference
```

## 🛠️ Development Commands

### Training & Testing

```bash
poe train-all     # Train both MLP and CTC workflows
poe test-all      # Test all trained models
poe full-pipeline # Complete pipeline: train + test + export
```

### Development Tools

```bash
poe lint          # Code quality check
poe format        # Format code (black + isort)
poe clean         # Clean build artifacts
poe info          # Show project overview
poe workflows     # List all available commands
```

### Debugging

```bash
poe debug-shared  # Test shared utilities
poe debug-mlp     # Test MLP workflow
poe debug-ctc     # Test CTC model structure
```

## 📊 Model Outputs

The system generates multiple deployment formats:

- **`.pkl`** - Scikit-learn models (fast Python inference)
- **`.pt`** - PyTorch traced models (advanced features)
- **`.onnx`** - ONNX models (Unreal Engine, web deployment)
- **`phoneme_labels.json`** - Label mappings

## 🏗️ Architecture

### Dual Workflow System

```text
phoneme-classifier/
├── workflows/
│   ├── mlp_control_workflow/    # Traditional MLP approach
│   ├── ctc_w2v2_workflow/       # Modern CTC approach  
│   └── shared/                  # Shared utilities
├── recordings/                  # Training data (phoneme audio)
├── dist/                       # Generated models & outputs
└── logs/                       # Training logs
```

### Data Flow

```text
Audio Recordings → Wav2Vec2 Features → ML Training → Model Export
     ↓                    ↓                ↓             ↓
  37 phonemes      768-dim vectors    MLP/CTC     .pkl/.pt/.onnx
```

## 🎮 Project Epics

This project follows a structured epic-based development approach:

1. **[🚧 Live Phoneme CTCs](https://www.notion.so/Epic-Live-Phoneme-CTCs-22b502b4855680149d70eec42adf84d3?pvs=21)** - CTC implementation in progress - critical blockers remain
2. **[Live Streaming Improvements](https://www.notion.so/Epic-Live-Streaming-Improvements-Temporal-Brain-22b502b48556801c86f0f3f5a7036010?pvs=21)** - Temporal processing
3. **[Whisper Teacher & Distillation](https://www.notion.so/Epic-Whisper-Teacher-Distillation-22b502b4855680da8047e51acd13ef1e?pvs=21)** - Model distillation
4. **[Multi-Model Bake-off Harness](https://www.notion.so/Epic-Multi-Model-Bake-off-Harness-22b502b485568092ab79fe7ec3901b36?pvs=21)** - Benchmarking
5. **[Game for Data Collection](https://www.notion.so/Epic-Game-for-Data-Collection-22b502b4855680dfa7d6f0c8ea071806?pvs=21)** - Interactive data collection
6. **[Offline Rescoring & Parent Summaries](https://www.notion.so/Epic-Offline-Rescoring-Parent-Summaries-22b502b485568051b73efd500dd632f3?pvs=21)** - Analysis tools
7. **[Personalization & Practice Packs](https://www.notion.so/Epic-Personalization-Practice-Packs-22b502b48556807a9d2ade60a605d358?pvs=21)** - Adaptive learning
8. **[Data Enrichment & Noise Robustness](https://www.notion.so/Epic-Data-Enrichment-Noise-Robustness-22b502b485568027a789c927a85a096b?pvs=21)** - Data augmentation
9. **[Evaluation & Progress Gates](https://www.notion.so/Epic-Evaluation-Progress-Gates-22b502b4855680dcb4f3e071691c4957?pvs=21)** - Quality assurance
10. **[Model Update & Export Pipeline](https://www.notion.so/Epic-Model-Update-Export-Pipeline-22b502b485568049af1fe48dcff0d011?pvs=21)** - Deployment automation

**Current Focus**: Epic 1 (Live Phoneme CTCs) - 🚧 **IN PROGRESS** - Critical blockers prevent completion. ONNX export to games not achieved.

## 🎯 Key Features

- **🔥 Two ML Approaches**: Traditional MLP + Modern CTC sequence modeling  
- **🎵 Audio Processing**: Complete Wav2Vec2 → embeddings → classification pipeline
- **🎮 Game Integration**: ONNX export for deployment of Unreal Engine, browser, and mobile games
- **🧪 Interactive Testing**: Real-time phoneme recording and classification
- **📊 Rich Visualization**: UMAP plots, confusion matrices, performance metrics
- **🔄 Cross-Platform**: WSL/Linux training → Windows deployment

## 📚 Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Detailed development commands and architecture
- **[SUPERCLAUDE.md](./SUPERCLAUDE.md)** - SuperClaude framework integration  
- **[docs/](./docs/)** - Architecture analysis and implementation summaries
- **[Notion Theme](https://www.notion.so/Theme-Automatic-Phoneme-Recognition-APR-22b502b4855680cba080ea07b68a3127)** - Complete project specification

## 🤝 Contributing

1. **Setup**: `poe setup`
2. **Develop**: Make changes, `poe format`, `poe lint`
3. **Test**: `poe test-all`
4. **Validate**: `poe full-pipeline`

## 📄 License

See project documentation for licensing details.
