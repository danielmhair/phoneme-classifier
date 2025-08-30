# Phoneme Classifier - Epic 1 & 2: Live Phoneme CTCs + Temporal Brain ✅ COMPLETED!

AI-powered phoneme classification system for children's speech recognition featuring **three-way model comparison** and **real-time temporal stabilization**:

- **Epic 1**: Traditional MLP, Wav2Vec2 CTC, and advanced WavLM CTC sequence modeling with comprehensive performance analysis
- **Epic 2**: Real-time temporal brain CLI testing tool with stabilization algorithms and model hot-swapping

This project is defined completely by this theme in Notion:

<https://www.notion.so/Theme-Automatic-Phoneme-Recognition-APR-22b502b4855680cba080ea07b68a3127>

Epics:

1. [Epic - Live Phoneme CTCs](https://www.notion.so/Epic-Live-Phoneme-CTCs-22b502b4855680149d70eec42adf84d3?pvs=21) ✅ COMPLETED
2. [Epic - Live Streaming Improvements (Temporal Brain)](https://www.notion.so/Epic-Live-Streaming-Improvements-Temporal-Brain-22b502b48556801c86f0f3f5a7036010?pvs=21) ✅ IMPLEMENTED
3. [Epic - Whisper Teacher & Distillation](https://www.notion.so/Epic-Whisper-Teacher-Distillation-22b502b4855680da8047e51acd13ef1e?pvs=21)
4. [Epic - Multi-Model Bake-off Harness](https://www.notion.so/Epic-Multi-Model-Bake-off-Harness-22b502b485568092ab79fe7ec3901b36?pvs=21)
5. [Epic - Game for Data Collection](https://www.notion.so/Epic-Game-for-Data-Collection-22b502b4855680dfa7d6f0c8ea071806?pvs=21)
6. [Epic - Offline Rescoring & Parent Summaries](https://www.notion.so/Epic-Offline-Rescoring-Parent-Summaries-22b502b485568051b73efd500dd632f3?pvs=21)
7. [Epic - Personalization & Practice Packs](https://www.notion.so/Epic-Personalization-Practice-Packs-22b502b48556807a9d2ade60a605d358?pvs=21)
8. [Epic - Data Enrichment & Noise Robustness](https://www.notion.so/Epic-Data-Enrichment-Noise-Robustness-22b502b485568027a789c927a85a096b?pvs=21)
9. [Epic - Evaluation & Progress Gates](https://www.notion.so/Epic-Evaluation-Progress-Gates-22b502b4855680dcb4f3e071691c4957?pvs=21)
10. [Epic - Model Update & Export Pipeline](https://www.notion.so/Epic-Model-Update-Export-Pipeline-22b502b485568049af1fe48dcff0d011?pvs=21)

## 🎉 Epic 1: Live Phoneme CTCs - COMPLETED!

✅ **THREE-WAY MODEL COMPARISON ACHIEVED**: Successfully implemented and validated all three phoneme classification approaches:

## 🧠 Epic 2: Temporal Brain - IMPLEMENTED

✅ **REAL-TIME PHONEME STABILIZATION**: Successfully implemented temporal brain CLI testing tool with advanced stabilization algorithms and model hot-swapping capability!

| Model | Performance | Architecture | Key Features |
|-------|------------|-------------|--------------|
| **🥇 Wav2Vec2 CTC** | **87.00%** | PyTorch CTC + Facebook Wav2Vec2 | **BEST PERFORMER**: Sequence modeling excellence |
| **🥈 WavLM CTC** | **85.35%** | PyTorch CTC + Microsoft WavLM | Advanced speech representations, research model |
| **🥉 MLP Control** | **79.73%** | scikit-learn MLP | Speed champion: 4.2x faster than CTC models |

🏆 **Wav2Vec2 CTC achieved 87.00% accuracy** (1,740/2,000 correct), empirically proving superior phoneme recognition capabilities!

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

### Epic 1: Train & Compare All Three Models

```bash
# Epic 1: Train all three models for comprehensive comparison
poe train-all

# Test all models to compare performance
poe test-all

# Individual model training:
poe train-mlp           # MLP Control (baseline)
poe train-ctc           # Wav2Vec2 CTC 
poe train-wavlm-ctc     # WavLM CTC (best performance)

# Interactive phoneme recording & testing
poe record-cli

# Epic 2: Real-time temporal brain testing
poe temporal-test                    # Interactive temporal brain testing
poe temporal-test -m wavlm_ctc       # Use specific model (mlp_control, wav2vec2_ctc, wavlm_ctc)
poe temporal-test --list-models      # Show available models
poe temporal-test --list-devices     # Show audio devices
poe test-temporal                    # Run all temporal brain tests (47 tests)
```

## 🎯 Epic 1: Three-Way Model Comparison

### **1. MLP Control Workflow** (Baseline)

- **Fast training** with scikit-learn
- **Single phoneme classification**
- **Low memory usage**
- **Quick prototyping baseline**

```bash
poe train-mlp     # Train MLP classifier
poe test-pkl      # Test pickle model
```

### **2. CTC Wav2Vec2 Workflow** (Advanced)

- **Sequence modeling** with PyTorch + LSTM
- **Facebook's Wav2Vec2** speech representations
- **Alignment-free CTC training**
- **Temporal phoneme sequences**

```bash
poe train-ctc     # Train Wav2Vec2 CTC model
poe test-ctc      # Test CTC inference
```

### **2. CTC Wav2Vec2 Workflow** (BEST Performance) 🏆

- **Superior sequence modeling** with PyTorch + LSTM
- **Facebook's Wav2Vec2** proven speech representations
- **87.00% test accuracy achieved**
- **Empirically best phoneme recognition**

```bash
poe train-ctc           # Train Wav2Vec2 CTC model (BEST)
poe test-ctc            # Test best CTC inference
```

### **3. CTC WavLM Workflow** (Advanced Research) 🥈

- **Advanced sequence modeling** with PyTorch + LSTM
- **Microsoft's WavLM** cutting-edge speech representations
- **85.35% test accuracy achieved**
- **Research-focused advanced features**

```bash
poe train-wavlm-ctc     # Train WavLM CTC model (research)
poe test-wavlm-ctc      # Test advanced CTC inference
```

## 🧠 Epic 2: Temporal Brain CLI Testing

### **Real-Time Phoneme Stabilization System**

- **Temporal Brain Algorithms**: Smoothing, hysteresis control, confidence gating
- **Model Hot-Swapping**: Switch between MLP, Wav2Vec2 CTC, WavLM CTC in real-time
- **Performance Metrics**: <15% flicker rate target, <150ms latency
- **Test-Driven Development**: 47 comprehensive unit tests
- **ONNX Runtime Integration**: Compatible with all Epic 1 models

```bash
poe temporal-test                    # Start interactive temporal brain testing
poe temporal-test -m mlp_control     # Test with MLP Control model
poe temporal-test -m wav2vec2_ctc    # Test with Wav2Vec2 CTC model
poe temporal-test -m wavlm_ctc       # Test with WavLM CTC model (default)
poe temporal-tune                    # Auto-tune temporal brain parameters
poe temporal-baseline                # Baseline measurements
poe temporal-compare                 # Compare temporal vs non-temporal performance
poe test-temporal                    # Run all temporal brain unit tests
```

### **Temporal Brain Components**

| Component | Purpose | Algorithm |
|-----------|---------|----------|
| **Smoothing** | Noise reduction | Moving average & exponential smoothing |
| **Hysteresis Control** | Prevents flip-flopping | Dual-threshold lock/unlock with minimum duration |
| **Confidence Gating** | Reliability filtering | Persistence-based emission with phoneme thresholds |
| **Flicker Tracker** | Performance monitoring | Real-time flicker rate measurement (<15% target) |
| **Audio Capture** | Real-time processing | 16kHz mono audio with callback architecture |

## 🛠️ Development Commands

### Training & Testing - Epic 1 Three-Way Comparison

```bash
poe train-all       # Train all three models (MLP + Wav2Vec2 CTC + WavLM CTC)
poe test-all        # Test all trained models for comparison
poe train-ctc-all   # Train both CTC models (Wav2Vec2 + WavLM)
poe test-ctc-all    # Test both CTC models for comparison
poe full-pipeline   # Complete pipeline: train all + test all + export models
```

### Development Tools

```bash
poe lint          # Code quality check
poe format        # Format code (black + isort)
poe clean         # Clean build artifacts
poe info          # Show project overview
poe workflows     # List all available commands

# Epic 2: Temporal Brain Development
poe test-temporal                    # Run all temporal brain unit tests
poe temporal-tune                    # Parameter tuning utility
poe temporal-baseline                # Baseline performance measurement
poe temporal-compare                 # Compare temporal vs non-temporal performance
```

### Debugging - All Three Workflows

```bash
poe debug-shared      # Test shared utilities
poe debug-mlp         # Test MLP workflow
poe debug-ctc         # Test Wav2Vec2 CTC model structure
poe debug-wavlm-ctc   # Test WavLM CTC model structure (new!)
```

## 📊 Model Outputs

The system generates multiple deployment formats:

- **`.pkl`** - Scikit-learn models (fast Python inference)
- **`.pt`** - PyTorch traced models (advanced features)
- **`.onnx`** - ONNX models (Unreal Engine, web deployment)
- **`phoneme_labels.json`** - Label mappings

## 🏗️ Architecture

### Epic 1: Three-Workflow Architecture

```text
phoneme-classifier/
├── workflows/
│   ├── mlp_control_workflow/    # Traditional MLP baseline
│   ├── ctc_w2v2_workflow/       # Wav2Vec2 CTC approach  
│   ├── ctc_wavlm_workflow/      # WavLM CTC approach (85.35% accuracy)
│   └── shared/                  # Shared utilities across all workflows
├── inference/                   # Epic 2: Temporal brain real-time inference
│   ├── temporal_brain/          # Core temporal brain algorithms
│   └── cli/                     # CLI testing tools
├── configs/                     # Configuration files
├── tests/                       # Comprehensive test suite (47 tests)
├── recordings/                  # Training data (phoneme audio)
├── dist/                       # Generated models & outputs (all three)
└── logs/                       # Training logs (comprehensive)
```

### Epic 1: Three-Way Data Flow

```text
**Epic 1: Training Flow**
Audio Recordings → Feature Extraction → ML Training → Model Export
     ↓                    ↓                ↓             ↓
  37 phonemes    Wav2Vec2/WavLM Features   MLP/CTC   .pkl/.pt/.onnx
                        ↓                   ↓
                 768-dim vectors    [MLP: single phoneme]
                                   [CTC: sequence modeling]
                                   
Performance: MLP (79.73% baseline) < WavLM CTC (85.35% good) < Wav2Vec2 CTC (87.00% BEST)

**Epic 2: Real-Time Inference Flow**
Live Audio → Audio Capture → ONNX Inference → Temporal Brain Pipeline → Stable Phoneme
    ↓              ↓              ↓               ↓                        ↓
  16kHz mono   Callback Queue   Raw Probabilities   Smoothing → Hysteresis → Confidence Gating
                                                                    ↓
                                                            <15% Flicker Rate Target
```

## 🎮 Project Epics

This project follows a structured epic-based development approach:

1. **[✅ Live Phoneme CTCs](https://www.notion.so/Epic-Live-Phoneme-CTCs-22b502b4855680149d70eec42adf84d3?pvs=21)** - **COMPLETED!** Three-way model comparison achieved with 87.00% Wav2Vec2 CTC best performance
2. **[✅ Live Streaming Improvements (Temporal Brain)](https://www.notion.so/Epic-Live-Streaming-Improvements-Temporal-Brain-22b502b48556801c86f0f3f5a7036010?pvs=21)** - **IMPLEMENTED** Real-time temporal brain CLI testing tool with advanced stabilization algorithms
3. **[Whisper Teacher & Distillation](https://www.notion.so/Epic-Whisper-Teacher-Distillation-22b502b4855680da8047e51acd13ef1e?pvs=21)** - Model distillation
4. **[Multi-Model Bake-off Harness](https://www.notion.so/Epic-Multi-Model-Bake-off-Harness-22b502b485568092ab79fe7ec3901b36?pvs=21)** - Benchmarking
5. **[Game for Data Collection](https://www.notion.so/Epic-Game-for-Data-Collection-22b502b4855680dfa7d6f0c8ea071806?pvs=21)** - Interactive data collection
6. **[Offline Rescoring & Parent Summaries](https://www.notion.so/Epic-Offline-Rescoring-Parent-Summaries-22b502b485568051b73efd500dd632f3?pvs=21)** - Analysis tools
7. **[Personalization & Practice Packs](https://www.notion.so/Epic-Personalization-Practice-Packs-22b502b48556807a9d2ade60a605d358?pvs=21)** - Adaptive learning
8. **[Data Enrichment & Noise Robustness](https://www.notion.so/Epic-Data-Enrichment-Noise-Robustness-22b502b485568027a789c927a85a096b?pvs=21)** - Data augmentation
9. **[Evaluation & Progress Gates](https://www.notion.so/Epic-Evaluation-Progress-Gates-22b502b4855680dcb4f3e071691c4957?pvs=21)** - Quality assurance
10. **[Model Update & Export Pipeline](https://www.notion.so/Epic-Model-Update-Export-Pipeline-22b502b485568049af1fe48dcff0d011?pvs=21)** - Deployment automation

**Current Status**: 
- Epic 1 (Live Phoneme CTCs) - ✅ **COMPLETED** - Three-way model comparison successfully implemented. Wav2Vec2 CTC achieved 87.00% accuracy (best performer). ONNX export pipeline operational.
- Epic 2 (Temporal Brain) - ✅ **IMPLEMENTED** - Real-time temporal brain CLI testing tool implemented with TDD approach. 47 unit tests passing, <15% flicker rate achieved.

## 🎯 Key Features - Epic 1 & 2 Achievement

### Epic 1: Live Phoneme CTCs
- **🔥 Three ML Approaches**: MLP baseline (79.73%) + WavLM CTC (85.35%) + Wav2Vec2 CTC (87.00% best!)
- **🎵 Advanced Audio Processing**: Complete Wav2Vec2/WavLM → embeddings → classification pipeline
- **🏆 Performance Comparison**: Comprehensive three-way model benchmarking and analysis
- **🎮 Game Integration**: ONNX export for deployment of Unreal Engine, browser, and mobile games
- **📊 Rich Visualization**: UMAP plots, confusion matrices, performance metrics for all workflows

### Epic 2: Temporal Brain
- **🧠 Real-Time Stabilization**: Smoothing, hysteresis control, confidence gating algorithms
- **🔄 Model Hot-Swapping**: Switch between all Epic 1 models in real-time without restart
- **⚡ Performance Targets**: <15% flicker rate, <150ms latency achieved
- **🧪 Interactive Testing**: Real-time phoneme testing with live audio capture
- **✅ Test-Driven Development**: 47 comprehensive unit tests covering all components
- **🎯 ONNX Integration**: Compatible with all Epic 1 ONNX models for deployment
- **🔄 Cross-Platform**: WSL/Linux training → Windows deployment pipeline

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
