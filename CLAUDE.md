# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a phoneme classification system that trains AI models to classify speech phonemes from children's voices. The project features:

- **Epic 1: Live Phoneme CTCs** - Three-way model comparison system (MLP, Wav2Vec2 CTC, WavLM CTC) ✅ COMPLETED
- **Epic 2: Temporal Brain** - Real-time phoneme stabilization with CLI testing tool - ✅ COMPLETED

The project uses **Poetry** for dependency management and **poethepoetry (poe)** for task execution.

## ALWAYS REMEMBER

Note, complete only happens when the user states its complete. Most tasks and epics involve human in the loop. Do not forget this

Also, you can't say something is complete, UNLESS all mocked pieces are NOT mocked, unless the user specifies.

## Quick Start

### Initial Setup

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies and CUDA support
poe setup

# Or step by step:
poetry install                    # Install dependencies
poetry add --group dev poethepoet    # Install poe task runner
poe setup-cuda                   # Install PyTorch with CUDA
```

### Training Workflows - Epic 1: Three-Way Model Comparison

```bash
# Epic 1: Train all three models for comparison
poe train-all           # Complete three-way comparison (MLP + Wav2Vec2 CTC + WavLM CTC)

# Individual workflows:
poe train-mlp           # MLP Control (traditional classifier baseline)
poe train-ctc           # Wav2Vec2 CTC (Facebook's speech model)
poe train-wavlm-ctc     # WavLM CTC (Microsoft's advanced speech model)

# CTC-only comparison:
poe train-ctc-all       # Both CTC workflows (Wav2Vec2 + WavLM)
```

### Model Testing - Epic 1: Three-Way Validation

```bash
# Test all three models for comparison
poe test-all            # Complete three-way testing (PKL + ONNX + Wav2Vec2 CTC + WavLM CTC)

# Individual model testing:
poe test-pkl            # MLP Control: Test pickle model (fast inference)
poe test-onnx           # MLP Control: Test ONNX model (deployment format)
poe test-ctc            # Wav2Vec2 CTC: Test sequence model
poe test-wavlm-ctc      # WavLM CTC: Test advanced sequence model

# CTC-specific testing:
poe test-ctc-all        # Both CTC models (Wav2Vec2 + WavLM)

# Model comparison and deployment:
poe compare-models      # Compare all three models (performance metrics)
./classify_voice_onnx.sh  # Deploy all three ONNX models (updated script)

# Interactive testing (all models):  
poe record-cli          # Record and classify phonemes interactively

# Epic 2: Temporal Brain CLI Testing
poe temporal-test       # Interactive temporal brain testing with model hot-swapping
poe temporal-tune       # Parameter tuning for temporal algorithms
poe temporal-baseline   # Baseline measurements without temporal brain
poe temporal-compare    # Compare temporal vs non-temporal performance
poe test-temporal       # Run all temporal brain unit tests
```

### Development Tools

```bash
# Code quality
poe lint          # Lint entire project
poe format        # Format code with black + isort
poe clean         # Clean build artifacts

# Debugging
poe debug-shared      # Test shared utilities
poe debug-mlp         # Test MLP imports
poe debug-ctc         # Test Wav2Vec2 CTC model structure
poe debug-wavlm-ctc   # Test WavLM CTC model structure

# Information
poe info          # Show project overview
poe workflows     # List all available tasks

# Epic 2: Temporal Brain CLI Tools
poe temporal-test --list-models    # List available models
poe temporal-test --list-devices   # List audio devices
poe temporal-test -m wavlm_ctc     # Use specific model (mlp_control, wav2vec2_ctc, wavlm_ctc)
```

### Development Workflows

```bash
# Quick development cycle
poe dev           # Setup + train MLP + test

# Complete pipeline
poe full-pipeline # Train all three + test all + export models

# Quality check
poe format && poe lint && poe test-all
```

## Legacy Commands (Still Supported)

For backwards compatibility, the original commands still work:

```bash
# Traditional approach (deprecated)
source .venv/bin/activate
pip install -r requirements.txt
python workflows/mlp_control_workflow/0_workflow.py

# Modern approach (recommended)
poe train-mlp
```

## Architecture Overview

### Epic 1: Three-Workflow Architecture

All three workflows (`workflows/mlp_control_workflow/0_workflow.py`, `workflows/ctc_w2v2_workflow/0_workflow.py`, and `workflows/ctc_wavlm_workflow/0_workflow.py`) use a shared execution framework from `workflows/shared/workflow_executor.py` for consistent step-by-step execution, timing, and error handling.

**Epic 1 Achievement**: Enables comprehensive model comparison with standardized training and evaluation across MLP, Wav2Vec2 CTC, and WavLM CTC approaches.

### MLP Control Workflow Pipeline

The MLP workflow (`workflows/mlp_control_workflow/0_workflow.py`) orchestrates 13 sequential steps:

1. **Data Cleanup** - Cleans previous dist/ outputs
2. **Data Preparation** - Organizes recordings from multiple sources and applies audio augmentation
3. **Embedding Extraction** - Uses Wav2Vec2 to extract embeddings from organized phoneme recordings
4. **Classifier Training** - Trains MLP classifier on embeddings with label encoding
5. **Visualization** - Generates UMAP plots and confusion matrices
6. **Analysis** - Analyzes confusion pairs and batch tests
7. **Benchmarking** - Performance testing and model saving
8. **Model Export** - Traces PyTorch models and exports to ONNX
9. **Unreal Integration** - Copies models to Unreal Engine project

### CTC Wav2Vec2 Workflow Pipeline

The CTC workflow (`workflows/ctc_w2v2_workflow/0_workflow.py`) orchestrates 5 sequential steps:

1. **Cleanup previous CTC outputs** - Cleans previous dist/ outputs
2. **Prepare audio dataset for CTC** - Organizes recordings and applies augmentation
3. **Extract temporal embeddings** - Preserves sequence information for CTC training
4. **Train CTC classifier** - Trains sequence-to-sequence CTC model
5. **Test CTC inference system** - Validates CTC model functionality

### CTC WavLM Workflow Pipeline

The WavLM CTC workflow (`workflows/ctc_wavlm_workflow/0_workflow.py`) orchestrates 10 comprehensive steps:

1. **Cleanup previous WavLM CTC outputs** - Cleans previous dist/ outputs
2. **Prepare audio dataset for WavLM CTC** - Organizes recordings and applies augmentation
3. **Extract WavLM temporal embeddings** - Uses Microsoft's WavLM for superior speech representations
4. **Train WavLM CTC classifier** - Trains advanced sequence-to-sequence CTC model
5. **Visualize WavLM CTC Results** - Generates UMAP plots and confusion matrices
6. **Analyze WavLM CTC confusion pairs** - Detailed error analysis and confusion patterns
7. **Batch test WavLM CTC phonemes** - Comprehensive model validation
8. **Export WavLM CTC to ONNX** - Deployment-ready model export
9. **Test WavLM CTC ONNX model** - ONNX functionality validation
10. **Test WavLM CTC inference system** - Production readiness validation

**Performance Achievement**: 85.35% accuracy on test set with 88.61% peak validation accuracy.

### Key Components

- **Audio Processing**: Uses `soundfile`, `sounddevice`, `librosa` for audio I/O and processing
- **Feature Extraction**: 
  - Wav2Vec2 (`facebook/wav2vec2-base`) for MLP and Wav2Vec2 CTC workflows
  - WavLM (`microsoft/wavlm-base`) for advanced WavLM CTC workflow
- **Classification**: 
  - scikit-learn MLPClassifier for traditional phoneme prediction
  - PyTorch CTC models for sequence-to-sequence classification
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

### Directory Structure - Epic 1 & Epic 2 System

**Epic 1: Training Workflows**
- `workflows/mlp_control_workflow/` - MLP baseline pipeline scripts and `0_workflow.py`
- `workflows/mlp_control_workflow/utils/` - Audio processing utilities
- `workflows/mlp_control_workflow/validations/` - MLP testing and validation scripts
- `workflows/ctc_w2v2_workflow/` - Wav2Vec2 CTC pipeline scripts and `0_workflow.py`
- `workflows/ctc_w2v2_workflow/validations/` - Wav2Vec2 CTC testing and validation scripts
- `workflows/ctc_wavlm_workflow/` - WavLM CTC pipeline scripts and `0_workflow.py`
- `workflows/ctc_wavlm_workflow/validations/` - WavLM CTC testing and validation scripts
- `workflows/shared/` - Shared workflow execution utilities for all three workflows

**Epic 2: Temporal Brain CLI**
- `inference/temporal_brain/` - Core temporal brain algorithms (smoothing, hysteresis, confidence gating)
- `inference/cli/` - CLI testing tools for real-time phoneme testing
- `configs/temporal_config.json` - Configuration for temporal brain parameters
- `tests/` - Comprehensive test suite with 47 tests covering all algorithms

**Data & Outputs**
- `recordings/` - Source audio data (phoneme recordings by speaker)
- `dist/` - Generated outputs (models, embeddings, visualizations)
- `logs/` - Workflow execution logs with timestamps for all workflows

## Epic 1: Live Phoneme CTCs - COMPLETED! 🎯

**Achievement**: Successfully implemented three-way model comparison system enabling comprehensive analysis of different phoneme classification approaches.

### Model Comparison Results

| Model | Type | Architecture | Performance | Training Time | Key Features |
|-------|------|-------------|-------------|---------------|--------------|
| **MLP Control** | Traditional | scikit-learn MLP | Baseline | ~Fast | Single phoneme prediction |
| **Wav2Vec2 CTC** | Sequence | PyTorch CTC + Wav2Vec2 | Good | ~Medium | Facebook's speech model |
| **WavLM CTC** | Advanced Sequence | PyTorch CTC + WavLM | **85.35%** | ~23 min | Microsoft's superior speech model |

### Epic 1 Components Status
- ✅ **MLP Control**: Baseline traditional classifier
- ✅ **Wav2Vec2 CTC**: Facebook's speech representation with CTC
- ✅ **WavLM CTC**: Microsoft's advanced speech representation with CTC  
- ✅ **Model Comparison**: Three-way validation and testing system
- ✅ **Unified API**: Consistent training and inference across all models

### Three-Way Comparison Commands
```bash
# Train all models for Epic 1 comparison
poe train-all

# Test all models 
poe test-all

# Individual model testing for comparison
poe test-pkl         # MLP Control
poe test-ctc         # Wav2Vec2 CTC  
poe test-wavlm-ctc   # WavLM CTC (best performance)
```

## Epic 2: Temporal Brain - IMPLEMENTED! 🧠

**Achievement**: Successfully implemented real-time phoneme stabilization system with CLI testing tool using Test-Driven Development approach. **REAL ONNX MODEL INFERENCE** now fully functional, replacing mock probabilities for meaningful model comparison.

### Temporal Brain Features

| Component | Purpose | Key Features |
|-----------|---------|-------------|
| **Smoothing** | Noise reduction | Moving average & exponential smoothing algorithms |
| **Hysteresis Control** | Prevents flip-flopping | Dual-threshold lock/unlock mechanism with minimum duration |
| **Confidence Gating** | Reliability filtering | Persistence-based emission with phoneme-specific thresholds |
| **Flicker Tracker** | Performance metrics | Real-time flicker rate measurement (<15% target) |
| **Temporal Processor** | Main orchestrator | Complete pipeline integration with configuration management |
| **Real Model Inference** | ONNX Integration | Two-stage MLP pipeline (Wav2Vec2 → MLP) with real inference replacing mocks |

### Key Epic 2 Achievement: Real Model Inference

❌ **Before**: Mock probabilities defeated Epic 2's purpose of model comparison  
✅ **After**: Real ONNX inference enables meaningful temporal brain testing

```python
# OLD: Mock probabilities (meaningless for model comparison)
def _create_mock_probabilities(self) -> np.ndarray:
    return np.random.dirichlet(np.ones(len(self.labels)))

# NEW: Real ONNX model inference (meaningful comparison)  
def _process_audio_frame(self, audio_data: np.ndarray):
    feature_result = self.feature_extractor.extract_features_for_model(...)
    real_probs = self.model_loader.run_inference(feature_result['features'])
    result = self.temporal_processor.process_frame(real_probs)
```

### Epic 2 Components Status
- ✅ **Core Algorithms**: All temporal brain algorithms implemented with TDD
- ✅ **CLI Testing Tool**: Interactive phoneme testing with model hot-swapping
- ✅ **ONNX Model Loading**: Compatible with all Epic 1 models (MLP, Wav2Vec2 CTC, WavLM CTC)
- ✅ **Real-time Audio**: Live audio capture and processing pipeline
- ✅ **Configuration System**: JSON-based parameter management
- ✅ **Comprehensive Tests**: 47 unit tests covering all components
- ✅ **Performance Metrics**: Flicker tracking and stability measurement

### Epic 2 CLI Commands
```bash
# Interactive temporal brain testing
poe temporal-test                    # Default (WavLM CTC model)
poe temporal-test -m mlp_control     # Use MLP Control model
poe temporal-test -m wav2vec2_ctc    # Use Wav2Vec2 CTC model
poe temporal-test --list-models      # Show available models
poe temporal-test --list-devices     # Show audio devices

# Testing and validation
poe test-temporal                    # Run all temporal brain tests (47 tests)
poe temporal-baseline                # Baseline performance measurement
poe temporal-compare                 # Compare temporal vs non-temporal
poe temporal-tune                    # Parameter tuning utility

# Custom configuration
poe temporal-test -c configs/custom_temporal_config.json
```

### Temporal Brain Architecture

```text
Raw Audio → Audio Capture → Mock/Real Inference → Temporal Processor Pipeline:
                                                   ↓
                                     1. Smoothing (noise reduction)
                                                   ↓
                                     2. Hysteresis (stability control)
                                                   ↓
                                     3. Confidence Gating (reliability)
                                                   ↓
                                     4. Flicker Tracking (metrics)
                                                   ↓
                                           Stable Phoneme
```

### Performance Targets (Epic 2)
- **Flicker Rate**: <15% (current system achieves target)
- **Latency**: <150ms real-time processing
- **Stability**: High-confidence phoneme emission only
- **Model Support**: All Epic 1 ONNX models (hot-swappable)
- **Test Coverage**: 47 comprehensive unit tests

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

## MCPs for you to use frequently
- Serena MCP - provides essential semantic code retrieval and editing tools that are akin to an IDE's capabilities, extracting code entities at the symbol level and exploiting relational structure.
- Context7 MCP - for up-to-date, version-specific documentation for any library or framework
- Sequentialthinking MCP - for breaking down complex problems into manageable steps and maintaining context across interactions.

## Other important things
- Always use poetry run to execute scripts, even for python and pip so we know its running in the poetry environment.