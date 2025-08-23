# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a phoneme classification system that trains AI models to classify speech phonemes from children's voices. The project uses **Poetry** for dependency management and **poethepoetry (poe)** for task execution.

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

### Training Workflows

```bash
# MLP Control Workflow (traditional classifier)
poe train-mlp

# CTC Wav2Vec2 Workflow (sequence modeling)
poe train-ctc

# Run both workflows
poe train-all
```

### Model Testing

```bash
# Test trained models
poe test-pkl      # Test pickle model (fast)
poe test-onnx     # Test ONNX model (deployment)
poe test-ctc      # Test CTC model (sequences)
poe test-all      # Run all tests

# Interactive testing
poe record-cli    # Record and classify phonemes interactively
```

### Development Tools

```bash
# Code quality
poe lint          # Lint entire project
poe format        # Format code with black + isort
poe clean         # Clean build artifacts

# Debugging
poe debug-shared  # Test shared utilities
poe debug-mlp     # Test MLP imports
poe debug-ctc     # Test CTC model structure

# Information
poe info          # Show project overview
poe workflows     # List all available tasks
```

### Development Workflows

```bash
# Quick development cycle
poe dev           # Setup + train MLP + test

# Complete pipeline
poe full-pipeline # Train both + test all + export models

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

### Workflow Structure

Both workflows (`workflows/mlp_control_workflow/0_workflow.py` and `workflows/ctc_w2v2_workflow/0_workflow.py`) use a shared execution framework from `workflows/shared/workflow_executor.py` for consistent step-by-step execution, timing, and error handling.

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

- `workflows/mlp_control_workflow/` - MLP pipeline scripts (s0-s12) and `0_workflow.py`
- `workflows/mlp_control_workflow/utils/` - Audio processing utilities
- `workflows/mlp_control_workflow/validations/` - Testing and validation scripts
- `workflows/ctc_w2v2_workflow/` - CTC pipeline scripts and `0_workflow.py`
- `workflows/ctc_w2v2_workflow/validations/` - CTC testing and validation scripts
- `workflows/shared/` - Shared workflow execution utilities
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

## MCPs for you to use frequently
- Serena MCP - provides essential semantic code retrieval and editing tools that are akin to an IDE's capabilities, extracting code entities at the symbol level and exploiting relational structure.
- Context7 MCP - for up-to-date, version-specific documentation for any library or framework
- Sequentialthinking MCP - for breaking down complex problems into manageable steps and maintaining context across interactions.

## Other important things
- Always use poetry run to execute scripts, even for python and pip so we know its running in the poetry environment.