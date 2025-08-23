# CTC Wav2Vec2 Workflow

## Epic 1: Live Phoneme CTCs Implementation

This directory contains the complete CTC (Connectionist Temporal Classification) implementation for phoneme sequence recognition, separate from the main MLP workflow.

## 🎯 Overview

The CTC workflow provides alignment-free phoneme sequence recognition using:

- **Wav2Vec2** for speech feature extraction
- **LSTM** for temporal sequence modeling  
- **CTC Loss** for alignment-free training
- **Beam Search** for improved inference

## 📁 Directory Structure

```text
ctc_w2v2_workflow/
├── models/
│   └── ctc_model.py           # CTC model architecture
├── validations/
│   └── classify_voice_ctc.py  # CTC inference & testing
├── ctc_workflow.py            # Main workflow script
├── s2_extract_embeddings_temporal.py  # Temporal feature extraction
├── s3_ctc_classifier.py       # CTC training pipeline
├── test_ctc_model.py          # Model tests
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ctc_w2v2_workflow
pip install -r requirements.txt
```

### 2. Run Complete CTC Workflow

```bash
python ctc_workflow.py
```

### 3. Test CTC Inference

```bash
# Interactive mode
python validations/classify_voice_ctc.py

# File mode  
python validations/classify_voice_ctc.py --file path/to/audio.wav
```

## 🔧 Individual Steps

### Extract Temporal Embeddings

```bash
python s2_extract_embeddings_temporal.py
```

### Train CTC Model

```bash  
python s3_ctc_classifier.py
```

### Run Model Tests

```bash
python test_ctc_model.py
```

## 📊 Architecture Comparison

| Feature | MLP Workflow | CTC Workflow |
|---------|--------------|--------------|
| Input Processing | Mean pooling | Temporal sequences |
| Output | Single phoneme | Phoneme sequences |
| Alignment | Required | Alignment-free |
| Temporal Info | Lost | Preserved |
| Model | sklearn MLP | PyTorch LSTM+CTC |

## 🎭 Dependencies & Fallbacks

### Required Dependencies

- `torch` - Deep learning framework
- `transformers` - Wav2Vec2 model
- `soundfile` - Audio I/O
- `numpy`, `pandas` - Data processing

### Graceful Fallbacks

When dependencies are missing:

- Mock classifiers provide random predictions
- Training steps are skipped with warnings
- Clear installation instructions provided

## 🔬 Testing & Validation

### Run All Tests

```bash
python test_ctc_model.py
```

### Compare with MLP

```bash
# MLP (existing)
cd ../mlp_control_workflow
python validations/classify_voice_pkl.py

# CTC (this workflow)
cd ../ctc_w2v2_workflow  
python validations/classify_voice_ctc.py
```

## 📈 Key Features

- **Alignment-free**: No need for explicit phoneme-audio alignment
- **Sequence modeling**: Handles variable-length phoneme sequences
- **Real-time inference**: Interactive CLI for live testing
- **Temporal preservation**: Maintains time-based speech information
- **Robust training**: Validation, checkpointing, early stopping
- **Beam search decoding**: Improved inference quality

## 🔮 Next Steps

1. **Train with real data**: Use your phoneme recordings
2. **Performance evaluation**: Compare CTC vs MLP accuracy
3. **Sequence data collection**: Gather multi-phoneme training samples
4. **ONNX export**: Add CTC model export for deployment
5. **Streaming inference**: Implement real-time streaming

## 🐛 Troubleshooting

### Import Errors

```bash
# Install missing dependencies
pip install torch transformers soundfile

# Check Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### CUDA Issues

```bash
# Install CPU-only PyTorch
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Or GPU version
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Audio Errors

```bash
# Install audio dependencies
pip install sounddevice librosa audiomentations
```

## 📞 Support

For issues or questions about the CTC implementation:

1. Check error messages for missing dependencies
2. Verify file paths and data availability  
3. Review logs in `./logs/ctc_workflow_log_*.txt`
4. Test with mock mode first (no dependencies required)

---

**🎉 Epic 1: Live Phoneme CTCs - CTC Implementation Complete!**
