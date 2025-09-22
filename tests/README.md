# Model Validation Testing Framework

This directory contains the testing framework for validating MLP and CTC models using clean test recordings separate from training data.

## Quick Start

1. **Create test recordings** (see Recording Instructions below)
2. **Run validation tests**:
   ```bash
   # Run all model validations
   poetry run python tests/test_model_validation.py
   
   # Or add to pyproject.toml tasks
   poe test-models  # (after adding task)
   ```

## Directory Structure

```
tests/
├── test_recordings/           # Clean test audio files (separate from training)
│   ├── mlp/                  # Single phoneme tests for MLP
│   │   ├── AE_test_001.wav   # Target: AE phoneme
│   │   ├── B_test_001.wav    # Target: B phoneme  
│   │   └── ...               # One per phoneme to test
│   └── ctc/                  # Sequence tests for CTC
│       ├── BAT_sequence_001.wav  # Target: B-AE-T sequence
│       ├── CAT_sequence_001.wav  # Target: K-AE-T sequence
│       └── ...               # Phoneme sequence recordings
├── validation_results/        # Saved test results (auto-created)
│   ├── validation_mlp_control_20241203_143052.json
│   └── validation_wavlm_ctc_20241203_143105.json
├── test_model_validation.py   # Main validation framework
└── README.md                 # This file
```

## Recording Instructions

### For MLP Testing (Single Phonemes)
Record clear single phoneme sounds:

```bash
# Create directory
mkdir -p tests/test_recordings/mlp

# Record phonemes (use your existing recording setup)
# Name format: {PHONEME}_test_{NUMBER}.wav
# Examples:
#   AE_test_001.wav  (say "ae" sound clearly)
#   B_test_001.wav   (say "buh" sound clearly) 
#   K_test_001.wav   (say "kuh" sound clearly)
```

**Suggested MLP test phonemes**:
- AE, B, K, T, D, F, G, P, S (basic sounds)
- SH, CH, TH (complex sounds)
- IY, OW, UW (vowels)

### For CTC Testing (Phoneme Sequences)  
Record phoneme sequences/words:

```bash
# Create directory
mkdir -p tests/test_recordings/ctc

# Record sequences (speak clearly with slight pauses between phonemes)
# Name format: {SEQUENCE}_sequence_{NUMBER}.wav
# Examples:
#   BAT_sequence_001.wav  (say "B...AE...T" blended)
#   CAT_sequence_001.wav  (say "K...AE...T" blended)
```

**Suggested CTC test sequences**:
- BAT, CAT, DOG, PIG (simple 3-phoneme words)
- FISH, SHIP (4+ phonemes with complex sounds)

## What the Tests Check

### MLP Validation
- ✅ Model loads correctly
- ✅ Single phoneme recognition accuracy  
- ✅ Confidence scores are reasonable (not 1.0/0.0)
- ✅ Feature extraction works properly
- ✅ Real-time inference performance

### CTC Validation  
- ✅ Model loads correctly
- ✅ Sequence processing (basic)
- 🚨 **KNOWN ISSUE**: CTC binary outputs (1.0/0.0) instead of probabilities
- 🚨 **KNOWN ISSUE**: CTC decoding not implemented (sequences not properly evaluated)
- ✅ Feature extraction for temporal models

## Understanding Results

### Accuracy Metrics
- **High accuracy (>90%)**: Model working well on clean audio
- **Medium accuracy (70-90%)**: Model working but may need improvement
- **Low accuracy (<70%)**: Model has significant issues

### Confidence Scores
- **Normal**: Values between 0.3-0.9 with variety
- **Binary Issue**: Only 1.0 and 0.0 values (indicates broken probability calculation)
- **Low Confidence**: All values <0.5 (model uncertain)

### Critical Issues to Fix
1. **CTC Binary Outputs**: CTC models returning 1.0/0.0 instead of probability distributions
2. **CTC Sequence Decoding**: Need proper CTC decoding for sequence evaluation  
3. **Data Leakage**: Ensure test recordings are completely separate from training

## Using Results for Iteration

1. **Save Results**: All results auto-saved to `validation_results/` as JSON
2. **Compare Over Time**: Track accuracy improvements after model changes
3. **Debug Issues**: Use detailed per-phoneme results to identify problem areas
4. **Fast Iteration**: Rerun tests quickly without re-recording audio

## Next Steps

1. **Record Test Audio**: Create clean test recordings following naming conventions
2. **Run Initial Tests**: Identify current model performance and issues
3. **Fix CTC Issues**: Address binary output and sequence decoding problems
4. **Implement as Poetry Task**: Add to pyproject.toml for easy execution
5. **Set Up CI**: Automate testing when models change

## Poetry Task Integration

Add to `pyproject.toml`:

```toml
[tool.poe.tasks]
test-models = "python tests/test_model_validation.py"
test-mlp = "python -c \"from tests.test_model_validation import ModelValidator; v=ModelValidator(); v.setup_models(); result=v.validate_mlp_model(); v.save_validation_results(result); print(f'MLP: {result.accuracy:.1%} accuracy')\""
test-ctc = "python -c \"from tests.test_model_validation import ModelValidator; v=ModelValidator(); v.setup_models(); result=v.validate_ctc_model(); v.save_validation_results(result); print(f'CTC: {result.accuracy:.1%} accuracy')\""
```

Then run:
```bash
poe test-models     # All models
poe test-mlp        # MLP only  
poe test-ctc        # CTC only
```