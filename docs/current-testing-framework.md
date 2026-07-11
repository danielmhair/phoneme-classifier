# Current Testing Framework & Recording Strategy

## Overview

This document describes our current testing framework and recording strategy for validating the phoneme classifier models. This system was built to address critical issues with model evaluation and create a foundation for iterative improvement.

## Background & Problems Identified

### Original Issues
- **Data Leakage**: MLP showing 95.56% accuracy was inflated due to testing on training data
- **CTC Binary Output Issue**: CTC models returning 1.0/0.0 instead of probability distributions
- **Lack of Proper Validation**: No clean test recordings separate from training data
- **Manual Testing Burden**: No automated way to test models iteratively

### Strategic Decision
Following divine guidance to "think small" and build foundational tools before complex game development, we focused on:
1. Creating proper test/validation split
2. Building systematic recording collection system
3. Developing automated validation framework
4. Identifying and fixing model issues

## Current Testing Architecture

### 1. Automated Validation Framework
**File**: `tests/test_model_validation.py`

Core classes:
- `ModelValidator` - Main validation orchestrator
- `ValidationResults` - Results container with accuracy metrics
- `TestResult` - Individual test result tracking

**Features**:
- Tests both MLP and CTC models using pre-recorded files
- Detects binary output issues automatically
- Saves JSON results for iteration tracking
- Validates against 38 official phonemes from `phoneme_labels.json`

**Poetry Tasks**:
```bash
poe test-models           # Run comprehensive validation
poe test-mlp-validation   # Test only MLP Control model
poe test-ctc-validation   # Test only CTC models
```

### 2. Interactive Testing CLI
**File**: `tests/interactive_model_test.py`

**Features**:
- Manual real-time recording and testing
- Input validation against 38 valid phonemes
- Model hot-swapping (MLP, Wav2Vec2 CTC, WavLM CTC)
- Results saving for analysis
- WSL2-compatible audio handling

**Poetry Tasks**:
```bash
poe test-mlp-interactive  # Interactive MLP testing
poe test-ctc-interactive  # Interactive CTC testing
```

**Usage Examples**:
```bash
# Test specific phoneme/sequence
python tests/interactive_model_test.py mlp AE      # Test MLP with AE
python tests/interactive_model_test.py ctc BAT     # Test CTC with BAT

# Interactive mode
python tests/interactive_model_test.py mlp         # Interactive MLP
python tests/interactive_model_test.py ctc         # Interactive CTC
```

### 3. Guided Recording CLI
**File**: `tests/record_test_data.py`

**Features**:
- Systematic recording of all planned test sounds
- Automatic file saving with proper naming conventions
- Progress tracking through Phase 1 & 2
- WSL2-compatible audio device selection with fallbacks
- Session summary generation

**Poetry Task**:
```bash
poe record-test-data      # Guided recording session
```

## Recording Strategy

### Phase-Based Collection System

**Phase 1: Core Foundation (13 recordings)**
- **8 MLP Phonemes**: B, K, S, A_Æ, SH, T, M, TH
- **5 CTC Sequences**: BAT, CAT, SAT, BATH, FISH

**Phase 2: Extended Coverage (19 recordings)** 
- **12 MLP Phonemes**: E, I, U, OO_UW, F, V, D, G, P, Z, CH, J_DƷ
- **8 CTC Sequences**: MAT, SHIP, CHIP, ZIP, BET, BIT, BUT, BEACH

**Total**: 32 recordings across both phases

### Recording File Organization
```
tests/test_recordings/
├── mlp/                 # MLP phoneme recordings
│   ├── B_test_001.wav
│   ├── K_test_001.wav
│   └── ...
├── ctc/                 # CTC sequence recordings  
│   ├── BAT_sequence_001.wav
│   ├── CAT_sequence_001.wav
│   └── ...
└── recording_session_summary.json
```

### Audio Specifications
- **Sample Rate**: 16kHz (matches Epic 1 model requirements)
- **Duration**: 3 seconds per recording
- **Format**: WAV files with proper phoneme/sequence naming
- **Quality**: Clean recordings separate from training data

## WSL2 Audio Compatibility

### Issues & Solutions
- **Problem**: WSL2 audio device selection errors
- **Solution**: Multi-layer fallback system with device validation
- **Implementation**: AudioCaptureConfig with device auto-selection

### Device Selection Logic
1. Query available devices with retry logic
2. Prefer 'pulse' or 'default' devices
3. Validate user-specified devices
4. Fallback to system default (None) on failure
5. Return silent audio if all methods fail

## Phoneme Validation System

### 38 Official Phonemes
Based on `workflows/mlp_control_workflow/dist/phoneme_labels.json`:

**Vowels**: a_æ, a_ɑ, ai_eɪ, e, ee, i, i_aɪ, ir_ɝ, oa, oo_uw, oo_ʊ, ow_aʊ, oy_oɪ_ɔɪ, u

**Consonants**: b, ch, d, dh, f, g, h, j_dʒ, k, l, m, n, ng, p, s, sh, su_ʒ, t, th, v, w, y_j, z

### Input Validation
All testing tools validate input against this official list to prevent:
- Invalid phoneme testing
- Inconsistent naming conventions
- Model confusion from unsupported sounds

## Current Issues to Address

### 1. CTC Binary Output Problem ⚠️
**Status**: PENDING INVESTIGATION

**Symptoms**:
- CTC models returning 1.0 or 0.0 instead of probability distributions
- Makes sequence decoding impossible
- Detected automatically by validation framework

**Next Steps**:
- Investigate CTC model training process
- Check loss function implementation
- Validate CTC decoding pipeline
- Test with different sequence lengths

### 2. Data Leakage Resolution ⚠️
**Status**: PARTIALLY ADDRESSED

**Progress**:
- ✅ Created separate test recordings
- ✅ Built validation framework
- ⚠️ Need proper train/test split evaluation
- ⚠️ Need to measure real model performance

## Integration with Existing System

### Epic 1 Compatibility
- All testing tools work with existing MLP, Wav2Vec2 CTC, and WavLM CTC models
- Uses same feature extraction pipelines
- Compatible with existing ONNX export workflow

### Epic 2 Preparation
- Recording infrastructure ready for Temporal Brain testing
- Audio capture systems compatible with real-time processing
- Model loading framework supports hot-swapping

## Usage Workflows

### Complete Validation Workflow
```bash
# 1. Record clean test data
poe record-test-data

# 2. Run automated validation
poe test-models

# 3. Interactive testing for specific cases
poe test-mlp-interactive
poe test-ctc-interactive

# 4. Analyze results and iterate
# Check tests/test_recordings/ for saved results
```

### Iterative Development Workflow
```bash
# 1. Make model changes
# 2. Test quickly with saved recordings
poe test-models

# 3. Check for improvements
cat tests/test_recordings/validation_results.json

# 4. Interactive testing for edge cases
poe test-mlp-interactive
```

## Future Enhancements

### Short Term
1. Fix CTC binary output issue
2. Implement proper train/test split evaluation
3. Add more comprehensive error detection
4. Expand test recording collection

### Long Term
1. Automated performance regression detection
2. Cross-model comparison metrics
3. Integration with Temporal Brain validation
4. Continuous integration testing pipeline

## Technical Notes

### Dependencies
- `soundfile` & `sounddevice` for audio I/O
- `numpy` for audio processing
- `json` for results serialization
- `pathlib` for cross-platform file handling

### Configuration
- Audio settings match Epic 1 requirements (16kHz mono)
- WSL2 compatibility built-in
- Poetry task integration for easy CLI access

### Error Handling
- Graceful audio device failure recovery
- Comprehensive error logging
- Silent audio fallbacks for automation

## Conclusion

This testing framework provides a solid foundation for:
1. **Proper Model Validation**: Clean test data separate from training
2. **Issue Detection**: Automatic identification of binary output problems
3. **Iterative Development**: Fast testing without manual intervention
4. **Systematic Data Collection**: Organized recording strategy across phoneme categories

The system is ready for immediate use and addresses the core validation issues that were preventing accurate model assessment. Next steps focus on fixing the identified CTC issues and expanding the validation coverage.