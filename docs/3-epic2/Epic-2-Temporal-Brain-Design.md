---
title: Epic 2 - Temporal Brain Phase 1 (CLI Testing Tool)
version: 1.0
date_created: 2025-01-26
epic: 2
status: in_design
tags: [temporal-brain, cli-tool, phoneme-stabilization, local-inference]
---

# Epic 2: Temporal Brain Phase 1 (CLI Testing Tool)

## Overview

Epic 2 builds upon the successful three-way model comparison system from Epic 1 by adding a temporal stabilization layer that eliminates flicker and provides stable real-time phoneme detection. This epic focuses specifically on **Phase 1: CLI Testing Tool** - a Python-based development and testing environment for temporal brain algorithms.

## Epic Goals

**Primary Goal**: Implement temporal stabilization CLI testing tool to achieve <15% flicker rate while maintaining <150ms latency for real-time phoneme detection.

**Secondary Goals**:
- Enable swappable models from Epic 1 (MLP Control, Wav2Vec2 CTC, WavLM CTC)
- Provide testing tools for family voice validation
- Create foundational temporal brain algorithms for future platform deployment
- Establish parameter tuning methodology and baseline measurements

## Architecture Decision: Local Processing Only

```
Audio Input → Local ONNX Model → Temporal Brain → Stable Phoneme Events
```

**Rationale**: Network latency makes cloud processing impossible for <150ms targets:
- **Network approach**: 150-450ms total ❌
- **Local approach**: 70-100ms total ✅

## Core Components

### 1. Smoothing Algorithm
- **Purpose**: Eliminate jittery frame-by-frame changes
- **Implementation**: Sliding window with configurable averaging
- **Target**: Reduce rapid oscillations in phoneme predictions

### 2. Hysteresis Control  
- **Purpose**: Prevent rapid flip-flopping between phonemes
- **Implementation**: Lock/unlock logic with confidence thresholds
- **Target**: Maintain stable phoneme output during transitions

### 3. Confidence Gating
- **Purpose**: Only emit high-confidence, stable phonemes
- **Implementation**: Per-phoneme threshold tuning with temporal persistence
- **Target**: Ensure reliable phoneme events with timestamps

## Success Metrics

- **Flicker Rate**: <15% in kid voice tests
- **Latency**: <150ms end-to-end (game-dependent)
- **Frustration Index**: ≤15% (fails twice on same target)
- **Phoneme Accuracy**: Maintain Epic 1 performance levels
- **Model Swapping**: Seamless switching between MLP/Wav2Vec2/WavLM

## CLI Testing Tool Implementation

**Platform**: Python with local ONNX inference  
**Purpose**: Algorithm development and family voice testing

**Key Features**:
- Real-time phoneme detection with temporal stabilization
- Model swapping (MLP/Wav2Vec2/WavLM from Epic 1)
- Parameter tuning interface
- Flicker rate measurement
- Baseline comparison with Epic 1 models
- Family voice sample recording and optimization

**Technology Stack**:
- `onnxruntime` for model inference
- `sounddevice` for audio capture
- `numpy` for signal processing
- Custom temporal brain implementation

**CLI Commands**:
```bash
poe temporal-test      # Interactive phoneme testing
poe temporal-tune      # Auto-tune parameters using voice samples
poe temporal-baseline  # Measure baseline flicker from Epic 1 models
poe temporal-compare   # Compare all 3 models with temporal brain
```

## Project Structure

```
fast-api-phoneme-python/
├── workflows/                    # Epic 1: Training & model preparation
├── inference/                    # 🆕 Epic 2: Local inference & temporal brain
│   ├── temporal_brain/          # Core temporal processing (Python)
│   │   ├── __init__.py
│   │   ├── smoothing.py         # Sliding window algorithms
│   │   ├── hysteresis.py        # Lock/unlock logic
│   │   ├── confidence_gating.py # Threshold management
│   │   └── temporal_processor.py # Main temporal brain class
│   └── cli/                     # Epic 2: CLI testing tool
│       ├── phoneme_tester.py    # Main CLI application
│       ├── model_loader.py      # ONNX model loading
│       ├── audio_capture.py     # Microphone capture
│       └── flicker_analyzer.py  # Flicker measurement
├── tests/temporal_brain/        # Testing infrastructure
├── configs/                     # Temporal brain configurations
│   ├── temporal_config.json
│   └── phoneme_thresholds.json
└── docs/
    ├── Epic-2-Temporal-Brain-Design.md    # This document
    ├── Voice-Controlled-Platform.md       # Phase 2 architecture spec
    └── temporal-brain-algorithms.md       # Algorithm specifications
```

## Model Integration Strategy

### Epic 1 Model Compatibility
All three models from Epic 1 are supported:
- **MLP Control**: `classifier.pkl` → ONNX export → Temporal brain
- **Wav2Vec2 CTC**: Direct ONNX → Temporal brain  
- **WavLM CTC**: Direct ONNX → Temporal brain (best performer: 85.35% accuracy)

### Swappable Architecture
```python
# Python CLI example
temporal_brain = TemporalBrain(config_path="temporal_config.json")
temporal_brain.load_model("wavlm_ctc.onnx", "phoneme_labels.json")
result = temporal_brain.process_audio(audio_data)
```


## Testing Strategy

### Family Voice Testing
- **Participants**: Developer, children, extended family
- **Test Scenarios**: Specific phoneme pronunciation challenges
- **Metrics**: Flicker rate, accuracy, frustration index
- **Data Collection**: Voice samples for parameter optimization

### Automated Testing
- **Baseline Measurement**: Epic 1 models without temporal brain
- **Algorithm Validation**: Synthetic audio with known ground truth  
- **Performance Testing**: Latency and memory usage monitoring
- **Cross-Platform**: Consistency across Python/JS/C++ implementations

### Acceptance Criteria
- [ ] <15% flicker rate in family voice tests
- [ ] Model swapping works seamlessly in CLI tool
- [ ] Latency remains <150ms for CLI implementation
- [ ] Temporal brain maintains Epic 1 accuracy levels
- [ ] Parameter tuning produces measurable flicker reduction

## Risk Mitigation

### Technical Risks
- **Latency Impact**: Monitor processing overhead, optimize algorithms
- **Accuracy Degradation**: Validate temporal brain doesn't hurt core accuracy
- **Platform Inconsistency**: Share configuration files and test data

### Implementation Risks  
- **Algorithm Complexity**: Start simple (moving averages) before advanced techniques
- **Parameter Tuning**: Use automated optimization rather than manual tuning
- **Family Voice Variability**: Account for diverse speech patterns in testing

## Size Estimation

**Scope**: Medium - Single platform (CLI) with core temporal brain algorithms
**Risk**: Medium - New algorithm development with parameter tuning complexity  
**Impact**: High - Foundation for all future temporal brain implementations

## Next Steps

1. **CLI Tool Implementation**: Create inference/ directory structure and CLI interface
2. **Algorithm Research**: Evaluate smoothing techniques for phoneme data
3. **Baseline Measurement**: Quantify current flicker rates from Epic 1
4. **Family Testing Setup**: Prepare voice recording and testing protocols
5. **Configuration Design**: Define parameter ranges and tuning methodology

## Integration with Epic 1

This epic builds directly on Epic 1's achievements:
- ✅ **Leverages**: Trained ONNX models from all three workflows
- ✅ **Extends**: Adds temporal stabilization layer
- ✅ **Maintains**: Model comparison capability
- ✅ **Enhances**: Real-time usability for interactive applications

Epic 2 establishes the foundational temporal brain algorithms and CLI testing framework that will enable future browser game and Unreal Engine implementations in subsequent epics.