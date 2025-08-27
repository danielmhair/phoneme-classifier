---
title: Epic 2 - Live Streaming Improvements (Temporal Brain) Design
version: 1.0
date_created: 2025-01-26
epic: 2
status: in_design
tags: [temporal-brain, real-time, phoneme-stabilization, local-inference]
---

# Epic 2: Live Streaming Improvements (Temporal Brain)

## Overview

Epic 2 builds upon the successful three-way model comparison system from Epic 1 by adding a temporal stabilization layer that eliminates flicker and provides stable real-time phoneme detection. The "Temporal Brain" processes raw model outputs to deliver consistent, accurate phoneme events suitable for interactive applications.

## Epic Goals

**Primary Goal**: Implement temporal stabilization techniques to achieve <15% flicker rate while maintaining <150ms latency for real-time phoneme detection.

**Secondary Goals**:
- Enable swappable models from Epic 1 (MLP Control, Wav2Vec2 CTC, WavLM CTC)
- Provide testing tools for family voice validation
- Create deployable solutions for CLI, web, and Unreal Engine platforms

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

## Three-Phase Implementation

### Phase 1: CLI Testing Tool 🖥️
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

### Phase 2: Browser Game 🌐
**Platform**: Web browser with client-side processing  
**Purpose**: Engaging phoneme practice game with temporal brain

**Architecture Alignment**: Based on `docs/Voice-Controlled-Platform.md`
- **React**: UI components and state management
- **Phaser.js**: 2D game engine with WebGL rendering
- **TypeScript**: Strict typing for reliability
- **ONNX.js**: Client-side model inference
- **Web Audio API**: Microphone access and processing

**Temporal Brain Integration**:
```typescript
interface TemporalBrainConfig {
  smoothingWindow: number;      // Default: 5 frames
  hysteresisThreshold: number;  // Default: 0.1
  confidenceGate: number;       // Default: 0.7
  lockDuration: number;         // Default: 100ms
}

interface TemporalBrainResult {
  phoneme: string;
  confidence: number;
  isStable: boolean;
  timestamp: number;
  flickerRate: number;
}
```

**Game Features**:
- Voice-controlled platformer mechanics
- Real-time phoneme visualization
- Progress tracking and scoring
- Model selection interface (MLP/Wav2Vec2/WavLM)
- Temporal brain parameter adjustment
- Flicker rate monitoring

**Performance Requirements**:
- 60fps game rendering
- <500ms voice recognition latency
- Model hot-swapping without restart
- Graceful fallback to keyboard controls

### Phase 3: Unreal Engine Integration 🎮
**Platform**: Unreal Engine with C++ temporal brain  
**Purpose**: Production-ready game integration with optimal performance

**Key Features**:
- Native C++ temporal brain implementation
- ONNX Runtime C++ integration
- Blueprint nodes for game logic
- Model selection system with performance hints
- Cross-platform deployment (Windows/Mac/Linux)
- Real-time performance monitoring

**Architecture**:
```cpp
class UTemporalBrain : public UObject {
public:
    UFUNCTION(BlueprintCallable)
    FPhonemeResult ProcessAudio(const TArray<float>& AudioData);
    
    UFUNCTION(BlueprintCallable)
    void SwapModel(const FString& ModelPath);
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FTemporalBrainConfig Config;
};
```

**Performance Targets**:
- <50ms total latency
- Real-time audio processing at 60fps
- Memory usage <50MB for temporal brain
- Seamless model switching

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
│   ├── cli/                     # Phase 1: CLI testing tool
│   │   ├── phoneme_tester.py    # Main CLI application
│   │   ├── model_loader.py      # ONNX model loading
│   │   ├── audio_capture.py     # Microphone capture
│   │   └── flicker_analyzer.py  # Flicker measurement
│   ├── web_game/               # Phase 2: Browser game
│   │   ├── index.html
│   │   ├── js/
│   │   │   ├── temporal_brain.js    # JS port of temporal brain
│   │   │   ├── onnx_inference.js    # ONNX.js model loading
│   │   │   ├── phaser_game.js       # Phaser game implementation
│   │   │   └── react_components.js  # React UI components
│   │   └── css/
│   └── unreal_integration/     # Phase 3: C++ implementation
│       ├── cpp/
│       │   ├── TemporalBrain.h
│       │   ├── TemporalBrain.cpp
│       │   └── OnnxInference.cpp
│       └── blueprints/
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

```javascript
// Browser example  
const temporalBrain = new TemporalBrain(config);
await temporalBrain.loadModel('wavlm_ctc.onnx', 'phoneme_labels.json');
const result = temporalBrain.processAudio(audioData);
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
- [ ] Model swapping works seamlessly across all platforms
- [ ] Latency remains <150ms (CLI), <500ms (browser), <50ms (Unreal)
- [ ] Temporal brain maintains Epic 1 accuracy levels
- [ ] All three platforms produce consistent results

## Risk Mitigation

### Technical Risks
- **Latency Impact**: Monitor processing overhead, optimize algorithms
- **Accuracy Degradation**: Validate temporal brain doesn't hurt core accuracy
- **Platform Inconsistency**: Share configuration files and test data

### Implementation Risks  
- **Algorithm Complexity**: Start simple (moving averages) before advanced techniques
- **Parameter Tuning**: Use automated optimization rather than manual tuning
- **Cross-Platform Porting**: Implement comprehensive test suite for validation

## Next Steps

1. **Phase 1 Implementation**: Start with CLI tool development
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

Epic 2 transforms Epic 1's static model comparison into a dynamic, real-time system suitable for games and interactive learning applications.