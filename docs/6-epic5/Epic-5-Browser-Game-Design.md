---
title: Epic 5 - Temporal Brain Browser Game (Voice-Controlled Platformer)
version: 1.0
date_created: 2025-01-26
epic: 5
status: planned
dependencies: [epic-2, epic-3, epic-4]
tags: [browser-game, temporal-brain, voice-controlled, react, phaser, onnx-js]
---

# Epic 5: Temporal Brain Browser Game (Voice-Controlled Platformer)

## Overview

Epic 5 implements a voice-controlled browser-based platformer game that leverages the temporal brain algorithms developed in Epic 2, enhanced models from Epic 3 (Whisper Teacher & Distillation), and validated through Epic 4 (Multi-Model Bake-off Harness). This creates an engaging web-based phoneme practice game with stable real-time voice recognition.

## Epic Goals

**Primary Goal**: Create a production-ready browser game with voice-controlled mechanics using temporal brain stabilization for <15% flicker rate and <500ms response latency.

**Secondary Goals**:
- Implement React + Phaser.js architecture with TypeScript
- Enable client-side ONNX.js model inference with temporal brain
- Provide engaging phoneme practice through gameplay
- Support progressive web application (PWA) deployment
- Enable model hot-swapping without game restart

## Architecture Alignment

**Based on**: `docs/3-epic2/Voice-Controlled-Platform.md` specification

**Technology Stack**:
- **Frontend**: React 18+ with TypeScript 5+
- **Game Engine**: Phaser 3.90+ with Matter.js physics
- **ML Inference**: ONNX.js with Web Workers for performance
- **Audio Processing**: Web Audio API with microphone capture
- **Temporal Brain**: JavaScript port from Epic 2 Python implementation
- **UI Components**: shadcn/ui for consistent design system

## Core Components

### 1. React-Phaser Bridge
```typescript
interface GameEvents {
  'score-update': (score: number) => void;
  'word-update': (word: string) => void;
  'game-state-update': (state: GameState) => void;
  'voice-command': (command: string) => void;
}

interface PhaserGameProps {
  onScoreUpdate: (score: number) => void;
  onWordUpdate: (word: string) => void;
  onGameStateUpdate: (state: GameState) => void;
  temporalBrain?: TemporalBrain | null;
}
```

### 2. Temporal Brain Integration
```typescript
interface TemporalBrainConfig {
  smoothingWindow: number;      // Default: 5 frames
  hysteresisThreshold: number;  // Default: 0.1
  confidenceGate: number;       // Default: 0.7
  lockDuration: number;         // Default: 100ms
  phonemeThresholds: Record<string, number>;
}

class TemporalBrain {
  constructor(config: TemporalBrainConfig);
  async loadModel(modelUrl: string, labelsUrl: string): Promise<void>;
  processAudio(audioData: Float32Array): Promise<TemporalBrainResult>;
  swapModel(modelUrl: string): Promise<void>;
  getFlickerRate(): number;
}
```

### 3. Voice Recognition Pipeline
```typescript
class VoiceRecognitionPipeline {
  private onnxSession: InferenceSession;
  private temporalBrain: TemporalBrain;
  private audioProcessor: AudioProcessor;
  
  async initialize(modelConfig: ModelConfig): Promise<void>;
  startListening(): Promise<void>;
  stopListening(): void;
  onPhonemeDetected: (phoneme: string, confidence: number) => void;
}
```

## Game Features

### Voice Commands
- **JUMP**: Character jumps vertically
- **RUN**: Character moves horizontally
- **CLIMB**: Character climbs platforms
- **DASH**: Character performs quick movement
- **FLIP**: Character performs acrobatic moves

### Game Mechanics
- **Platform Physics**: Matter.js-based collision detection
- **Collectibles**: Voice-activated gem collection
- **Progressive Difficulty**: Increasingly complex voice challenges
- **Real-time Feedback**: Visual indicators for voice recognition
- **Score System**: Points based on accuracy and speed
- **Practice Mode**: Isolated phoneme training exercises

### UI Components
```typescript
interface GameUI {
  VoiceIndicator: React.FC<{isListening: boolean; confidence: number}>;
  PhonemeDisplay: React.FC<{currentPhoneme: string; isStable: boolean}>;
  ScoreBoard: React.FC<{score: number; accuracy: number}>;
  ModelSelector: React.FC<{models: ModelInfo[]; onSelect: (id: string) => void}>;
  SettingsPanel: React.FC<{config: TemporalBrainConfig; onChange: (config) => void}>;
}
```

## Performance Requirements

### Game Performance
- **Frame Rate**: Maintain 60fps during active voice processing
- **Voice Latency**: <500ms from speech end to game action
- **Model Loading**: <10 seconds for ONNX model initialization
- **Memory Usage**: <100MB for core game components
- **Page Load**: <3 seconds on 3G connection

### Voice Recognition Performance
- **Flicker Rate**: <15% using Epic 2 temporal brain algorithms
- **Accuracy**: Maintain Epic 1/3/4 model performance levels
- **Confidence Gating**: Stable phoneme detection with configurable thresholds
- **Model Swapping**: Hot-swap between models without game restart

## Project Structure

```
fast-api-phoneme-python/
├── inference/
│   └── web_game/               # Epic 5: Browser game
│       ├── public/
│       │   ├── models/         # ONNX models and labels
│       │   └── assets/         # Game sprites and audio
│       ├── src/
│       │   ├── components/     # React UI components
│       │   │   ├── GameCanvas.tsx
│       │   │   ├── VoiceControls.tsx
│       │   │   ├── ModelSelector.tsx
│       │   │   └── SettingsPanel.tsx
│       │   ├── game/          # Phaser game implementation
│       │   │   ├── scenes/
│       │   │   │   ├── MainGameScene.ts
│       │   │   │   ├── MenuScene.ts
│       │   │   │   └── SettingsScene.ts
│       │   │   ├── entities/
│       │   │   │   ├── Player.ts
│       │   │   │   ├── Platform.ts
│       │   │   │   └── Collectible.ts
│       │   │   └── GameManager.ts
│       │   ├── voice/         # Voice recognition system
│       │   │   ├── TemporalBrain.ts    # JS port from Epic 2
│       │   │   ├── OnnxInference.ts    # ONNX.js integration
│       │   │   ├── AudioProcessor.ts   # Web Audio API
│       │   │   └── VoiceRecognition.ts # Main voice pipeline
│       │   ├── utils/         # Shared utilities
│       │   │   ├── ModelLoader.ts
│       │   │   ├── ConfigManager.ts
│       │   │   └── PerformanceMonitor.ts
│       │   └── App.tsx        # Main React application
│       ├── package.json
│       ├── tsconfig.json
│       ├── vite.config.ts
│       └── README.md
```

## Model Integration Strategy

### Epic Dependencies Integration
- **Epic 2**: Use temporal brain algorithms ported to JavaScript
- **Epic 3**: Load Whisper-distilled models for robust speech recognition
- **Epic 4**: Integrate validated models from bake-off harness testing

### Swappable Model Architecture
```typescript
interface ModelConfig {
  id: string;
  name: string;
  modelUrl: string;
  labelsUrl: string;
  description: string;
  performance: {
    accuracy: number;
    latency: number;
    size: number;
  };
}

class ModelManager {
  async loadModel(config: ModelConfig): Promise<void>;
  listAvailableModels(): ModelConfig[];
  getCurrentModel(): ModelConfig | null;
  swapModel(modelId: string): Promise<void>;
}
```

## Testing Strategy

### Automated Testing
- **Unit Tests**: Jest for component logic and utilities
- **Integration Tests**: React Testing Library for component behavior
- **E2E Tests**: Playwright for complete user workflows
- **Performance Tests**: Web Audio API mocks and voice simulation

### Voice Recognition Testing
- **Synthetic Audio**: Pre-recorded samples for consistent testing
- **Model Validation**: Cross-reference with Epic 4 bake-off results
- **Temporal Brain Validation**: Consistency with Epic 2 Python implementation
- **Cross-Browser Testing**: Chrome, Firefox, Safari, Edge compatibility

### Game Testing
- **Physics Simulation**: Matter.js collision detection validation
- **React-Phaser Bridge**: Event communication testing
- **Voice Command Execution**: Complete voice-to-action pipeline testing
- **Progressive Web App**: Installation and offline functionality

## Size Estimation

**Scope**: Large - Complete browser game with voice integration, React UI, Phaser game engine, and ONNX.js inference
**Risk**: High - Complex integration of multiple technologies, browser compatibility challenges, performance optimization
**Impact**: High - Primary user-facing application, foundation for phoneme learning games

## Acceptance Criteria

- [ ] Voice commands (JUMP, RUN, CLIMB, DASH, FLIP) execute reliably within 500ms
- [ ] Game maintains 60fps during active voice processing
- [ ] Temporal brain achieves <15% flicker rate in browser environment
- [ ] Model hot-swapping works without game restart
- [ ] Progressive Web App installs and works offline
- [ ] Cross-browser compatibility (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- [ ] Graceful fallback to keyboard controls when voice fails
- [ ] Real-time performance monitoring and adjustment

## Risk Mitigation

### Technical Risks
- **Browser Performance**: Use Web Workers for ONNX inference, optimize audio processing
- **Model Loading**: Implement progressive loading and caching strategies
- **Cross-Browser Compatibility**: Extensive testing and polyfill strategies

### User Experience Risks
- **Voice Recognition Failure**: Provide clear keyboard alternatives and error feedback
- **Microphone Permission**: Guide users through permission flow
- **Performance Variability**: Adaptive quality settings based on device capabilities

## Integration with Previous Epics

**Epic 1 Integration**: 
- ✅ Use ONNX models from three-way comparison
- ✅ Leverage phoneme label mappings

**Epic 2 Integration**:
- ✅ Port temporal brain algorithms to JavaScript
- ✅ Use parameter configurations and tuning results

**Epic 3 Integration**:
- ✅ Load Whisper-distilled models for robust recognition
- ✅ Benefit from multi-accent and noise robustness

**Epic 4 Integration**:
- ✅ Use validated models from bake-off harness
- ✅ Apply performance benchmarks and quality gates

Epic 5 represents the culmination of foundational work from Epics 1-4, creating an engaging user-facing application that demonstrates the full power of the phoneme recognition system in an interactive gaming context.