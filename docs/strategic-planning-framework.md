# Light Haven Phoneme Foundation: Strategic Planning Framework

> **2026-07-10 update**: This document correctly identified the data-leakage problem below and called for "Foundation First: No game development until core models work reliably in real conditions." That work is now done - see the [Evaluation Foundation PRD](../plans/prds/07-10-2026-PRD-models-trustworthy.md) for the honest leave-one-speaker-out numbers, the LOSO evaluation harness (`evaluation/harness/`), and the CTC decode/label-mismatch/ONNX-export bugs that were fixed along the way. The rest of this document's analysis and reasoning still holds; only the "not yet done" framing below is now out of date.

## Executive Summary

This document defines the strategic framework for building Light Haven's educational phoneme classification system. Based on divine guidance to focus on the foundational phoneme classifier and "think small" with initial implementations, this plan maps game types to model requirements and establishes a clear development sequence.

## Current Reality Check

**Critical Issue Identified**: Game Type 1 (MLP Control) claims 85% accuracy, but **real microphone performance doesn't match evaluation results** due to:
1. **Data leakage** in current evaluation (testing on training data)
2. **Inappropriate CTC evaluation** (testing sequence models on single phonemes)
3. **Lack of proper iterative testing framework** for model improvement

## The "Foundation" Definition

The phoneme foundation is **not just one model** - it's an adaptive intelligence system comprising:

### Core Components
1. **Multi-Model Inference Engine**
   - MLP Control: Single phoneme classification (immediate feedback)
   - CTC Models: Sequence recognition (phoneme blending)
   - Whisper Teacher-Student: Accent/noise robustness
   - Temporal Brain: Stability layer for ALL models

2. **Educational Intelligence**
   - Bloom's taxonomy progression tracking
   - Adaptive difficulty management
   - Assessment and confidence scoring
   - Learning analytics and progress visualization

3. **Data Collection & Improvement Pipeline**
   - Real-time gameplay data collection
   - Active learning for model improvement
   - Expert validation workflows
   - Privacy-preserving data pipelines

## Game Types → Model Architecture Mapping

### Game Type 1: Direct Voice-to-Action (typing.com equivalent)
**Educational Goal**: Knowledge/Remember level - immediate phoneme recognition  
**Model Requirements**: MLP Control + Temporal Brain  
**Technical Specifications**:
- <150ms latency for real-time responsiveness
- Single phoneme classification
- High stability (>85% consistent recognition)
- Browser deployment via ONNX.js

**Current Status**: Claims 85% accuracy but **fails in real microphone testing** - needs data leakage fix and proper validation

### Game Type 2: Progressive Difficulty Platformers
**Educational Goal**: Comprehension/Application - pattern recognition and skill building  
**Model Requirements**: Intelligent model routing based on difficulty level  
**Technical Specifications**:
- Adaptive difficulty engine
- Progress tracking across phoneme complexity
- Mixed model usage (MLP for basics, CTC for advanced)

### Game Type 3: Sequence Recognition Games
**Educational Goal**: Application/Analysis - phoneme blending to words  
**Model Requirements**: CTC Models + proper sequence decoding  
**Technical Specifications**:
- Multi-phoneme sequence processing
- Blend detection and word formation
- Temporal sequence alignment
- Support for variable-length inputs

**Current Status**: Current evaluation tests CTC on single phonemes which **doesn't make sense for sequence models**

### Game Type 4: Assessment & Practice Games
**Educational Goal**: Evaluation/Synthesis - pronunciation accuracy and improvement  
**Model Requirements**: All models + confidence scoring  
**Technical Specifications**:
- Multi-model consensus scoring
- Progress analytics and improvement tracking
- Expert validation integration
- Detailed feedback systems

## Data Collection Through Gamification Strategy

### Core Concept
Each game simultaneously:
1. **Educates children** in phoneme recognition/production
2. **Collects diverse speech data** from different accents/regions
3. **Improves models** through active learning algorithms

### Data Pipeline Architecture
1. **Real-time Confidence Scoring**: Flag uncertain predictions during gameplay
2. **Speaker Profiling**: Automatic accent/demographic classification
3. **Active Learning**: Identify which phonemes need more data from specific groups
4. **Expert Validation**: Route flagged samples to phoneticians
5. **Model Updates**: Continuous improvement without retraining from scratch

### Parent Engagement Layer
**What Parents CAN Validate**:
- Child engagement and motivation levels
- Progress over time (before/after improvements)
- System responsiveness and game behavior
- Educational value and learning outcomes

**Parent Game Concepts**:
- Progress visualization dashboards
- Celebration of child milestones
- Family pronunciation challenges
- Learning outcome insights

## Development Sequence (Following Divine Guidance)

### Phase 1: CLI Foundation Validation ⚡ **CURRENT PRIORITY**
**Goal**: Prove each model component works reliably through proper testing

**Immediate Actions Required**:
1. **Fix Data Leakage**: Create proper train/test split evaluation
2. **Build Proper CTC Testing**: Create sequence-based testing CLI (not single phonemes)
3. **Create Interactive Testing Game**: Simple CLI game for children to test both MLP and CTC
4. **Build Unit Test Framework**: Automated tests for continuous iteration and validation

### Phase 2: Minimal Browser Game 🎯 *"Think Small"*
**Goal**: Voice-controlled typing.com equivalent  
**Requirements**: 
- MLP Control + Temporal Brain only
- Single phoneme → character action
- Immediate visual feedback
- Test with your children for engagement validation

### Phase 3: Data Collection Integration
**Goal**: Turn game into data collection instrument
**Add**: Confidence scoring, analytics, active learning algorithms

### Phase 4: Multi-Game Ecosystem
**Goal**: All 4 game types with appropriate model selection
**Expand**: Progressive difficulty, sequences, assessment games

### Phase 5: Adaptive Intelligence
**Goal**: Models improve continuously from gameplay data
**Add**: Expert validation workflows, model update pipelines

## Bloom's Taxonomy Integration

The system must support all levels of cognitive development:

1. **Remember** (Knowledge): Single phoneme recognition with immediate feedback
2. **Understand** (Comprehension): Pattern recognition across phonemes with confidence scoring
3. **Apply** (Application): Blending phonemes into words using sequence models
4. **Analyze** (Analysis): Speech pattern assessment with multi-model analysis
5. **Evaluate** (Synthesis): Accuracy self-assessment with meta-cognitive feedback
6. **Create** (Evaluation): Project-based phonemic creation with creative sandbox

## Immediate Next Steps (Week 1-2)

### Priority 1: Fix Foundation Testing Issues
1. **Resolve Data Leakage**
   - Extract proper 20% test split from original training
   - Re-evaluate all models on clean test data
   - Document actual real-world performance

2. **Build Proper CTC Testing CLI**
   - Create sequence-based testing (not single phonemes)
   - Test phoneme blending capabilities
   - Validate sequence-to-sequence performance

3. **Create Interactive CLI Game**
   - Simple voice-controlled game for children
   - Test both MLP and CTC in appropriate contexts
   - Allow rapid iteration and feedback collection

4. **Develop Unit Test Framework**
   - Automated testing for continuous model validation
   - Performance benchmarking across different scenarios
   - Regression testing for model improvements

### Priority 2: Real-World Validation
1. Test models with live microphone input
2. Validate temporal brain stability across all models
3. Measure actual latency and accuracy in real conditions
4. Document what works vs. what needs improvement

## Success Metrics

### Technical Metrics
- **Accuracy**: >90% on clean test data (no data leakage)
- **Latency**: <150ms for real-time games
- **Stability**: <15% flicker rate with temporal brain
- **Robustness**: Performance across different accents/speakers

### Educational Metrics
- **Engagement**: Children play voluntarily like typing.com
- **Learning**: Measurable phoneme improvement over time
- **Retention**: Long-term engagement and skill maintenance
- **Progression**: Clear advancement through Bloom's taxonomy levels

### Data Collection Metrics
- **Diversity**: Representative data across accents/demographics
- **Quality**: Expert validation of collected pronunciation data
- **Volume**: Sufficient data for continuous model improvement
- **Privacy**: Full compliance with children's privacy requirements

## Key Insights

1. **Foundation First**: No game development until core models work reliably in real conditions
2. **Proper Testing**: Each model type needs appropriate evaluation methodology
3. **Iterative Approach**: Build testing framework for continuous improvement
4. **Child-Centered**: All technical decisions must serve educational outcomes
5. **Data as Product**: Games collect data that improves the entire system

## Questions for Continued Planning

1. Which phoneme sequences should we test for CTC validation?
2. What specific voice interactions work best for children?
3. How do we balance model accuracy vs. game responsiveness?
4. What privacy frameworks do we need for child data collection?
5. How do we measure educational effectiveness vs. just technical performance?

---

**Next Actions**: Build proper testing framework to validate what actually works before investing in game development.