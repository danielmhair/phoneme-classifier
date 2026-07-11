# Project Documentation

This directory contains all project documentation organized by epic and category.

## 📁 Directory Structure

> **2026-07-10 note**: The accuracy figures throughout this docs/ tree (87.00% / 85.35% / 79.73%) were found to come from evaluations with data leakage and, for the CTC models, broken decoding. See the [Evaluation Foundation PRD](../plans/prds/07-10-2026-PRD-models-trustworthy.md) and [docs/codebase-map.md](codebase-map.md) for the honest leave-one-speaker-out numbers and what was fixed. Files under `1-initial/`, `2-epic1/`, and `3-epic2/` are largely historical snapshots from before this was discovered - read them as "what was believed/built at the time," not current fact. `NEXT_STEPS_GUIDE.md` was deleted (2026-07-10) since its entire roadmap was built on the debunked numbers.

### 📋 1-initial/ - Initial Project Documentation
Early project documentation, architecture analysis, and foundational design work.

**Key Documents:**
- `DOCUMENTATION_AUDIT_SUMMARY.md` - Documentation audit results
- `2-architecture-analysis.md` - Overall project architecture analysis
- `1-commands-list.md` - Available project commands reference

### 📊 2-epic1/ - Live Phoneme CTCs (implementation complete, accuracy superseded - see note above)
Epic 1 focused on implementing a three-way model comparison system enabling comprehensive analysis of MLP, Wav2Vec2 CTC, and WavLM CTC approaches.

**Key Documents:**
- `EPIC_1_COMPLETION_SUMMARY.md` - Complete Epic 1 achievement summary
- `WAVLM_CTC_SYSTEM_DESIGN.md` - WavLM CTC implementation design
- `MODEL_PERFORMANCE_COMPARISON.md` - Three-way model comparison results
- `EPIC1_CTC_ANALYSIS.md` - CTC implementation analysis
- `WAVLM_CTC_WORKFLOW_GUIDE.md` - WavLM workflow implementation guide

**Epic 1 Achievements:**
- ✅ MLP Control: Baseline traditional classifier
- ✅ Wav2Vec2 CTC: Facebook's speech representation with CTC
- ✅ WavLM CTC: Microsoft's advanced speech representation
- ✅ Model Comparison: Three-way validation system
- ✅ ONNX Export: Deployment-ready models for all platforms

### 🧠 3-epic2/ - Temporal Brain Phase 1 (CLI Testing Tool) (IN DESIGN)
Epic 2 focuses on implementing the foundational temporal brain CLI testing tool with temporal stabilization algorithms.

**Key Documents:**
- `Epic-2-Temporal-Brain-Design.md` - Epic 2 CLI testing tool design and implementation plan
- `temporal-brain-algorithms.md` - Detailed algorithm specifications with code examples
- `Voice-Controlled-Platform.md` - Future browser game architecture specification

**Epic 2 Goals:**
- 🎯 <15% flicker rate in CLI testing
- 🎯 <150ms end-to-end latency maintenance
- 🎯 Model-swappable architecture using Epic 1 ONNX models
- 🎯 Family voice testing and parameter optimization

### 🌐 6-epic5/ - Temporal Brain Browser Game (PLANNED)
Epic 5 implements a voice-controlled browser-based platformer game using temporal brain algorithms from Epic 2.

**Key Documents:**
- `Epic-5-Browser-Game-Design.md` - Complete browser game design and architecture

**Epic 5 Goals:**
- 🎯 React + Phaser.js voice-controlled platformer
- 🎯 Client-side ONNX.js inference with temporal brain
- 🎯 <500ms voice recognition latency
- 🎯 Progressive Web Application deployment

### 🎮 7-epic6/ - Temporal Brain Unreal Engine (PLANNED)
Epic 6 implements production-ready Unreal Engine integration with native C++ temporal brain implementation.

**Key Documents:**
- `Epic-6-Unreal-Engine-Design.md` - Complete Unreal Engine plugin design and implementation

**Epic 6 Goals:**
- 🎯 Native C++ temporal brain with <50ms latency
- 🎯 ONNX Runtime C++ integration
- 🎯 Blueprint nodes for game designers
- 🎯 Cross-platform deployment (Windows/Mac/Linux)

### 🗂️ notion/ - Notion Epic Exports (re-ranked 2026-07-11)

Local snapshots of the Notion epic pages (Notion is the source of truth). Each `Epic - *.md` now carries a dated status banner re-ranking it against the honest LOSO results:

- **Promoted**: Game for Data Collection (target-age data is the measured bottleneck), Personalization & Practice Packs (the Chloe-vs-others gap is speaker mismatch - exactly what per-child adaptation attacks)
- **Largely delivered early**: Multi-Model Bake-off Harness (built 2026-07-10 as `evaluation/harness/`; latency/ABX/CI remain), plus the model-eval half of Evaluation & Progress Gates
- **Premise spike before committing**: Whisper Teacher & Distillation (Whisper is word-level adult-speech ASR; isolated child phonemes are its weakest input - spike-test on ~50 clips first)
- **Unchanged/downstream**: Unreal Engine, Offline Rescoring, Model Update & Export Pipeline, Data Enrichment (augmentation pilot in flight)

`Vision/` is an identical duplicate set, kept in sync. The same status banners were added to the live Notion pages on 2026-07-11, so local exports and Notion agree.

## 🎯 Epic Roadmap Overview

| Epic | Status | Goal | Achievement |
|------|--------|------|-------------|
| **Epic 1** | Implemented, accuracy superseded | Three-way model comparison | See [PRD](../plans/prds/07-10-2026-PRD-models-trustworthy.md) for honest LOSO numbers |
| **Epic 2** | 🏗️ **IMPLEMENTED** | Temporal Brain CLI Tool | <15% flicker rate target; re-tune pending against fixed CTC decoding |
| **Epic 3** | ⚠️ **PREMISE SPIKE FIRST** | Whisper Teacher & Distillation | Verify Whisper can label isolated child phonemes before committing |
| **Epic 4** | ✅ **LARGELY DELIVERED** (2026-07-10) | Multi-Model Bake-off Harness | `evaluation/harness/` LOSO + unified interface; latency/ABX/CI remain |
| **Epic 5** | 📋 **PLANNED** | Temporal Brain Browser Game | Voice-controlled platformer |
| **Epic 6** | 📋 **PLANNED** | Temporal Brain Unreal Engine | Production game integration |

## 🚀 Epic Implementation Sequence

> **2026-07-10**: This sequence (Epic 2 → Epic 3 Whisper → Epic 4 Bake-off Harness) was superseded. Research and the [Evaluation Foundation PRD](../plans/prds/07-10-2026-PRD-models-trustworthy.md) found that tuning Epic 2's temporal brain, or moving to Epic 3/4, on top of unverified Epic 1 accuracy numbers would mean building on a signal nobody had actually confirmed works. The evaluation/bake-off harness work (much of what Epic 4 describes) was pulled forward and completed first - see `evaluation/harness/`. Epic 3/4/5/6 are still genuinely future work, just not necessarily in this original order.

### Completed out of order: Evaluation Foundation (LOSO harness, CTC decode fix)
**What Epic 4's "Multi-Model Bake-off Harness" described, built first because Epic 1's numbers needed verifying before anything else could be trusted**
- Leave-one-speaker-out evaluation across all 3 models (`evaluation/harness/`)
- Fixed CTC decoding, WavLM embedding extraction bug, ONNX export silent-failure
- See [plans/prds/07-10-2026-PRD-models-trustworthy.md](../plans/prds/07-10-2026-PRD-models-trustworthy.md)

### Epic 2 (Temporal Brain CLI Tool) - implemented, not yet re-validated against honest Epic 1 numbers
**CLI Testing Tool Development**
- Real-time phoneme detection with temporal brain
- Family voice testing and parameter optimization
- Baseline flicker measurement from Epic 1 models
- Foundational algorithms for future platforms

### Next (re-ranked 2026-07-11 against LOSO results)
**In order of measured leverage:**
1. **Augmentation pilot** (in flight) - does production-style noise/speed/pitch augmentation close any of the Chloe gap? (`evaluation/harness/augmentation.py`, Chloe fold, run pending on the more powerful machine)
2. **Game for Data Collection** (promoted) - the binding constraint is target-age-band data; Chloe's 78-81% known-target top-3 rate means a target-aware game is likely playable enough to collect data today
3. **Personalization / per-child calibration** (promoted) - the ~9-point held-out-average-vs-Chloe gap is speaker mismatch, exactly what few-shot adaptation attacks
4. **Whisper spike** (cheap, parallel) - before Epic 3 becomes an epic, verify Whisper can usefully label ~50 isolated child-phoneme clips at all; if not, its premise collapses

**Epic 3 (Whisper Teacher & Distillation) - demoted pending that spike:**
- Whisper ASR teacher for diverse speech labeling
- CTC student model distillation for on-device use
- Multi-accent and noisy environment training
- Automated model improvement pipeline

### Future: Epic 5 (Temporal Brain Browser Game)
**Voice-Controlled Web Game**
- React + Phaser.js game engine
- Client-side ONNX.js inference with temporal brain
- Voice-controlled platformer mechanics
- Progressive web application deployment

### Final: Epic 6 (Temporal Brain Unreal Engine)
**Production Game Integration**
- Native C++ temporal brain implementation
- ONNX Runtime C++ integration
- Blueprint nodes and cross-platform deployment
- Production-ready game performance

## 📚 Documentation Standards

- **Epic-specific**: Documents focused on single epic implementation
- **Cross-epic**: General architecture, project setup, and shared utilities
- **Living docs**: Updated as implementation progresses
- **Code examples**: Concrete implementation guidance for developers
- **Size estimates**: Based on scope, risk, and impact rather than timelines

## 🔗 Quick Navigation

- 🎯 **Evaluation Foundation (start here)**: [../plans/prds/07-10-2026-PRD-models-trustworthy.md](../plans/prds/07-10-2026-PRD-models-trustworthy.md)
- 🗺️ **Codebase Map**: [codebase-map.md](codebase-map.md)
- 📊 **Epic 1 Summary** (accuracy figures superseded, see note above): [2-epic1/EPIC_1_COMPLETION_SUMMARY.md](2-epic1/EPIC_1_COMPLETION_SUMMARY.md)
- 🧠 **Epic 2 CLI Design**: [3-epic2/Epic-2-Temporal-Brain-Design.md](3-epic2/Epic-2-Temporal-Brain-Design.md)
- ⚡ **Temporal Brain Algorithms**: [3-epic2/temporal-brain-algorithms.md](3-epic2/temporal-brain-algorithms.md)
- 🌐 **Epic 5 Browser Game**: [6-epic5/Epic-5-Browser-Game-Design.md](6-epic5/Epic-5-Browser-Game-Design.md)
- 🎮 **Epic 6 Unreal Engine**: [7-epic6/Epic-6-Unreal-Engine-Design.md](7-epic6/Epic-6-Unreal-Engine-Design.md)
- 📋 **Commands**: [1-initial/1-commands-list.md](1-initial/1-commands-list.md)