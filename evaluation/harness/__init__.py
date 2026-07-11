"""
Leave-one-speaker-out (LOSO) evaluation harness.

Single interface for training and evaluating all three Epic 1 models
(MLP Control, Wav2Vec2 CTC, WavLM CTC) on the same honest, speaker-independent
metric. See plans/prds/07-10-2026-PRD-models-trustworthy.md for the design
rationale (why LOSO, why greedy CTC decode, why the known-target margin
metric, why augmentation/GOP/SpeechOcean762 are explicitly out of scope here).
"""
