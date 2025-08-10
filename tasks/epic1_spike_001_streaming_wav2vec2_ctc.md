# Epic 1 Spike 001: Minimal Streaming Wav2Vec2 -> CTC Pipeline

Status: In Progress (initial scaffold implemented)
Date: 2025-08-09
Owner: (assign)

## Goal

Establish a minimal, end-to-end streaming inference loop for phoneme recognition using a wav2vec2 backbone plus a lightweight CTC head. This spike validates architectural assumptions, chunking strategy, and latency feasibility before investing in training, decoding sophistication (beam/LM), or endpointing.

## Scope (Included)

- Chunked streaming over a WAV file (simulated real-time) with overlap.
- Wav2Vec2 backbone (pretrained, not fine-tuned here).
- Randomly initialized Linear CTC head producing logits over (blank + phoneme vocab).
- Greedy CTC decoding (argmax, collapse repeats, drop blank) per chunk.
- Simple stabilization heuristic (emit symbol only after N consecutive chunk confirmations).
- Latency metrics per chunk and aggregate (avg, max, RTF proxy).
- Offline full-file greedy decode for rough coverage comparison.

## Out of Scope (Deferred)

- Training / fine-tuning the CTC head.
- Beam search / language model integration.
- Confidence scoring and smoothing across chunk boundaries.
- Accurate timestamp alignment (currently approximate by chunk end time).
- Silence / endpoint detection.
- Microphone live capture (placeholder only; file streaming used).
- Resampling (expects 16 kHz input in spike).

## Rationale

Starting with a lightweight spike de-risks real-time decoding by revealing: model throughput, chunk size viability, overlap requirements, and early structural changes needed. It avoids premature optimization in decoding or model training until basic latency and pipelining are understood.

## Key Metrics / Acceptance Criteria

1. Script runs end-to-end on a provided 16 kHz mono wav file. (Done)
2. Incremental phoneme symbols printed as they stabilize. (Done)
3. Latency metrics reported: avg per chunk, max per chunk, real-time factor (processing_time / audio_duration). (Done)
4. Offline reference greedy decode performed for qualitative comparison. (Done)
5. Checklist of next steps & TODOs captured for follow-up tasks. (Done)
6. Clear demarcation of non-production aspects (random head, no resampling). (Done)
7. Code isolated / additive (does not break existing workflows). (Done)
8. (Optional / Future) Add mic input mode toggle. (Pending)

## Implementation Summary

File: `w2v2-ctc_workflow/stream_wav2vec2_ctc_spike.py`

- Loads phoneme vocab (`tokenizer/phoneme_vocab.txt`) and prepends blank symbol.
- Uses pretrained `facebook/wav2vec2-base` for feature extraction.
- Adds randomly initialized linear layer -> logits (no training in spike).
- Chunking: 40 ms window, 20 ms stride (50% overlap). Configurable via CLI args.
- Stabilization: require N (default=2) consecutive chunk confirmations of same terminal symbol before emission.
- Metrics collected per chunk (processing time) aggregated into summary including a real-time factor approximation.
- Offline decode uses same greedy collapse for baseline.

## CLI Usage

```bash
python -m w2v2-ctc_workflow.stream_wav2vec2_ctc_spike --wav path/to/audio.wav \
  --phoneme_vocab tokenizer/phoneme_vocab.txt \
  --chunk_ms 40 --stride_ms 20 --stabilization_repeats 2
```

## Current Limitations / Risks

- Random CTC head means emitted symbols are not meaningful: only latency mechanics validated.
- Without trained head, coverage metric is noise; used purely to exercise comparison logic.
- No handling for long-form audio memory optimization (full forward pass per chunk, no state caching).
- Potential throughput improvements (batching, shorter context windows, quantization) not explored yet.
- Type ignores added for HF model typing quirks—should be revisited if stricter linting is desired.

## Next Step Candidates

1. Replace random head with trained linear CTC layer (fine-tune on existing phoneme dataset).  
2. Implement microphone streaming (`sounddevice` callback, ring buffer).  
3. Add resampling path (e.g., librosa or torchaudio) to enforce 16 kHz.  
4. Confidence scoring: track per-symbol average max-logit across stabilization window.  
5. Beam search decoder (HF `CTCDecoder` or torchaudio beam with optional LM) behind a flag.  
6. Timestamp refinement: accumulate frame alignment indices & compute midpoints.  
7. Endpoint detection (silence energy threshold or blank-run length).  
8. Sliding window state optimization (retain receptive field tail instead of reprocessing full chunk).  
9. Export ONNX / TorchScript variant for lower-latency inference.  
10. Add unit test to assert non-empty output & latency bounds (< real-time).  

## Task Checklist

- [x] Create spike script file
- [x] Load wav2vec2 backbone & initialize CTC head
- [x] Implement chunked overlapping audio iterator
- [x] Greedy per-chunk CTC collapse
- [x] Stabilization heuristic across chunks
- [x] Emit incremental phoneme outputs with naive timestamps
- [x] Collect latency metrics (avg, max, RTF proxy)
- [x] Perform offline full-file greedy decode
- [x] Coverage / comparison printout
- [x] Document limitations & next steps in task MD
- [ ] Add microphone live capture mode
- [ ] Add proper resampling for non-16kHz inputs
- [ ] Replace random head with trained / loaded weights
- [ ] Add confidence scoring & smoothing enhancements
- [ ] Implement beam search / LM option
- [ ] Endpoint (silence) detection
- [ ] Accurate timestamp assignment (frame-level)
- [ ] Sliding window state caching optimization
- [ ] Packaging / API wiring (future service integration)
- [ ] Basic unit test to ensure script returns non-empty sequence & latency < audio length

## Decision Log

- Chose 40 ms window / 20 ms stride (50% overlap) as a starting balance between latency and context for convolutional front-end; subject to tuning after real logits are available.
- Picked stabilization requirement of 2 consecutive confirmations to reduce flicker while keeping latency low; may require dynamic adjustment.
- Deferred beam decoding until meaningful logits exist to avoid premature optimization.

## Open Questions

- Final target latency & RTF thresholds (define acceptance target: e.g., RTF <= 0.25?).
- Phoneme inventory stability: will additional symbols (stress markers, silence) be needed in vocab?
- Integration path: direct WebSocket streaming vs. batching on server side?

---
Feel free to update status and promote remaining checklist items into individual tasks / issues under Epic 1.
