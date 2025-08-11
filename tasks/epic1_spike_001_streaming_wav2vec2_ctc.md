# Epic 1 Spike 001: Minimal Streaming Wav2Vec2 -> CTC Pipeline

Status: In Progress (initial scaffold implemented; first runtime observations added)
Date: 2025-08-09 (updated 2025-08-11)
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
- JSON or plain-text phoneme vocabulary loading (supports `dist/phoneme_labels.json`).

## Out of Scope (Deferred)

- Training / fine-tuning the CTC head.
- Beam search / language model integration.
- Confidence scoring and smoothing across chunk boundaries.
- Accurate timestamp alignment (currently approximate by chunk end time).
- Silence / endpoint detection.
- Microphone live capture (placeholder only; file streaming used).
- Resampling (expects 16 kHz input in spike).
- Warmup separation & steady-state latency reporting (planned improvement).

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
9. Support JSON phoneme vocab file (Done)

## Implementation Summary

File: `w2v2-ctc_workflow/stream_wav2vec2_ctc_spike.py`

- Loads phoneme vocab (plain text OR JSON). Example full vocab file: `dist/phoneme_labels.json` (37 symbols) automatically handled.
- Prepends blank symbol at index 0.
- Uses pretrained `facebook/wav2vec2-base` for feature extraction.
- Adds randomly initialized linear layer -> logits (no training in spike).
- Chunking: 40 ms window, 20 ms stride (50% overlap). Configurable via CLI args.
- Stabilization: require N (default=2) consecutive chunk confirmations of same terminal symbol before emission.
- Metrics collected per chunk (processing time) aggregated into summary including a real-time factor approximation.
- Offline decode uses same greedy collapse for baseline.

## CLI Usage

```bash
# Using text vocab (original)
python -m w2v2-ctc_workflow.stream_wav2vec2_ctc_spike \
  --wav path/to/audio.wav \
  --phoneme_vocab tokenizer/phoneme_vocab.txt \
  --chunk_ms 40 --stride_ms 20 --stabilization_repeats 2

# Using JSON vocab (preferred full set)
python -m w2v2-ctc_workflow.stream_wav2vec2_ctc_spike \
  --wav path/to/audio.wav \
  --phoneme_vocab dist/phoneme_labels.json
```

Batch helper:
```bash
python -m w2v2-ctc_workflow.run_spike_on_directory \
  --dir dist_last/organized_recordings/h --limit 1 \
  --ignore-aug --log-dir logs/spike_runs \
  --phoneme_vocab dist/phoneme_labels.json
```

## First Run Observations (2025-08-11)

- A short 0.39 s clip produced 19 chunks (expected due to overlap + tail).
- Average per-chunk latency: ~33 ms. Max latency: ~527 ms (first chunk warmup overhead). Warmup accounts for the elevated RTF (1.60) on this very short file.
- Emitted the same symbol repeatedly (`tʃ…`) because the CTC head is random (uninformative logits). Stabilization still functionally debounced emissions.
- Coverage metric trivially high / meaningless with random head; kept only to exercise comparison path.
- GPU detected; model ran on CUDA (Torch 2.7.0 + cu126). Opportunity to reduce max latency by explicit warmup before timing.

### Second Run (With Warmup Enabled)

Command: `--warmup` flag, same audio & JSON vocab.

Results:
- Warmup forward: ~578 ms (excluded from metrics).
- Streamed sequence empty (random head now produced no stabilized symbol in this short clip; acceptable for latency test).
- Per-chunk latency (post-warmup): avg 7.6 ms, max 21.3 ms.
- Overall RTF (excluding warmup): 0.366.
- Steady-state (excluding first measured chunk): avg 6.8 ms, RTF 0.312.
- Offline length 3 vs streamed 0 (again, logits meaningless at this stage).

Interpretation:
- Warmup removal drops steady-state latency substantially (from ~33 ms to <8 ms per 40 ms chunk) giving real-time factor well below 1.
- Remaining optimization headroom: fp16, quantization, and head training (to yield meaningful emissions).

### Immediate Improvement Opportunities

| Area | Action | Status |
|------|--------|--------|
| Warmup spike | Run a dummy forward before processing & exclude from metrics | Planned |
| Steady-state RTF | Report RTF excluding first N warmup chunks | Planned |
| FP16 inference | Optional `--fp16` flag (autocast / `.half()`) | Planned |
| Vocab load | JSON support added | Done |
| Trained head | Load existing phoneme classifier weights | Pending |
| Confidence | Track max softmax or log-prob across stabilization window | Pending |
| Resampling | Add automatic resample path for non-16k input | Pending |
| Mic input | Real-time stream via `sounddevice` | Pending |

## Current Limitations / Risks

- Random CTC head => emitted sequence not semantically meaningful.
- Warmup latency skews RTF for short clips; need warmup exclusion.
- No state reuse between chunks (full forward each chunk, though model inherently handles variable length); potential to optimize with receptive field caching if migrating to streaming-friendly backbone.
- No precision / quantization experiments yet (fp16, int8) for latency.

## Next Step Candidates

1. Replace random head with trained linear CTC layer (fine-tune on existing phoneme dataset).  
2. Add pre-forward warmup + separate steady-state latency/RTF metrics.  
3. Implement microphone streaming (`sounddevice` callback, ring buffer).  
4. Add resampling path (e.g., `torchaudio.functional.resample`) to enforce 16 kHz.  
5. Confidence scoring: track per-symbol average max-logit across stabilization window.  
6. Beam search decoder (HF / torchaudio beam with optional LM) behind a flag.  
7. Timestamp refinement: accumulate frame alignment indices & compute midpoints.  
8. Endpoint detection (silence energy threshold or blank-run length).  
9. Sliding window state optimization (retain receptive field tail instead of reprocessing full chunk).  
10. Export ONNX / TorchScript variant for lower-latency inference.  
11. Add unit test to assert non-empty output & latency bounds (< real-time).  
12. Optional fp16 / autocast support toggle for speed.  

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
- [x] Support JSON phoneme vocab file (`phoneme_labels.json`)
- [x] Warmup forward pass (exclude from metrics)
- [x] Steady-state RTF reporting
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
- [ ] Optional fp16 inference flag

## Decision Log

- Chose 40 ms window / 20 ms stride (50% overlap) as starting point; adjust after meaningful logits exist.
- Stabilization requirement of 2 repeats used to reduce flicker without large latency addition.
- Deferred beam decoding until non-random logits reduce noise; avoids wasted tuning effort.
- Added JSON vocab support after confirming full inventory is stored in `phoneme_labels.json` (vs earlier minimal test vocab).
- Identified need to separate warmup latency from steady-state metrics for accurate RTF.

## Open Questions

- Final target latency & RTF thresholds (define acceptance target: e.g., steady-state RTF <= 0.25?).
- Phoneme inventory stability: are composite symbols final or will splitting / merging occur?
- Integration path: direct WebSocket streaming vs. batching on server side?
- Do we require alignment (frame-level timestamps) in MVP or only stabilized symbol stream?

---
Feel free to update status and promote remaining checklist items into individual tasks / issues under Epic 1.
