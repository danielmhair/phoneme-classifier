# Channel-Augmentation Experiment

**Approved by Daniel 2026-07-12; run this in its own session/window.**
Spun out of the [Evaluation Foundation PRD](prds/07-10-2026-PRD-models-trustworthy.md)'s
live-mic smoke test findings (see its Implementation status section for the numbers).

## Why

Mic quality alone costs ~21-24 top-1 points on every model (measured 2026-07-12:
WavLM-CTC 65.77% on a good mic vs 44.14% on a bad mic, same speaker, same
protocol, 111 fresh clips each). The existing augmentation transforms
(`workflows/shared/s0b_augment_audio.py`: speed 0.9x/1.1x, pitch ±2 semitones,
noise overlay) simulate none of what a cheap mic does. Channel augmentation
directly targets the largest single measured accuracy factor.

## Design - do NOT evaluate this via LOSO

The corpus test folds are all clean recordings, so channel robustness will not
show up in a LOSO rerun. The held-out test set is the two saved live-mic
sessions (222 genuinely fresh clips, already scored against current models):

- `evaluation/live_mic_sessions/dan_goodmic_20260712_154201/`
- `evaluation/live_mic_sessions/dan_badmic_20260712_155952/`

Baseline to beat (current full-corpus models, `evaluation/full_models/`,
trained no-aug/20 CTC epochs - per-model numbers in each session's `summary.json`):

| Model | Good mic top-1 | Bad mic top-1 |
|---|---|---|
| MLP Control | 36.94% | 16.22% |
| Wav2Vec2 CTC | 56.76% | 32.43% |
| WavLM CTC | 65.77% | 44.14% |

## Steps

1. Add channel-simulation transforms alongside the existing ones (scipy/pydub,
   no new dependencies): bandwidth limiting (e.g. bandpass ~300-3400Hz and/or
   lowpass at 4-8kHz), dynamic-range compression, mild clipping/gain jitter,
   optional bit/sample-rate reduction round-trip. Seed per-file like
   `evaluation/harness/augmentation.py` does (deterministic, resumable).
2. Extend `evaluation/harness/augmentation.py` (or a sibling module) to emit
   channel-augmented variants; keep them cacheable under
   `evaluation/harness_cache/` like the existing pipeline.
3. Retrain full-corpus models WITH channel augmentation via the
   `evaluation/harness/full_train.py` pattern (all 6,033 files + variants;
   embeddings cache already holds all originals). Consider `ctc_epochs` scaled
   down like the augmented pilot did (8 epochs at ~6x data) to keep gradient
   updates comparable. Save to a separate dir (e.g.
   `evaluation/full_models_channel_aug/`) - do not overwrite the baseline models.
4. Re-score both sessions:
   `poetry run python evaluation/live_mic_smoke.py --models-dir <new dir> --replay <session dir>`
5. Compare against the table above. Success = a meaningful chunk of the
   good-vs-bad-mic gap recovered without degrading good-mic accuracy.
   If it works, propose promoting the transforms into
   `workflows/shared/s0b_augment_audio.py` (production) and the LOSO harness.

## Results (2026-07-12, run complete)

Models trained on 30,165 rows (6,033 originals + 24,132 channel variants),
8 CTC epochs, saved to `evaluation/full_models_channel_aug/`. Both sessions
re-scored (per-clip results in each session dir's `*_channel_aug.*` files):

| Model | Good mic: baseline → channel-aug | Bad mic: baseline → channel-aug |
|---|---|---|
| MLP Control | 36.94% → 38.74% (+1.8) | 16.22% → 17.12% (+0.9) |
| Wav2Vec2 CTC | 56.76% → **63.96%** (+7.2) | 32.43% → **39.64%** (+7.2) |
| WavLM CTC | 65.77% → 64.86% (-0.9) | 44.14% → **37.84%** (**-6.3**) |

**Honest verdict: the success criterion was NOT met.** The overall best
configuration is still the baseline (no-channel-aug) WavLM-CTC at 44.14%
bad-mic / 65.77% good-mic - no channel-augmented model beats it on either
condition. Within-model, though, the result is genuinely informative:

- **Wav2Vec2-CTC gained +7.2 on BOTH mics** - the transforms do teach channel
  robustness to a model that lacks it, and w2v2+channel-aug nearly closes the
  gap to WavLM on the good mic (63.96 vs 64.86).
- **WavLM-CTC regressed, hardest on the bad mic (-6.3).** Plausible
  explanation: WavLM's pretraining already includes simulated noisy/overlapped
  speech (its headline advantage), so crude channel simulation may distort its
  input distribution more than it teaches robustness. An epochs confound (8 vs
  the baseline's 20) can't be fully excluded, though the same tradeoff helped
  WavLM in the earlier augmented LOSO pilot.

Possible follow-ups (not started, user's call): milder/probabilistic transform
severity, mixing ratio control (e.g. 1-2 channel variants per file instead of
4), WavLM-specific run at higher epochs, or accepting w2v2+channel-aug as the
robustness candidate and comparing it head-to-head with WavLM on a fresh
child-voice session when data exists.

## Follow-up runs (2026-07-13, approved by Daniel, complete)

Run A = all 4 variants, 20 CTC epochs (kills the epochs confound). Run B =
mild dose (cheapmic+clip only), 8 epochs (tests the severity hypothesis).
Models in `evaluation/full_models_channel_e20/` and `..._channel_mild/`
(replay summaries stored alongside each). CTC top-1, all configs:

| Config | W2V2 good | W2V2 bad | WavLM good | WavLM bad |
|---|---|---|---|---|
| baseline (no aug, 20 ep) | 56.76% | 32.43% | 65.77% | **44.14%** |
| channel 8 ep, all variants | 63.96% | **39.64%** | 64.86% | 37.84% |
| channel 20 ep, all variants (A) | **70.27%** | 38.74% | 61.26% | 30.63% |
| channel 8 ep, mild dose (B) | 62.16% | 36.94% | 65.77% | 42.34% |

Findings:
- **Wav2Vec2-CTC + channel aug at 20 epochs is the new best good-mic model
  overall: 70.27%** (+13.5 over its baseline, +4.5 over baseline WavLM, with
  86.5% top-3). Its bad-mic stays ~+6-7 over its own baseline.
- **The epochs confound is dead, and the severity hypothesis is confirmed as
  a monotonic dose-response for WavLM**: more channel-aug training hurts it
  more (20 ep: 30.63% bad mic, its worst), milder dose hurts less (42.34%),
  zero dose is best (44.14%). Channel augmentation as implemented is simply
  bad for WavLM, at every dose - consistent with its pretraining already
  covering degraded speech.
- **Best-per-condition is now split across models**: good mic ->
  w2v2+channel-e20 (70.27%), bad mic -> baseline WavLM (44.14%). No single
  config dominates. A two-model combination (differently-trained, likely
  decorrelated errors) is a plausible future experiment, as is deciding by
  deployment channel quality.

## Two-model pairing evaluation (2026-07-13) - DECISION: adopt the fused pair

Paired the two per-condition winners (w2v2+channel@20ep + baseline WavLM) and
scored both sessions with deployable combiners. Unlike the earlier 3-model
ensemble (which failed - same training data, correlated errors, MLP score
pollution), this pair is trained on different distributions and shares CTC
score semantics:

| Strategy | Good mic top-1 | Bad mic top-1 | Good top-3 | Bad top-3 |
|---|---|---|---|---|
| best single per condition | 70.27% | 44.14% | 86.49% | 70.27% |
| probability-average fusion | **78.38%** | **52.25%** | **93.69%** | 70.27% |
| max-confidence pick | 78.38% | 51.35% | 86.49% | 71.17% |
| oracle (upper bound) | 85.59% | 57.66% | 98.20% | 80.18% |

**Fusion beats the best single model by ~+8 top-1 on BOTH conditions** - the
new best live numbers are 78.38% good mic / 52.25% bad mic (vs 65.77 / 44.14
when the sessions were first scored). Good-mic known-target top-3 is 93.69%.

Decision (as the accuracy-maximizing path): the working configuration is the
fused pair - extract both backbones' embeddings, run both CTC heads, average
the two probability vectors. Inference cost is ~2x a single model (both
backbones), the MLP is not part of the configuration. Caveats: (1) all of
this selection was done on one adult speaker's two sessions (n=111 each) -
MUST be re-validated on child data when the data-collection game delivers it;
(2) production promotion (s0b_augment_audio.py, UE5 runtime) intentionally
deferred until that validation.

## Environment gotchas (hard-won, see docs/codebase-map.md)

- Windows-native Poetry, Python 3.9.13, CPU only. `PYTHONUTF8=1` for poe tasks.
- Multi-line `python -c` silently no-ops in Git Bash here - use script files.
- Background tasks die if the Claude Code process restarts - keep long runs
  resumable via the cache-index pattern (writes are atomic now; if an
  index.json is ever corrupt, rebuild it from the on-disk .npy files).
- Update the PRD's Implementation status when results land; completion is
  declared by Daniel only.
