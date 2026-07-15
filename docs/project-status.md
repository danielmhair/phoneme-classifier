# Project Status: Where We've Been, Where We're Going

**Last updated:** 2026-07-14 (keep this in sync as milestones land; completion of any item is declared by Daniel, per CLAUDE.md)

The one-paragraph version: this project went from *"we have accuracy numbers nobody can trust"* to *"we have honest numbers, they predict live reality, and the best configuration scores 78% top-1 / 94% known-target top-3 on a fresh good-mic session."* The remaining gap to the 85% ship bar cannot be closed by modeling tricks on the current data - it needs recordings of the children the product is actually for, which is what the data-collection game exists to gather. That's where we are going.

---

## Part 1: Where we've been

### The starting point (before 2026-07-10): nothing was trustworthy

- The headline numbers (87% / 85% / 80%) came from evaluations with **data leakage** and **broken CTC decoding** (three different decode algorithms coexisted; the one every consumer used averaged log-probs over time, destroying CTC's alignment).
- A historical precedent loomed over everything: an earlier MLP-only version tested at **82% and delivered 18%** in the real world - a 64-point evaluation-vs-reality gap.
- WavLM-CTC had once scored 2.78% (random chance) in a re-evaluation, cause unknown.
- The "best" model's ONNX export was a 0-byte file.

### The Evaluation Foundation (2026-07-10 → 07-12): make the numbers mean something

Full story: [Evaluation Foundation PRD](../plans/prds/07-10-2026-PRD-models-trustworthy.md) - core deliverables declared complete by Daniel 2026-07-12.

What it took, in brief:
- **Fixed the measurement bugs**: unified all CTC decoding into [workflows/shared/ctc_decode.py](../workflows/shared/ctc_decode.py); fixed the WavLM embedding-extraction crash that plausibly explains the 2.78%; closed the label-index-mismatch bug class by construction; made ONNX export failures loud.
- **Fixed the data**: 192 wavs renamed (stale speaker prefixes that would have contaminated fold purity), 54 zero-byte wavs restored, full 6,033-file corpus verified readable/16kHz.
- **Built the LOSO harness** ([evaluation/harness/](../evaluation/harness/)): leave-one-speaker-out evaluation with persisted split manifests, canonical label ordering, resumable embedding caches, and known-target rank/margin metrics.
- **The honest baseline** (Chloe = the only target-age speaker): MLP 45.07%, Wav2Vec2-CTC 58.69%, WavLM-CTC 57.90%. No model near the ship bar; every model far above the bug floor. Both CTC models beat MLP on every fold.

### The live-mic smoke test (2026-07-12): the numbers describe reality

Fresh recording sessions (111 clips each, same speaker, good mic and bad mic) scored via [evaluation/live_mic_smoke.py](../evaluation/live_mic_smoke.py) against full-corpus models:

- **The historical 82→18 collapse did not recur.** Good-mic live top-1 landed slightly *above* the honest LOSO prediction. The evaluation now tells the truth - that was the whole point.
- **Mic quality alone costs ~21-24 top-1 points** on every model - the largest single factor measured anywhere in the project. (MLP on a consumer mic scores ~16%, essentially the historical real-world number - the old collapse now looks substantially explained as MLP + channel effects.)
- Known-target top-3 (the gameplay-relevant metric, since the game knows which phoneme it asked for) runs far ahead of open 37-way top-1 throughout.

### The augmentation experiments (2026-07-11 → 07-13): a model-specific lever, and a surprise winner

Full story: [channel-augmentation experiment](../plans/channel-augmentation-experiment.md).

- **Production-style augmentation** (speed/pitch/noise) on the Chloe LOSO fold: +8.8 for MLP, only +1.75/+3.32 for the CTC models. Helpful, not decisive.
- **Channel/mic-simulation augmentation** (telephone bandpass, cheap-mic lowpass, clipping, 8kHz roundtrip): transformed Wav2Vec2-CTC (+13.5 good mic at 20 epochs, new best single model at 70.27% live) while *hurting* WavLM-CTC at every dose (monotonic dose-response; its pretraining already covers degraded speech). **Lesson: augmentation strategy is model-specific.**
- **The fused pair** (probability-averaging w2v2+channel-aug with baseline WavLM - two models trained on different distributions, decorrelated errors): **78.38% good mic / 52.25% bad mic, ~+8 over any single model on both conditions; good-mic known-target top-3 93.69%.** This is the working configuration.

### Where the numbers stand today

| Metric | Value | Source |
|---|---|---|
| Honest LOSO, target-age speaker (best single model) | 57.9-58.7% | full 5-fold run, 2026-07-10 |
| Live good-mic, fused pair, adult speaker | **78.38%** top-1, **93.69%** known-target top-3 | pairing eval, 2026-07-13 |
| Live bad-mic, fused pair | **52.25%** top-1 | pairing eval, 2026-07-13 |
| Ship bar (target-age children) | 85% - **not met** | PRD §6 |

Important honesty note: the fused-pair numbers were selected against one adult speaker's two sessions. They demonstrate the *method*; the *number that matters* will be measured on held-out children.

---

## Part 2: Where we're going

The confirmed conclusion of the Evaluation Foundation is that the remaining gap is a **data problem, not a modeling problem**: only one voice in the training corpus belongs to the product's target age band. Everything below serves closing that gap and proving it closed.

### 1. The data-collection game (in progress - the main thrust)

The Phoneme Hatchery: a web game + parent onboarding + reviewer app (lives in the separate `light-haven-sites` repo; see [game/README.md](../game/README.md) for the split). PRD: [data-collection game](../plans/prds/07-12-2026-PRD-data-collection-game.md).

- **Wave 1 target: ~20 families**, children ages 4-8 (the product's band; 9-15 phased later), ~2 short sessions per child, device diversity welcomed - home mics are exactly the channel variety the models need.
- **Label quality is the top data risk**: verified-good clips only, via the reviewer flow (the base corpus's 192 misnamed + 54 zero-byte files are the cautionary tale).
- Training-side plumbing already landed in this repo: [game/export_tool/export_verified.py](../game/export_tool/export_verified.py) (`poe game-export`) is the ingestion pipeline - reviewed-verified clips move into `recordings/<child_code>/` with age-band metadata as verdicts land (idempotent, re-runnable), with a structural audio gate (16kHz mono, non-empty, atomic writes) so nothing malformed can enter the corpus. Proven end-to-end 2026-07-14 with a test child (game -> review -> export -> harness manifest + holdout split all correct; test child then removed).

**Before wave-1 families (the remaining checklist - training-repo side is done, all of these are app/ops items):**

1. Record the reference prompt WAVs the game plays per phoneme (app README §3).
2. Seed golden clips (a few known-good/known-bad clips flagged `is_golden` with `golden_expected_verdict`) - reviewer reliability stats are meaningless without them; currently a manual-SQL step per the app README.
3. Device smoke-test on real family hardware (iOS Safari mic permissions / AudioWorklet support - PRD §12).
4. ~~Solve the single-reviewer stall~~ **Resolved 2026-07-14 (Daniel's call: one reviewer)**: with a single distinct reviewer, confusion-pair phonemes (dh, th, s, sh, m, n) now resolve by the same majority rule as other phonemes; the stricter two-review agreement rule re-arms automatically if a second reviewer is ever onboarded. Exports report `confusion_resolved_single_review` for transparency. Golden clips (item 2) are now the *only* reviewer-reliability check, which makes seeding them matter more.
5. Tombstone the test child (child_002) in the DB so `poe game-export` doesn't re-ingest it once real collection starts.

### 2. Holdout discipline + the learning-curve stopping rule (tooling landed)

- ~20% of collected children (stratified by age band, flagged at signup) are **holdout children** - their clips never enter any training fold: [evaluation/harness/holdout.py](../evaluation/harness/holdout.py).
- "How many kids do we need?" is answered empirically, not by guessing: train at N = 10, 20, 30... children and plot held-out accuracy per age band ([evaluation/harness/learning_curve.py](../evaluation/harness/learning_curve.py)). Curve still climbing → recruit more; flat → stop. Per-age-band reporting is mandatory so "78% on kids" can't hide "61% on 5-year-olds."

### 3. Re-validate the fused pair on child data (the pre-registered test)

The first real question the child data answers: does the fused-pair advantage (and channel augmentation's w2v2 boost) hold on target-age voices? Model selection stops until this data exists - further tuning against the existing adult sessions would be overfitting to one voice.

**The retrain → replay loop** (run whenever a meaningful batch of new children has been ingested; all with `PYTHONUTF8=1`):

```bash
poetry run poe game-export        # 1. ingest newly verified clips
poetry run python -m evaluation.harness.full_train                # 2. retrain baseline models (WavLM half; holdout children auto-excluded)
poetry run python -m evaluation.harness.full_train --channel-aug --ctc-epochs 20 --out-dir evaluation/full_models_channel_e20   # 3. retrain the w2v2+channel half
poetry run poe live-mic --replay evaluation/live_mic_sessions/<session> --models-dir <models dir>   # 4. per-model numbers on the saved good/bad-mic sessions
poetry run poe export-fused-onnx  # 5. fused-pair headline numbers on both sessions (+ ONNX parity)
poetry run poe holdout-eval       # 6. the number that decides anything: held-out children
```

Steps 4-5 replay the saved adult sessions (robustness trend; copy each session's `summary.json` aside first to keep a before/after trail). Step 6 is the pre-registered test itself and needs holdout-flagged children to exist.

### 4. Production promotion (deferred until validation passes)

- **The ONNX-export/latency readiness check is done (2026-07-14)**: the fused pair exports cleanly (the historical 0-byte failure mode is exercised and closed), and the pure-ONNX chain matches PyTorch on all 222 saved live-mic clips exactly. Latency is the honest catch - ~2x a single model, marginal at 0.5s real-time frames and over the ~150ms temporal-brain budget beyond that, though comfortably fast for the game's per-clip scoring (~0.3s per clip). Artifacts + full report in `evaluation/full_models_fused_onnx/`; details in the [channel-augmentation experiment doc](../plans/channel-augmentation-experiment.md).
- Still deferred until child-data validation: promote channel transforms into the production augmentation ([workflows/shared/s0b_augment_audio.py](../workflows/shared/s0b_augment_audio.py)), promote fusion into the inference path, and decide the real-time shape (single-model WavLM streaming with fused end-of-utterance rescoring vs 0.5s hops vs quantization).
- Then the downstream epics unlock in order: temporal-brain tuning on real signal, Whisper distillation, GOP scoring - all were deliberately blocked on trustworthy numbers.

### The bar that ends this chapter

**85% top-1 on held-out target-age children** (with known-target top-3 already far ahead of that). When the learning curve crosses it - or flattens below it and forces a new conversation - this document gets its next major update.
