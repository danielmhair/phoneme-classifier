# PRD: Evaluation Foundation — Making Model Accuracy Trustworthy

**Status:** In progress - full no-augmentation LOSO baseline done; augmented Chloe-fold pilot done, awaiting user decision on full 5-fold augmented rerun
**Owner:** Daniel Hair
**Last updated:** 2026-07-11
**Source Context**: [Grill Me Session](./07-10-2026-PRD-models-trustworthy-grill-me)

> Per CLAUDE.md: this PRD must be kept in sync with actual status as work progresses. Completion is declared by the user, not inferred from code existing.

## Implementation status (2026-07-10)

**Not yet declared complete by the user - do not treat anything below as "done" in the PRD-completion sense, only as "landed in code."**

Diagnostics (§5.1) and CTC decoding fix (§5.2):
- Full codebase audit found three different CTC decode algorithms coexisting (correct greedy decode used only in-process for training-time validation; a "first non-blank timestep" heuristic in `s7_test_onnx.py`; and the documented `np.mean`-over-time bug in `inference/cli/model_loader.py`, which is what every ONNX/CLI/eval consumer actually used). All three now route through one module, `workflows/shared/ctc_decode.py` (greedy decode for top-1, peak-over-time posterior for ranking/margin scoring). See [docs/codebase-map.md](../../docs/codebase-map.md) for full detail.
- Label-index mismatch risk audited: CTC workflows' `.pkl`/`.json` are guaranteed consistent by construction (same in-memory list); MLP workflow has a real (if guarded) divergence risk, mitigated in the harness by using one fixed canonical label list for every model/fold rather than a per-fold-refit encoder.
- ONNX export (`s5_export_ctc_onnx.py`, both CTC workflows) now asserts the output file is non-empty and raises loudly instead of swallowing failures - root cause of the historical 0-byte export on the reference machine wasn't reproducible here (this machine had never completed a training run at all), but the silent-failure path that would have hidden it is closed.
- Found and fixed a real, previously-undiscovered bug: WavLM embedding extraction (`ctc_wavlm_workflow/s2_extract_embeddings_temporal.py`) called `Wav2Vec2Processor.from_pretrained("microsoft/wavlm-base")`, which requires a tokenizer config that checkpoint doesn't have - this raises `OSError` immediately, uncaught locally, swallowed by `workflow_executor.py`'s catch-and-continue. Plausible real explanation for WavLM-CTC's near-random (2.78%) reported accuracy: if this ran, there was no trained model, not a badly-trained one. Fixed to use `Wav2Vec2FeatureExtractor`.
- Fixed an unrelated environment bug blocking everything: a stray, untracked `torchvision==0.19.1` in the Poetry venv, incompatible with pinned `torch==2.3.1`, broke every `transformers` import that touched image utilities (including `Wav2Vec2Processor`/`Wav2Vec2FeatureExtractor`).

LOSO harness (§5.3):
- Built `evaluation/harness/` - speaker-first manifest (built directly from `recordings/`, sidestepping the phoneme-first `organized_recordings/` structure), resumable per-base-model embedding cache, unified MLP/CTC fit-predict interface, fold runner persisting a real split manifest (`split_manifest.json` - actual held-out file_ids, not a `random_state`) and per-sample known-target rank/confidence-margin (`predictions.csv`), plus a summary with bug-floor/ship-bar flags (`summary.json`).
- **Scope decision made during implementation, not explicitly in the original PRD text**: this first harness pass trains on original recordings only, no noise/speed/pitch augmentation (`s0b_augment_audio.py`'s pipeline). Flagged here for visibility, not yet confirmed with the user.
- Smoke-tested end-to-end on a 3-phoneme/60-file subset (2 CTC epochs) - mechanically sound, all artifacts produced correctly.
- **The harness's split-manifest persistence caught a real, material data-quality bug**: 159 files under `recordings/chloe/` (~14%) and 33 files under `recordings/callie/` were filenamed with a different speaker's prefix than the directory they lived in. Since a mislabeled speaker would have contaminated LOSO fold purity (a training speaker's voice counted as "held-out" data), this was confirmed with the user (audio content matches the directory, filenames were stale) and fixed by renaming at the source - not a harness workaround.
- Full run completed 2026-07-10 (all 5 speakers, all 37 phonemes, all 3 models, 20 CTC epochs). Results in `evaluation/loso_results/full_run_20260710/`. No fold for any model landed anywhere near the ~10% bug floor - the harness itself looks trustworthy (§6 "harness done" bar appears met, pending user review).

### Full LOSO results (headline: Chloe)

| Model | callie | cassie | **chloe (headline)** | dan | sky | Meets 85% ship bar? |
|---|---|---|---|---|---|---|
| MLP Control | 63.80% | 56.54% | **45.07%** | 35.66% | 43.01% | No |
| Wav2Vec2 CTC | 82.38% | 70.23% | **58.69%** | 54.70% | 61.60% | No |
| WavLM CTC | 74.50% | 67.84% | **57.90%** | 60.50% | 64.51% | No |

Known-target metric (chloe, top-3 = true phoneme ranked in model's top 3 guesses): MLP 70.13%, Wav2Vec2-CTC 77.82%, WavLM-CTC 80.96%.

**Honest read, per §6/§7's own framing**: no ship bar met. Every model is well clear of the bug floor (real signal, not a measurement artifact) and well below 85% (not game-ready). Chloe scores below the other-4-speaker average on every model (e.g. MLP: 49.75% average of the other four vs 45.07% on Chloe) - consistent with her being the only target-age-band speaker, i.e. a genuine generalization gap, not noise. Both CTC models clearly and consistently beat MLP control on every single fold, including the headline fold - a real, reproducible finding, unlike the old flawed evaluation's ranking (which had Wav2Vec2 87% > WavLM 85.35% > MLP 79.73%, now replaced by honest numbers roughly 30 points lower across the board). Per §6: "the correct conclusion is needs more target-age-band data, not the project failed or abandon this architecture."

Live-mic smoke test (§5.4) - in progress (2026-07-12), tooling landed, fresh-clip recording session not yet done:
- **The flagged mean-pooling question is answered - not a live bug.** Verified by loading `wav2vec2.onnx` and running audio through it: mean-pooling is baked into the ONNX graph itself, output is already `[1, 768]` (exactly what `phoneme_mlp.onnx` expects, which itself outputs `[1, 37]`). The commented-out pooling and stale `[1, T, H]` comment in `classify_voice_onnx.py` were misleading but harmless; comment corrected in place.
- **The existing CTC live-mic scripts (`workflows/ctc_*/validations/classify_voice_ctc.py`) were found genuinely broken, never routed through the 2026-07-10 decode fix**: they feed raw audio into `CTCModel`, which takes pre-extracted `[batch, T, 768]` embeddings (no feature extraction happens at all); the WavLM copy also has the known `Wav2Vec2Processor("microsoft/wavlm-base")` OSError bug; both fall back to a `MockCTCClassifier` emitting random phonemes when imports fail. Left untouched pending user decision (delete vs repair) - superseded for §5.4 purposes by the unified tool below.
- **New unified smoke-test tooling** (covers all three models with one honest pipeline, per the harness's bake-off intent): `evaluation/harness/full_train.py` (`poe live-mic-train`) trains all three models on the full 5-speaker corpus from the harness embedding caches (no augmentation, 20 CTC epochs - same settings as the LOSO baseline the smoke test is compared against) and saves them to `evaluation/full_models/`; `evaluation/live_mic_smoke.py` (`poe live-mic --speaker <name>`) records fresh clips (raw 3s/16kHz mono, saved then scored from the file - byte-identical audio path to training data), scores every clip against all three models via the shared CTC decode, and writes per-clip `predictions.csv` + `summary.json` with the same top-1/known-target-rank/margin metrics as the LOSO harness. `--replay <session-dir>` re-scores a saved session without re-recording.
- **First fresh session recorded and scored (2026-07-12, good mic, user's adult voice, 111 clips = 37 phonemes x 3)** - session `dan_goodmic_20260712_154201` under `evaluation/live_mic_sessions/`. Results vs the most-comparable honest references:

  | Model | Live-mic top-1 (fresh session) | Dan's LOSO fold (model never heard him) | Training-set replay sanity check | Live top-3 known-target |
  |---|---|---|---|---|
  | MLP Control | 36.94% | 35.66% | ~100% | 63.96% |
  | Wav2Vec2 CTC | 56.76% | 54.70% | ~100% | 77.48% |
  | WavLM CTC | 65.77% | 60.50% | ~100% | 83.78% |

  Honest read: **the historical 82%→18% collapse did NOT recur** - live-mic scores are consistent with (slightly above) LOSO levels, so the evaluation pipeline and the LOSO numbers describe reality. But note what that means: these full-corpus models were *trained on 1,276 of this speaker's own clips*, yet a fresh session only scores at roughly the level of a never-heard speaker - i.e. **session/recording-conditions shift costs approximately the entire benefit of being a known speaker**. This independently supports both the augmentation lever (robustness to conditions) and the more-target-age-data conclusion. WavLM-CTC is the clear live-mic winner (65.77% top-1, 83.78% top-3). Consistent live failure classes across models: dh (always scored as th), n/m nasal confusion, and MLP is far weaker live than the CTC models (16 phonemes at 0/3).
- **Bad-mic comparison session recorded and scored (2026-07-12, same protocol, 111 clips)** - session `dan_badmic_20260712_155952`:

  | Model | Good mic top-1 | Bad mic top-1 | Mic cost | Good mic top-3 | Bad mic top-3 |
  |---|---|---|---|---|---|
  | MLP Control | 36.94% | 16.22% | -20.7 | 63.96% | 45.05% |
  | Wav2Vec2 CTC | 56.76% | 32.43% | -24.3 | 77.48% | 57.66% |
  | WavLM CTC | 65.77% | 44.14% | -21.6 | 83.78% | 70.27% |

  Honest read (framing corrected per user): **this is a dramatically better outcome than the historical system, on both axes that matter.** Historically the MLP-only version *claimed* 82% and *delivered* 18% absolute real-world accuracy - a 64-point evaluation-vs-reality gap. Now: (1) the gap is essentially closed - good-mic live top-1 (65.77% WavLM-CTC) lands slightly *above* the honest LOSO prediction (60.50% on the dan fold), i.e. the evaluation numbers describe reality, which was this PRD's core goal; and (2) absolute real-world accuracy is 2.4-3.6x the historical 18% (44.1% worst-case bad mic, 65.8% good mic, best model). Within that, mic quality is the largest single measured factor (~21-24 top-1 points on every model) - and today's MLP-on-consumer-mic score (16.2%, ~the historical 18%) is consistent with the old real-world number having been an MLP+channel-effects result. The existing augmentation transforms (speed/pitch/noise overlay) do not simulate channel/mic effects (bandwidth, EQ, compression) - channel augmentation is a concrete, newly-motivated lever alongside more target-age data. WavLM-CTC stays the best model under both conditions.
- §5.4's originally-scoped work (extend live-mic validation to CTC models, verify the pooling question, record a fresh tripwire set) is now all done pending user review; user judgment on whether the tripwire outcome is satisfactory, and completion declaration, are Daniel's.

### Augmented Chloe-fold pilot (2026-07-11)

The no-augmentation scope call above was revisited with the user: an augmented pilot on the Chloe fold only was approved and run before paying for a full 5-fold augmented rerun. Same harness, same held-out set (all 1,145 chloe files), training set = 4 other speakers' 4,888 originals + 25,364 production-style augmented variants (speed 0.9x/1.1x, pitch ±2 semitones, stochastic noise overlay - the `s0b_augment_audio.py` transforms, seeded per-file for determinism). `ctc_epochs=8` instead of the baseline's 20, deliberately: the augmented training set is ~6x larger, so total gradient updates stay comparable. Results in `evaluation/loso_results/augmented_chloe_pilot/`.

| Model | Chloe top-1 (baseline, no aug) | Chloe top-1 (augmented) | Delta | Chloe top-3 known-target (baseline → augmented) |
|---|---|---|---|---|
| MLP Control | 45.07% | **53.89%** | **+8.82** | 70.13% → 78.52% |
| Wav2Vec2 CTC | 58.69% | **60.44%** | +1.75 | 77.82% → 82.45% |
| WavLM CTC | 57.90% | **61.22%** | +3.32 | 80.96% → 85.33% |

No fold near the bug floor; no model meets the 85% ship bar. Honest read: augmentation genuinely helps every model, but the gain is largest where it matters least (MLP, the weakest model, +8.8) and modest on the two CTC models (+1.75 / +3.32). The best available Chloe number moved 58.69% → 61.22% - real progress, still ~24 points short of the ship bar. Augmentation alone does not close the target-age generalization gap; §6's "needs more target-age-band data" conclusion stands. Top-3 known-target improved across the board (WavLM-CTC now 85.33%), which keeps the known-target gameplay mode looking materially more viable than open 37-way top-1.

**Open for user decision**: whether to run the full 5-fold augmented rerun (MLP's +8.8 crossed the agreed ~+5 "material" threshold, the CTC models did not; the pilot's per-model deltas may not replicate on other folds). Proposed, not started.

Data/infrastructure notes from the pilot (secondary machine, `c:\Workspace\fast-api-phoneme-python`):
- This machine's copy of `recordings/` predated two fixes from the primary machine and was brought up to parity before the run: the 192 stale speaker-prefix filenames (159 chloe + 33 callie) were renamed to match their speaker directory (user-approved, same fix as the primary), and 54 zero-byte wavs (truncated in the original copy; concentrated in ai_eɪ/b/ch/f/sh/th across all 5 speakers) were re-copied from the primary machine by the user. Full corpus then verified: 6,033 files, all readable, all 16kHz, zero prefix mismatches.
- A mid-run process kill left `harness_cache/wav2vec2/index.json` as 4.4MB of NTFS null bytes, breaking resume. The index was rebuilt from the on-disk `.npy` files (all 21,012 header-validated, zero corrupt), and cache-index writes in `evaluation/harness/embeddings_cache.py` + `augmentation.py` are now atomic (temp file + `os.replace`) so a kill can no longer corrupt them.

## 1. Problem Statement

Every accuracy number currently associated with this project's three phoneme models (MLP Control, Wav2Vec2 CTC, WavLM CTC) is unverified or known-flawed:

- The headline "87.00% / 85.35% / 79.73%" figures in `README.md`/`CLAUDE.md` come from evaluations with **data leakage** (tested on data overlapping training) and, for the CTC models, a **broken decoding method** (`np.mean` across timestep log-probs, which destroys CTC's alignment structure instead of doing greedy/beam collapse).
- A corrected fast re-evaluation (`evaluation/accuracy_summary_fast.json`) showed MLP 95.56%, Wav2Vec2-CTC 64.44%, WavLM-CTC 2.78% (~random chance for 37 classes) — internally inconsistent with the documented numbers and with each other.
- No leave-one-speaker-out (LOSO) or held-out-child validation has ever been run. Of 5 recorded speakers, only Chloe (age 7) is in the product's actual target age band.
- A historical precedent exists (documented in Notion, "Phoneme Classifier Explained") of a properly-split 82% test accuracy collapsing to 18-20% real-world accuracy for an earlier MLP-only version — proof that even a leakage-free split can hide real generalization failure.
- A historical precedent also exists of a **label-index mismatch** bug producing a near-identical failure signature (~99% on a handful of classes, ~0.6% on the rest) to what today's WavLM-CTC 2.78% result looks like — meaning the current bad number could be a measurement bug, not a real ceiling, and this has not been ruled out.
- Even on the one machine where training is known to have completed end-to-end (a sibling Windows checkout, `C:\Workspace\fast-api-phoneme-python`), the Wav2Vec2-CTC ONNX export (`phoneme_ctc.onnx`) is a **0-byte file** — meaning the model documented as "BEST PERFORMER" cannot currently be loaded for inference at all, even in the best case.

**Bottom line:** nothing in this repository can currently tell you whether a given model is actually good. This PRD is the prerequisite for every downstream decision (which model to ship, whether the temporal brain is stabilizing signal or noise, whether more data is needed) — none of those can be answered honestly until this is fixed.

## 2. Goals

1. Produce one number per model (MLP, Wav2Vec2-CTC, WavLM-CTC) that everyone agrees is an honest measure of "how well does this work on a child it has never heard before."
2. Rule out measurement bugs (label mismatch, broken decoding, broken export) as an explanation for bad numbers, before drawing conclusions about model quality.
3. Leave behind a reusable, single-interface harness (matching the existing "Multi-Model Bake-off Harness" epic's intent) that later work (Whisper distillation, new data, new models) can plug into without rebuilding evaluation from scratch.

## 3. Non-Goals (explicitly out of scope for this pass)

- **CVC / multi-phoneme sequence evaluation.** The product's core interaction is a child producing one isolated phoneme at a time, interactively, during gameplay — confirmed as the primary design, not a simplification. Full-word or CVC-carrier evaluation is deferred until real CVC data exists (a later data-collection epic), and synthetic concatenation of isolated clips was considered and rejected (doesn't reflect real coarticulation).
- **Full GOP (Goodness-of-Pronunciation) accept/reject scoring.** Most gameplay already knows the target phoneme, which makes GOP-style scoring a good long-term fit — but full GOP needs calibrated per-phoneme thresholds against human-judged correctness labels that don't exist yet. Deferred.
- **SpeechOcean762 or any external dataset integration.** Considered and rejected for this pass — SpeechOcean762 is continuous non-native-speaker read-aloud audio, not isolated phonemes; segmenting it to fit would force data into a shape it wasn't recorded for ("circle into a square"), risking worse artifacts than the problem it would solve. Only revisit if the existing 5-speaker dataset proves genuinely insufficient after this pass.
- **Runtime changes (sherpa-onnx / replacing UE5 NNE).** Confirmed the Unreal integration already exists and works for the MLP pipeline (`PhonemeClassifierSubsystem`). Runtime work is downstream of knowing which model is worth deploying — not blocking this pass.
- **Any new game features, temporal-brain parameter tuning, or Whisper distillation work.** All of it is downstream of trustworthy numbers and should not proceed until this PRD's deliverables land.

## 4. Environment

**Decision: Windows-native Poetry. No WSL, no Conda.**

Empirical basis: the one machine with a complete, working end-to-end training run (`C:\Workspace\fast-api-phoneme-python`, a sibling checkout) uses a Windows-native Poetry virtualenv (via Git Bash / MINGW64), not WSL. WSL only appears as the origin of two symlinked data folders, not as the environment that ran training. The current machine's Conda environment was found broken (`pytorch 1.9.1` installed alongside `torchvision 0.15.2`, which requires `torch>=2.0` — an unpinned-install version conflict).

**Status: verified working end-to-end on this machine (2026-07-10)** — `poetry install` completes, all core imports succeed, `poe` tasks run, and `poe debug-shared` confirms actual project code imports correctly. It took several fixes beyond just copying the reference machine's versions:

| Package | Version | Note |
|---|---|---|
| Python | 3.9.13 | Must be pinned explicitly (`poetry env use ...`) — otherwise resolves to system Python 3.13, which breaks numpy's build |
| torch | 2.3.1 | matches reference machine |
| torchaudio | 2.3.1 | pinned exact; loose range let the resolver pick an incompatible 2.11.0 |
| transformers | 4.50.3 | matches reference machine |
| onnxruntime | **1.19.2**, not 1.20.1 | 1.20.1 has no Python-3.9 Windows wheel at all — it was never actually functional on the reference machine either (explains why `import onnxruntime` failed there) |
| scikit-learn | 1.6.1 | matches reference machine |

Also required:
- **Removed unused dependencies** that were breaking the Windows install: `montreal-forced-aligner` (pulls in `greenlet`, which needs MSVC Build Tools to compile — not installed), `praat-parselmouth`, `phonemizer`, `pronouncing` (all forced-alignment/GOP tooling, out of scope per §3), `fastapi`/`uvicorn`/`python-multipart` (dead FastAPI server, confirmed abandoned), `openai-whisper` (declares an unmarked `triton` dependency with zero Windows wheels — the actual root cause of an early install failure; also out of scope per §3, Whisper distillation epic). Confirmed via grep that nothing in `workflows/`, `inference/`, `evaluation/`, or `tests/` imports any of these.
- **`PYTHONUTF8=1` required** when running `poe` commands — task definitions print emoji, and Windows' default console codepage (`cp1252`) can't encode them, crashing the task otherwise. Not a dependency issue, a Windows console-encoding gap.

**Explicitly decided against for this pass:** upgrading to a newer Python (latest `torch` is 2.13.0, requiring Python 3.10+, ten minor versions ahead of what's verified here). See §8.

## 5. Scope of Work

### 5.1 Diagnostics (run first, before any code changes)

1. **Label-index mismatch check.** For each model, confirm `phoneme_labels.json` order matches the label encoder's / ONNX output's actual class order. Historical precedent: a mismatch here previously produced a "favors 9 classes, ~0% on the rest" failure signature nearly identical to today's WavLM-CTC 2.78% result. Cheap to check, must be ruled out before trusting any decoding-bug theory.
2. **Confirm/fix the Wav2Vec2-CTC ONNX export.** `phoneme_ctc.onnx` is 0 bytes on the reference machine. Determine why export silently produced an empty file and fix it — this model cannot be evaluated via ONNX at all until this is resolved.

### 5.2 CTC decoding fix

Replace the current `np.mean(log_probs_seq, axis=0)` approach (which destroys CTC's frame-by-frame alignment structure) with **greedy decoding**: read frames in time order, collapse repeated consecutive predictions, drop blank tokens. Beam search is explicitly deferred — phonemes here are short, isolated single sounds with minimal sequence ambiguity, so greedy is sufficient and much simpler to implement/debug. Revisit beam search only if greedy is observed to blur genuinely distinct phonemes.

### 5.3 Leave-one-speaker-out (LOSO) evaluation harness

- Rotate through all 5 speakers (dan, callie, sky, cassie, chloe): train on 4, evaluate on the 1 held out, for each of the 3 models.
- **Sky and Cassie (speech-impediment speakers) are included normally** in training and the LOSO rotation, same as everyone else — recording sessions were specifically supervised to ensure produced sounds matched target labels, so the mislabeling risk that would otherwise justify excluding them does not apply here.
- **Chloe's held-out run is the headline number** — she is the only speaker in the product's actual target age band (7 years old), so her LOSO score is the best available proxy for "does this work on the kid this product is for."
- Persist the actual list of held-out files per fold (not just a `random_state` seed) so results are reproducible without re-running training code unchanged. No such manifest currently exists anywhere in the repo.
- Report a **secondary, low-cost metric alongside top-1 accuracy**: for each sample, since the true/intended phoneme is always known, log where it ranked among the model's predictions and its confidence margin vs. the top rival (the "log-prob delta" concept). This is near-free to compute from the same posteriors and gives a more representative read of the known-target gameplay mode than strict open 37-way top-1 accuracy alone. This is *not* full GOP — no threshold calibration or accept/reject decision is being built here, just an additional reported number.
- Unify all three models behind one evaluation interface, per the existing "Multi-Model Bake-off Harness" epic's intent, so later models/data slot in without rework.

### 5.4 Live-mic smoke test

After LOSO numbers look trustworthy, extend the existing live-mic validation pattern (`workflows/mlp_control_workflow/validations/classify_voice_onnx.py`) to the CTC models, and personally record a small fresh set of clips (new session, not from the existing `recordings/` corpus) as a tripwire against a repeat of the historical 82%-test-to-18%-real-world collapse. Small and cheap — not a new data collection effort.

While extending this script, verify the embedding shape/mean-pooling question flagged during review (line 52-53 of `classify_voice_onnx.py`: mean-pooling is commented out — confirm whether `wav2vec2.onnx`'s output shape already matches what the MLP/CTC heads expect, or whether this is a live bug).

## 6. Success Criteria

**"Harness done" and "model ship-ready" are separate bars — do not conflate them.**

- **Harness done:** LOSO evaluation runs cleanly across all 3 models, all 5 speakers, with correct CTC decoding, no label-index bugs, and a persisted split manifest. This is true regardless of what accuracy number comes out — a correctly-measured 60% is a *successful* harness outcome (we now know the truth), not a failure.
- **Bug floor:** any model scoring below ~10% (near the ~2.7% random-chance floor for 37 classes) indicates a measurement bug, not a real accuracy ceiling. Investigate before concluding anything about the model.
- **Ship bar:** ~85% LOSO accuracy on Chloe specifically, for a given model, is the bar for "ready to move to the next phase" (live-mic pilot, GOP layer). Reasoning: below that, roughly 1-in-6-or-worse wrong feedback starts to read as "this doesn't actually listen" to a child or parent, undermining the product's core trust proposition. (90%+ is the aspirational long-term target; 85% is the working bar for this pass.)
- **If the ship bar isn't met:** the correct conclusion is "needs more target-age-band data" (which the existing "Game for Data Collection" epic exists to address), not "the project failed" or "abandon this architecture."

## 7. Open Risks

- Windows-native Poetry env may not reproduce cleanly on this machine even matching known-good versions (untested).
- The Wav2Vec2-CTC ONNX export bug's root cause is unknown until investigated — could be a quick fix or could reveal a deeper issue with that workflow.
- 85% on Chloe may not be achievable in this first pass given only 4 other (non-target-age) training speakers — this is a real possibility, not just a formality, and should be treated as useful information if it happens.

## 8. Explicit Non-Decisions (deferred, tracked for later, not forgotten)

| Item | Deferred to | Trigger to revisit |
|---|---|---|
| CVC/sequence evaluation | Future data-collection epic | Real CVC-carrier recordings exist |
| Full GOP accept/reject scoring | Future epic | Known-target margin metric proves useful; human-judged correctness labels available |
| SpeechOcean762 / external data | "Data Enrichment" epic | Existing 5-speaker data proves insufficient after this pass |
| sherpa-onnx / runtime replacement | Later epic (per existing Unreal integration already working for MLP) | A model is chosen to ship and CTC needs live decoding in-engine |
