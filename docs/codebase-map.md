# Codebase Map

A factual reference to how the training/inference pipeline actually works today, for whoever picks this up next. Written during the [Evaluation Foundation PRD](../plans/prds/07-10-2026-PRD-models-trustworthy.md) implementation (2026-07-10) while auditing every CTC decode site, label ordering, and split-logic path in the repo. See [README.md](../README.md) for the project overview and quick-start commands, and [CLAUDE.md](../CLAUDE.md) for working conventions.

## Workflow step structure

All three workflows share `workflows/shared/workflow_executor.py`, which runs a list of `(label, func, *params)` steps. **Each step's exceptions are caught, printed, and swallowed - the workflow continues to the next step regardless of failure.** This matters: a step can fail loudly in the log and the workflow will still print an overall "done" summary. There is no step-level exit code check anywhere upstream.

### `workflows/mlp_control_workflow/` (13 steps)

| Step | Does |
|---|---|
| `s2_extract_embeddings_for_phonemes.py` | Wav2Vec2 **mean-pooled** embeddings per wav; writes `phoneme_labels.json` (`sorted([dir names])`) |
| `s3_classifier_encoder.py` | sklearn `LabelEncoder` + `MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42)`; 80/20 `train_test_split(stratify=y, random_state=42)`; hard-asserts `n_classes == 37` before saving |
| `s4_fine_tune_on_noisy.py` | Fine-tunes on noisy/low-quality embeddings (not in the numbered `0_workflow.py` sequence) |
| `s5_visualize_results.py` / `s6_confusion_pairs.py` / `s7_batch_test_phonemes.py` | Analysis/visualization |
| `s8_benchmark_inference_and_save.py` / `s9_trace_mlp_classifier.py` | Benchmarking + tracing |
| `s10_onnx_export.py` | Exports Wav2Vec2 (**mean-pooling baked into the ONNX graph itself**, see below) + MLP to ONNX |
| `s11_onnx_test.py` / `s12_overwrite_onnx_unreal.py` | ONNX validation + copy into the Unreal project |

### `workflows/ctc_w2v2_workflow/` and `workflows/ctc_wavlm_workflow/` (identical structure)

| Step | Does |
|---|---|
| `s2_extract_embeddings_temporal.py` | Extracts embeddings **preserving the full temporal sequence** `[T, 768]` (no pooling); writes `phoneme_labels.json` |
| `s3_ctc_classifier.py` | Trains `CTCModel` (2-layer BiLSTM + CTC head) via `PhonemeDataset`/`CTCTrainer`; `torch.utils.data.random_split(0.8/0.2)`, **no fixed seed**; saves `ctc_model_best.pt` + `ctc_label_encoder.pkl` (a plain pickled list, not a fitted sklearn encoder) |
| `s4_visualize_ctc_results.py` / `s5_confusion_analysis.py` | Analysis |
| `s5_export_ctc_onnx.py` | Exports to ONNX + `phoneme_ctc_metadata.json` |
| `s6_batch_test_ctc.py` / `s7_test_onnx.py` | Batch/ONNX testing |

Every file in `s3`–`s6` is **byte-identical** between the w2v2 and wavlm workflows. Only `s2_extract_embeddings_temporal.py` differs (which HF model/checkpoint it loads).

## Model architecture (`workflows/*/models/ctc_model.py`, byte-identical between workflows)

- `CTCModel.forward()` returns `log_probs` shaped `(batch, time, num_classes)`.
- **Blank token = last class index** (`num_classes - 1`). `create_ctc_model(num_classes=N)` internally passes `N + 1` to `CTCModel` - the blank is added on top of the phoneme count, not baked into `phoneme_labels.json` (which lists only the N phoneme strings).
- `phoneme_ctc_metadata.json` (written by the export step) is the only artifact that explicitly stores `blank_token_id`; everything else derives it at runtime as `len(phoneme_labels)`.

## CTC decoding - history and current state

Before 2026-07-10, **three different decode algorithms** coexisted:

1. **Correct greedy decode** (argmax per frame, collapse repeats, drop blank) - `CTCModel._greedy_decode()`, used only for in-process PyTorch training-time validation/visualization/confusion-analysis/CLI tools.
2. **"First non-blank timestep"** - `s7_test_onnx.py`, an arbitrary early-frame bias that ignores the rest of the sequence.
3. **`np.mean(log_probs, axis=0)` then softmax** - `inference/cli/model_loader.py`, used by the CLI, temporal brain, and `evaluation/model_comparison_*.py`. This is the documented "broken decoding" bug: averaging log-probs across time destroys CTC's frame-alignment structure. It's why WavLM-CTC scored 2.78% (~random) in the honest re-evaluation.

**Fixed 2026-07-10**: all three sites now call into `workflows/shared/ctc_decode.py`, the single source of truth. It provides:
- `greedy_ctc_decode()` - the correct collapse-repeats/drop-blank algorithm.
- `ctc_class_scores()` - **peak (max-over-time) log-prob per class**, not a time-average. This is the scoring approach used for ranking/confidence-margin reporting (see the LOSO harness below); it asks "did this class ever appear confidently," which is the right question for a short isolated-phoneme clip.
- `ctc_predict()` - combines both into a top-1 label + a probability vector aligned to `phoneme_labels.json` order (since blank is always the last index, the non-blank classes are exactly `range(len(phoneme_labels))` in order).

## Other bugs found and fixed while building the LOSO harness

- **WavLM embedding extraction used the wrong HF class.** `s2_extract_embeddings_temporal.py` (wavlm workflow) called `Wav2Vec2Processor.from_pretrained("microsoft/wavlm-base")`, which requires a tokenizer config - but `microsoft/wavlm-base` doesn't ship one (WavLM here is audio-only feature extraction, no text decoding). This raises `OSError` immediately, which - because the step has no local try/except - propagates up to `workflow_executor.py`'s catch-and-continue. The workflow would report a failure and move on; every downstream step would then no-op (no `phoneme_embeddings_temporal/` to read). **This is a plausible real explanation for WavLM-CTC's near-random reported accuracy**: if this ran, there was no trained model at all, not a badly-trained one. Fixed to use `Wav2Vec2FeatureExtractor` (works for both `facebook/wav2vec2-base` and `microsoft/wavlm-base`, since only feature extraction is ever used).
- **`ctc_w2v2_workflow/s5_export_ctc_onnx.py` swallowed export failures.** Wrapped in a blanket `try/except` that prints and returns `None` on any error - combined with `workflow_executor.py`'s own catch-and-continue, a failed export could go unnoticed until someone tried to load the resulting (missing or 0-byte) ONNX file much later. Now asserts the output file exists and is non-empty before declaring success, and raises loudly otherwise. Same fix applied to the wavlm workflow's identical copy.
- **`ctc_wavlm_workflow/s5_export_ctc_onnx.py` imported `create_ctc_model` from the w2v2 workflow's module**, not its own (copy/paste artifact). Harmless today since the two `ctc_model.py` files are byte-identical, but corrected to import from its own workflow.
- **A stray, untracked `torchvision==0.19.1+cpu`** was installed in this machine's Poetry venv (not in `pyproject.toml`/`poetry.lock` - leftover from an earlier setup attempt) and is incompatible with the pinned `torch==2.3.1` (`torch.library.register_fake` doesn't exist until torch 2.4+). `transformers` pulls in `torchvision` transitively for an unrelated image-processing code path, so any `Wav2Vec2Processor`/`Wav2Vec2FeatureExtractor` import crashed. Removed via `pip uninstall torchvision` (not a declared dependency, not needed for audio-only use).

## Label ordering

- **MLP workflow**: `phoneme_labels.json` = raw sorted directory names. `label_encoder.pkl` = sklearn `LabelEncoder` fit on the **post-filter observed labels** (classes with <2 samples dropped) - a different source than `phoneme_labels.json`, though both land on alphabetical order in practice. Guarded by a hard `n_classes == 37` assertion.
- **CTC workflows**: `ctc_label_encoder.pkl` is the *same in-memory list* used to write `phoneme_labels.json`, pickled verbatim (not a fitted encoder) - guaranteed consistent by construction, no divergence risk.
- No LOSO/`GroupKFold`/speaker-aware splitting existed anywhere before the harness below. All existing `train_test_split`/`random_split` calls operate on embeddings + phoneme labels only; speaker identity (`child_name` column in `organized_recordings/metadata.csv`) is never read by any split logic.

## Data layout

`recordings/<speaker>/<phoneme>/*.wav` - **speaker-first**, 5 speakers (`callie`, `cassie`, `chloe`, `dan`, `sky`) x 37 phoneme directories each, 6,034 files total. Filename convention: `<speaker>_<phoneme>_ep-<ipa>_<date>_<time>_<take>.wav`.

`workflows/shared/s1_prepare_wav_files.py` merges this into `dist/organized_recordings/<phoneme>/<speaker>_<phoneme>_<counter>.wav` - **phoneme-first**, speaker identity surviving only as a filename prefix and a `metadata.csv` column. `workflows/shared/s0b_augment_audio.py` then mirrors that same phoneme-first structure under `dist/augmented_recordings/aug/...`. **There is no per-speaker directory anywhere downstream of `recordings/`** - this is why the LOSO harness (below) builds its manifest directly from `recordings/`, not from the organized/augmented output.

## The LOSO evaluation harness (`evaluation/harness/`)

Built to implement PRD §5.3. One fit/predict interface for all three models:

- `dataset.py` - manifest built directly from `recordings/` (speaker-first, sidesteps the phoneme-first problem above); `canonical_phoneme_labels()` is a single, fixed label→index mapping computed once and required of every fold/model, closing the label-index-mismatch bug class by construction.
- `embeddings_cache.py` - extracts + caches temporal embeddings once per base model (Wav2Vec2 shared by MLP and Wav2Vec2-CTC; WavLM for WavLM-CTC), resumable, no augmentation (deterministic, cacheable across folds).
- `models.py` - `MLPPhonemeModel` (mean-pooled features -> sklearn MLP, matching production hyperparameters) and `CTCPhonemeModel` (reuses the production `CTCTrainer`/`create_ctc_model` training loop, predicts via the shared `ctc_predict()`).
- `loso_runner.py` - orchestrates the 5-speaker rotation, persists the actual held-out file list per fold (`split_manifest.json` - not just a `random_state`), per-sample predictions with the known-target rank/confidence-margin metric (`predictions.csv`), and a summary with the Chloe headline number and bug-floor flags (`summary.json`).

Results land in `evaluation/loso_results/<run_id>/`.
