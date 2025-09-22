# Task 001: Setup Global Workflow for all the encoders

Define a common workflow skeleton that each encoder uses to build itself

- Have a “startup” + “complete” phases that all share
- Each workflow will have engine-specific middle steps (CTC vs MLP vs Whisper) that are different
- Standardize export artifacts (model + thresholds + stabilizer + vocab + metadata)
- Specify a ctcPredict runner contract and per-engine adapters to run in python
- For MLP Control workflow, ONNX embedding model (W2V2 features) + ONNX MLP classifier will be inside the same ctcPredict adapter

## **Unifying the workflows in python**

**Startup (shared across engines)**

- Step 1 - Cleanup previous runs
- Step 2 - Prepare dataset (organize + augment; save metadata.csv; enforce sample rate; produce silence ref)
- Step 3 - Phoneme labels/vocab normalization
    1. Ensure a single `phoneme_labels.json` and `phoneme_vocab.txt` across workflows.

**Engine-specific tasks**

- Control MLP
    - Already defined
- CTC (W2V2‑CTC / WavLM‑CTC)
    - Prepare CTC training data (targets/timing schema)
    - Fine-tune CTC, checkpoint, threshold calibration per phoneme

**Complete (shared)**

- Create artifacts that prove the model’s accuracy both on good mics and bad mics

## **CTC: build baseline first, add Whisper distillation later**

**Phases**

- Phase A (baseline): train CTC on your current data, ship live ctcPredict with stabilizer/thresholds, hit latency/UX gates, and instrument metrics.
- Phase B (teacher-student epic): add Whisper teacher runners and optional forced alignment/GoP; then add a distillation pass and retune thresholds.

**Why this order works**

- You de-risk live latency and UX early (the hardest product requirement).
- Distillation is an orthogonal quality upgrade; it slots in without changing the app contract.

**Make it easy to add distillation later**

- Keep token/vocab stable from day one (same phoneme inventory, blank).
- Log and persist training/eval manifests so you can re-run with teacher logits/labels later.