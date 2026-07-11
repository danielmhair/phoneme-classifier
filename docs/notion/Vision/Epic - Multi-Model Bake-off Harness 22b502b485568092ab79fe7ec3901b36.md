# Epic - Multi-Model Bake-off Harness

> **2026-07-11 status - largely delivered early:** The core of this epic was built 2026-07-10 as `evaluation/harness/` (single fit/predict interface across MLP control, W2V2-CTC, and WavLM-CTC; leave-one-speaker-out evaluation; persisted split manifests; known-target rank/margin metric) - see `plans/prds/07-10-2026-PRD-models-trustworthy.md`. Still open from this epic's scope: latency (p90) and ABX metrics, the Whisper-student model slot (gated on that epic's premise spike), CI/regression automation, and the in-game model selector.

Integrate W2V2→MLP (control), W2V2-CTC, WavLM-CTC, and Distilled Whisper Student into a single interchangeable pipeline for latency, accuracy, and robustness testing.

## Models in the bake‑off (and their jobs)

- **A. Control:** W2V2→MLP (today’s pipeline) — sanity baseline.
- **B. W2V2‑CTC:** safest shippable front end.
- **C. WavLM‑CTC:** likely robustness bump (noise/multi‑speaker).
- **D. Distilled student:** B/C trained with **Whisper teacher labels** for extra generalization.
- **Teacher (Whisper):** *not* in the live loop; used for **distillation**, hard‑case checking, and offline summaries.

**Delivery Goal (Binary):**

Be able to swap and compare all models (control, W2V2-CTC, WavLM-CTC, Whisper-student) in the same test harness.

**Kill Criteria (Binary):**

If harness can’t measure p90 latency, ABX, and accuracy consistently across models, stop and fix before proceeding.

**Personas:**

[Developer], [QA]

**Metrics:**

[PhonemeAccuracy], [LatencyP90], [ABXScore]

**Capabilities / User Stories:**

- As a developer, I can run the same test set through all models and compare metrics.
- As QA, I can verify no regression before model promotion.

**SIR Sizing:**

- Scope: M
- Impact: High
- Risk: Low

**What this Epic Delivers:**

One interface (`ctcPredict`) that all models use, plus scripts to run tests, log results, and generate comparison reports. This is our fair “race track” for picking the best model.

**Active Tasks:**

- Implement model selector in game UI
- Build batch evaluation script (latency, ABX, accuracy)
- Add export logging for test sessions