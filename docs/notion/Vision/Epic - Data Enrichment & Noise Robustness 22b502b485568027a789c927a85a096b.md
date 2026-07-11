# Epic - Data Enrichment & Noise Robustness

> **2026-07-11 status - partially in flight:** A production-parity augmentation pilot (noise/speed/pitch, Chloe fold) is the first slice of this epic, built as `evaluation/harness/augmentation.py` (run pending - moved to the more powerful machine). SpeechOcean762 remains rejected for now (forcing continuous read-aloud audio into isolated-phoneme shape risks worse artifacts than it fixes - PRD section 3); MUSAN/impulse-response/SNR-sweep work is still future.

Incorporate environment variety, multiple accents, background noises, and augmentation (MUSAN, IRs, SNR sweeps) into training and testing pipelines.

**Delivery Goal (Binary):**

Model maintains accuracy across multiple accents, devices, and noise levels in held-out test sets.

**Kill Criteria (Binary):**

If accuracy drops >5% in noisy or accented conditions vs. clean baseline, stop and retrain with better augmentation/data.

**Personas:**

[Child-ADHD], [Child-Dyslexia], [Child-Autism]

**Metrics:**

[Robustness], [PhonemeAccuracy]

**Capabilities / User Stories:**

- As a child, I can play in a noisy room and still be understood.
- As a developer, I can train with varied voices and environments.

**SIR Sizing:**

- Scope: L
- Impact: High
- Risk: Medium

**What this Epic Delivers:**

Training and testing pipelines include diverse voices, accents, and background conditions using MUSAN, impulse responses, and SNR sweeps.

**Active Tasks:**

- Curate diverse audio sources
- Integrate augmentation in training pipeline
- Test across noise/accent SNR buckets