# Epic - Offline Rescoring & Parent Summaries

Build the CLI/server rescoring worker to process recorded audio through full models and Whisper teacher, producing accuracy summaries and next-day practice packs.

1. **Whisper teacher (Post-game Accuracy brain):** *not* in the live loop. Used to:
    - Re‑score attempts (log‑prob deltas; optional **GoP** via forced alignment),
    - Generate G2P phones for summaries and **distillation**,
    - Calibrate quick in‑game buckets (Great/Almost/Try again) derived from **CTC posteriors**.

**Delivery Goal (Binary):**

Produce daily parent reports and practice packs by rescoring gameplay audio offline.

**Kill Criteria (Binary):**

If rescoring process fails to complete or accuracy metrics can’t be generated reliably, stop and fix before release.

**Personas:**

[Parent], [Developer]

**Metrics:**

[PhonemeAccuracy], [DailyEngagement]

**Capabilities / User Stories:**

- As a parent, I can see my child’s strengths and weaknesses from the day.
- As a developer, I can run rescoring locally or in the cloud.

**SIR Sizing:**

- Scope: L
- Impact: High
- Risk: Medium

**What this Epic Delivers:**

A CLI + optional server worker that replays recorded audio through the full CTC or Whisper teacher, generates detailed accuracy scores, and packages them into parent-friendly summaries with next-day practice packs.

**Active Tasks:**

- Build rescoring CLI tool
- Build server endpoint for Whisper rescoring
- Generate practice pack JSON format