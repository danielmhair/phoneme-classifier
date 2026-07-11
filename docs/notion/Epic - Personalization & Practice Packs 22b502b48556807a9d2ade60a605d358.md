# Epic - Personalization & Practice Packs

> **2026-07-11 status - value up:** LOSO measured a ~9-point gap between the held-out-speaker average (~67%) and Chloe (~58%) - that gap is speaker mismatch, which is exactly what per-child calibration/few-shot adaptation attacks. The "per-child thresholds" slice may be worth pulling forward; it could buy more accuracy per unit effort than any architecture change.

Develop per-child thresholds, adaptive difficulty, and custom daily practice packs based on offline scoring results.

1. **Personalization:** per‑child thresholds and **practice packs** that adapt targets and difficulty daily.

**Delivery Goal (Binary):**

Adjust difficulty, targets, and thresholds daily based on child’s offline scores.

**Kill Criteria (Binary):**

If adjustments don’t improve weak phonemes or cause frustration index to rise, pause and retune.

**Personas:**

[Child-ADHD], [Child-Dyslexia], [Child-Autism], [Parent]

**Metrics:**

[KidFrustrationIndex], [DailyEngagement], [PhonemeAccuracy]

**Capabilities / User Stories:**

- As a child, I practice sounds I struggled with yesterday.
- As a parent, I see the game adapt to my child.

**SIR Sizing:**

- Scope: M
- Impact: High
- Risk: Medium

**What this Epic Delivers:**

Each child’s scores feed into a “practice pack” that sets the next day’s targets, hints, and difficulty. Live thresholds can be personalized per phoneme.

**Active Tasks:**

- Define practice pack schema
- Implement per-child threshold overrides
- Integrate practice pack loader into game