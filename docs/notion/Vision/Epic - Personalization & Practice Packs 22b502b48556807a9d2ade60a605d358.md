# Epic - Personalization & Practice Packs

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