# Epic - Game for Data Collection

> **2026-07-11 status - PROMOTED, most-justified next epic:** LOSO results (2026-07-10) show the binding constraint is target-age-band data: Chloe (the only age-7 speaker) scores below the other-four-speaker average on every model. Encouraging detail: her known-target top-3 rate is 78-81%, so a game that knows the target phoneme and accepts top-3-with-margin is likely playable enough to collect data today. Supervised recording sessions (like the ones that produced the current dataset) are a cheaper non-game way to start while the game is built.

Design and integrate fun, in-game mini-activities to collect high-quality, labeled phoneme data (isolated sounds and CVC carriers) from children in natural environments.

**Delivery Goal (Binary):**

Collect at least 1–2 hours of clean, labeled kid phoneme data through fun gameplay.

**Kill Criteria (Binary):**

If >25% of collected clips are unusable (noise, cut-off, wrong target), rework game design.

**Personas:**

[Child-ADHD], [Child-Dyslexia], [Child-Autism]

**Metrics:**

[DailyEngagement], [PhonemeAccuracy]

**Capabilities / User Stories:**

- As a child, I have fun unlocking rewards by speaking sounds.
- As a developer, I get labeled phoneme audio from real play.

**SIR Sizing:**

- Scope: M
- Impact: High
- Risk: Medium

**What this Epic Delivers:**

Mini-games that embed phoneme prompts naturally into play (isolated sounds, CVC carriers). All attempts are recorded with timestamps, target labels, and live confidence scores.

**Active Tasks:**

- Design engaging mini-game flow
- Integrate phoneme prompts into gameplay
- Ensure logging of all attempts with metadata