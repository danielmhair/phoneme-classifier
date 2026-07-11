# Epic - Whisper Teacher & Distillation

> **2026-07-11 status - premise at risk, spike before committing:** Whisper is a word-level ASR trained overwhelmingly on adult continuous speech; this product's core signal is isolated single phonemes from children - the input Whisper is weakest on (no linguistic context to lean on). Before this epic starts, run a half-day spike: have Whisper label ~50 existing clips and check whether its labels are usable at all. Also note `openai-whisper` was removed from pyproject.toml (2026-07-10, Windows install failure via its unmarked triton dependency) and would need deliberate reintroduction. LOSO results suggest target-age data collection is the higher-leverage robustness play right now.

Use Whisper as a robust, high-accuracy “teacher” model to re-score offline attempts, label training data, and distill knowledge into a smaller, on-device CTC student model.

Whisper acts as our **big teacher**. It re-scores recorded attempts offline, labels noisy/multi-accent audio, and feeds those labels into a smaller CTC student. The student learns both the *answers* (phonemes) and the *confidence shape* from Whisper. This keeps robustness to accents and noise without shipping Whisper live.

## **Delivery Goal**

Have Whisper label and re-score data, and train a **distilled CTC student** that retains ≥95% of Whisper’s phoneme accuracy on test data.

## **Kill Criteria**

If distilled model is >10% slower or loses >5% accuracy vs. baseline CTC after distillation, stop and rework.

## **Personas**

[Child-ADHD], [Child-Dyslexia], [Child-Autism], [Parent]

## **Metrics**

[PhonemeAccuracy], [ABXScore], [Robustness]

## **Capabilities / User Stories**

- As a developer, I can get Whisper to label new audio automatically.
- As a child, I benefit from a more robust model without lag.

## **Estimate**

- Scope: L
- Impact: High
- Risk: High

## **Active Tasks**

- Stand up Whisper teacher API (local/server)
- Integrate G2P to turn transcripts into phonemes
- Train distilled W2V2-CTC and WavLM-CTC students
- Evaluate students against Whisper on held-out test set