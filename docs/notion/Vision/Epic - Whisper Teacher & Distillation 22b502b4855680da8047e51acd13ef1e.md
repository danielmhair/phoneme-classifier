# Epic - Whisper Teacher & Distillation

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