# Epic - Live Streaming Improvements (Temporal Brain)

Implement smoothing techniques, hysteresis controls, and probability-based gating to eliminate flicker and prevent misfires during phoneme detection.

1. **Smoothing** to avoid jittery changes
2. **Hysteresis** (a simple "memory" that stops the output from changing too quickly) to prevent rapid flip‑flopping between phonemes
3. **Confidence gating** (only emitting a phoneme if its probability stays high enough for long enough) so that final phoneme events are stable and come with reliable timestamps.

## **Delivery Goal:**

Live phoneme stream is **stable and accurate** during real-time speech, with <15% flicker rate in kid tests.

This is the “steady hands” for our real-time ear. We take the raw frame-by-frame phoneme guesses from the CTC and **smooth** them over time. We add **hysteresis** so once a phoneme is “locked,” it won’t switch unless another has clearly won for several frames. We also use **confidence gating** to make sure we only show a phoneme when we’re sure enough — avoiding random blips.

## **Kill Criteria:**

If smoothing, hysteresis, and gating fail to keep flicker below 15% without hurting latency, we stop and redesign.

**Personas:**

[Child-ADHD], [Child-Dyslexia], [Child-Autism]

**Metrics:**

[KidFrustrationIndex], [PhonemeAccuracy], [LatencyP90]

## **Capabilities / User Stories:**

- As a child, I hear/see the same answer from the game while I’m still speaking that sound.
- As a parent, I know the game won’t flip between “right” and “wrong” too fast.

## **Estimate:**

- Scope: M
- Impact: High
- Risk: Medium

## **Active Tasks:**

- Implement smoothing algorithm over sliding time window
- Add hysteresis lock/unlock logic
- Tune confidence thresholds per phoneme type
- Test stability in varied noise levels