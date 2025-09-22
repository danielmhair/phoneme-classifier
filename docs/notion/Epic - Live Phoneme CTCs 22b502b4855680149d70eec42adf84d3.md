# Epic - Live Phoneme CTCs

Build the on-device, low-latency CTC phoneme recognizers (W2V2-CTC and WavLM-CTC) with stabilizer and confidence gating so feedback feels instant and stable.

## **Delivery Goal**

Children can speak into the game and get **instant, stable phoneme feedback** with p90 latency ≤ 150 ms.

This is the “ears” of our system. We’ll take two strong speech encoders — **Wav2Vec2-CTC** and **WavLM-CTC** — and run them right on the device. They’ll produce a continuous stream of phoneme guesses as the child talks. We wrap those guesses with a **temporal stabilizer** so they don’t flicker, and **confidence gating** so the game only reacts when we’re sure enough.

Your current W2V2→MLP pipeline stays in the loop as a **control** so we can see exactly what’s improved. All models plug into the same `ctcPredict` API, so swapping between them is a one-line change.

### `ctcPredict` Contract

This is the unified **on‑device interface** the game calls to get **stabilized phoneme events**. It’s ***not*** a network API; it’s a ***local runner interface*** so any CTC model (W2V2, WavLM, your MLP control) can be swapped behind the same I/O.

A minimal contract (suggested)

- init(model_id | model_loader, config): loads model, thresholds, stabilizer.
- pushAudio(float32 PCM chunk | Int16): streams mic frames in.
- onEvent(callback): emits stabilized phoneme events {phoneme, t_start, t_end, confidence}.
- metrics/logging hooks: timestamps for latency, JSON+audio export toggles.
- reset()/dispose(): lifecycle.

One-line swap means changing the model binding (e.g., model_id or factory) without touching game logic. Offline rescoring/Whisper teacher is a separate service path, not this API.

## **Why it matters:**

Without this live front end, the rest of the system can’t happen. We need an engine that hears clearly, responds fast, and behaves the same way regardless of which underlying model we use — so testing, distillation, and personalization work on a stable base.

## **Kill Criteria**

If latency stays above 150ms or live feedback flickers/misfires in more than 15% of kid trials after stability tuning, we stop and redesign the approach.

## **Personas**

[Persona - Neuro-divergent Child](https://www.notion.so/Persona-Neuro-divergent-Child-22b502b485568055b64cd25ffc228f6a?pvs=21) 

[Persona - Neuro-typical Child](https://www.notion.so/Persona-Neuro-typical-Child-22b502b4855680af857af265c8c1670a?pvs=21) 

[Persona - Parent](https://www.notion.so/Persona-Parent-22b502b48556803e8f03df8798422115?pvs=21)

## **Metrics**

- Phoneme Accuracy
- Latency ≤ 150ms
- Kid Frustration Index

## **Capabilities / User Stories:**

- As a developer, I can swap in different CTC models without changing the logic.

## Estimate

Scope is Large.

Impact is High.

## **Tasks:**

- [Task 001: Setup Global Workflow for all the encoders](Task%20001%20Setup%20Global%20Workflow%20for%20all%20the%20encoder%20251502b4855680829126d97b8f7e20aa.md)
- W2V2-CTC [Spike 001: Minimal Streaming Wav2Vec2 -> CTC Pipeline](Spike%20001%20Minimal%20Streaming%20Wav2Vec2%20-%20CTC%20Pipelin%2024b502b48556813a923ccc94dfd46eec.md)
- Implement `ctcPredict` runner for W2V2-CTC
- Implement `ctcPredict` runner for WavLM-CTC
- Implement `ctcPredict` runner for W2V2-MLP control
- Integrate temporal stabilizer logic (smoothing + hysteresis)
- Integrate confidence gating thresholds
- Build live mic capture → phoneme event pipeline in the game
- Add model selector to the game UI for bake-off testing
- Benchmark latency ≤ 150ms on browser

Sub pages

[Spike 001: Minimal Streaming Wav2Vec2 -> CTC Pipeline](Spike%20001%20Minimal%20Streaming%20Wav2Vec2%20-%20CTC%20Pipelin%2024b502b48556813a923ccc94dfd46eec.md)

Pivot: Python-first ctcPredict harness (not game).

- Build streaming mic capture → phoneme event pipeline in the Python ctcPredict harness
- Add model selector to the harness (CLI/config) for bake-off testing
- Benchmark latency ≤ 150 ms in the Python ctcPredict harness
- As a developer, I can swap engines without changing the harness logic