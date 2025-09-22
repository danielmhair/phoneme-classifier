# Theme: Automatic Phoneme Recognition (APR)

# **Goal**

<aside>
✨

**Build a phoneme‑level ASR  —** a phoneme‑first, low‑latency voice foundation for learning games

</aside>

<aside>
✨

We need a foundational voice layer that’s robust enough to handle kids’ varied voices, accents, and noisy environments, yet light enough to run in real time on-device. This system will support all phoneme-centric educational gameplay in Light Haven products. By combining a small on-device CTC phoneme recognizer with a robust Whisper teacher, we can deliver instant, kind feedback to children, give accurate post-session reports to parents, and continuously improve the model without rebuilding the apps.

</aside>

It feels instant to a child, gives gentle and accurate feedback to caregivers, and is architected so we can swap models, add data, and improve continuously—without redoing the apps on top.

# **Personas**

- [Persona - Neuro-typical Child](https://www.notion.so/Persona-Neuro-typical-Child-22b502b4855680af857af265c8c1670a?pvs=21)
- [Persona - Neuro-divergent Child](https://www.notion.so/Persona-Neuro-divergent-Child-22b502b485568055b64cd25ffc228f6a?pvs=21)
- [Persona - Parent](https://www.notion.so/Persona-Parent-22b502b48556803e8f03df8798422115?pvs=21)

# **Metrics**

- Able to give proper phoneme accuracy
    
    Definition: Online target detection (per‑phoneme precision/recall/F1), Offline PER, and calibration (ECE).
    
    - Measure (online): log target_id, decision (accept/reject), top‑k phonemes + confidences; compute per‑phoneme precision/recall/F1, FAR, FRR.
    - Measure (offline): Phoneme Error Rate (PER) vs teacher/human labels on isolated phones + CVC; compute reliability curve + ECE per phoneme.
    - Targets: F1 ≥ 0.90 (isolated), ≥ 0.85 (CVC); FAR/FRR ≤ 10% each; PER ≤ 20% initial; improve over time with data/tuning.
- **Latency (live):** ≤ 150 ms end‑to‑end.
    
    Definition: Time from first captured frame to stabilized phoneme event (includes VAD, inference, stabilizer/gating).
    
    - Measure: log t_capture_first_frame, t_infer_start/end, t_stabilized_emit; report p50/p90/p95 end‑to‑end latency per device.
    - Targets: p95 ≤ 150 ms; p50 ≤ 100 ms; monitor CPU/RTF for regressions.
- **Kid experience:** “frustration index” ≤ 15% (fails twice on same target).
    
    Definition: Frustration Index (FI) = #targets that require ≥2 consecutive attempts / #unique targets in a session.
    
    - Measure: track per‑target retries and outcomes; also report time‑to‑success to separate slow vs fail.
    - Targets: median FI ≤ 10%, p75 FI ≤ 15%; spikes trigger per‑phoneme threshold tuning.
- ABX ≥ 85% on critical minimal pairs (/b/–/p/, /k/–/g/, /s/–/ʃ/).
    - See [ABX phoneme discriminability](https://www.notion.so/ABX-phoneme-discriminability-24a502b48556806ebadbe417385efd5f?pvs=21)
    
    Definition: ABX on minimal pairs (within/across speaker), evaluated across SNR buckets and accents.
    
    - Measure: build balanced ABX sets per pair; score with posterior/log‑likelihood distance using the same live inference path.
    - Targets: macro ≥ 85% overall and ≥ 80% in hardest SNR/accent buckets.
- **Robustness:** stable across SNR buckets and a small set of accents.
    
    Definition: Performance stability across SNR buckets and accents for Online F1, ABX, PER, and latency.
    
    - Measure: SNR buckets {<5, 5–10, 10–20, >20 dB}; accent groups from pilot; compute metrics per bucket/group.
    - Targets: ABX spread ≤ 5 pp; Online F1 spread ≤ 8 pp; latency p95 spread ≤ 20 ms (same device class).

## What we are building

- **Phoneme Stream:** a live microphone **CTC phoneme stream** on‑device (quantized for speed) with a tiny **temporal stabilizer** and **confidence gating** so feedback never flickers.
    - [Epic - Live Phoneme CTCs](Epic%20-%20Live%20Phoneme%20CTCs%2022b502b4855680149d70eec42adf84d3.md)
    - [Epic - Live Streaming Improvements (Temporal Brain)](Epic%20-%20Live%20Streaming%20Improvements%20(Temporal%20Brain%2022b502b48556801c86f0f3f5a7036010.md)
- **A teacher–student loop:** the robust **Whisper ASR teacher** labels diverse, noisy, multi‑accent speech; our small **CTC student** distills that robustness for on‑device use.
    - [Epic - Whisper Teacher & Distillation](Epic%20-%20Whisper%20Teacher%20&%20Distillation%2022b502b4855680da8047e51acd13ef1e.md)
- **Swap‑able models & pipelines:** identical I/O and exports (ONNX full + INT8) for each candidate so bake‑offs are apples‑to‑apples.
    - [Epic - Multi-Model Bake-off Harness](Epic%20-%20Multi-Model%20Bake-off%20Harness%2022b502b485568092ab79fe7ec3901b36.md)
- **Data Collection through Play:** A **mini‑game** collects short, labeled attempts (isolated phones + CVC carriers), so we learn from natural, fun sessions—not lab scripts.
    - [Epic - Game for Data Collection](Epic%20-%20Game%20for%20Data%20Collection%2022b502b4855680dfa7d6f0c8ea071806.md)
- **Offline scorer:** After play, the same audio is re‑scored by a **full model** and the **Whisper teacher** to produce trustworthy accuracy summaries and next‑day practice plans.
    - [Epic - Offline Rescoring & Parent Summaries](Epic%20-%20Offline%20Rescoring%20&%20Parent%20Summaries%2022b502b485568051b73efd500dd632f3.md)
    - [Epic - Personalization & Practice Packs](Epic%20-%20Personalization%20&%20Practice%20Packs%2022b502b48556807a9d2ade60a605d358.md)
- **Improvements & Future-proofing:** Now that we have a good system in place, lets improve the robustness of the model, and setup a system to continuously improve overtime
    - [Epic - Data Enrichment & Noise Robustness](Epic%20-%20Data%20Enrichment%20&%20Noise%20Robustness%2022b502b485568027a789c927a85a096b.md)
    - [Epic - Evaluation & Progress Gates](Epic%20-%20Evaluation%20&%20Progress%20Gates%2022b502b4855680dcb4f3e071691c4957.md)
    - [Epic - Model Update & Export Pipeline](Epic%20-%20Model%20Update%20&%20Export%20Pipeline%2022b502b485568049af1fe48dcff0d011.md)

# Epics

1. [Epic - Live Phoneme CTCs](Epic%20-%20Live%20Phoneme%20CTCs%2022b502b4855680149d70eec42adf84d3.md)
2. [Epic - Live Streaming Improvements (Temporal Brain)](Epic%20-%20Live%20Streaming%20Improvements%20(Temporal%20Brain%2022b502b48556801c86f0f3f5a7036010.md)
3. [Epic - Whisper Teacher & Distillation](Epic%20-%20Whisper%20Teacher%20&%20Distillation%2022b502b4855680da8047e51acd13ef1e.md) 
4. [Epic - Multi-Model Bake-off Harness](Epic%20-%20Multi-Model%20Bake-off%20Harness%2022b502b485568092ab79fe7ec3901b36.md) 
5. [Epic - Game for Data Collection](Epic%20-%20Game%20for%20Data%20Collection%2022b502b4855680dfa7d6f0c8ea071806.md) 
6. [Epic - Offline Rescoring & Parent Summaries](Epic%20-%20Offline%20Rescoring%20&%20Parent%20Summaries%2022b502b485568051b73efd500dd632f3.md) 
7. [Epic - Personalization & Practice Packs](Epic%20-%20Personalization%20&%20Practice%20Packs%2022b502b48556807a9d2ade60a605d358.md) 
8. [Epic - Data Enrichment & Noise Robustness](Epic%20-%20Data%20Enrichment%20&%20Noise%20Robustness%2022b502b485568027a789c927a85a096b.md) 
9. [Epic - Evaluation & Progress Gates](Epic%20-%20Evaluation%20&%20Progress%20Gates%2022b502b4855680dcb4f3e071691c4957.md) 
10. [Epic - Model Update & Export Pipeline](Epic%20-%20Model%20Update%20&%20Export%20Pipeline%2022b502b485568049af1fe48dcff0d011.md) 

## Why this approach works

- **Kids need quick feedback:** CTC streams let us decide *as audio flows*; no waiting for a full word. The stabilizer and thresholds keep UX kind.
- **Robustness without lag:** we inherit big‑model toughness (accents, noise) via **Whisper-to-CTC distillation**, but ship a tiny student so its fast.
- **Evidence, not vibes:** a **results pipeline** (ABX, log‑prob deltas, GoP/forced alignment) validates scores and tunes thresholds per phoneme.

## Data strategy

- **Micro‑pilot through play:** 10–15 kids, ~1–2 hours total, across **multiple accents**, rooms, and devices. Targets are **phoneme sequences** (isolated phones + short CVC carriers) to keep it phoneme‑centric.
- **Representation matters:** add adult and environment variety (household noise, classroom, cafeteria, people talking next to them normally, or even two people playing next to each other with two laptops?) to teach the Whisper-student to ignore background chaos.
- **Privacy first:** parent consent, local storage by default, encrypted upload for optional cloud rescoring; auto‑delete raw audio on schedule, keep derived scores.

---

## Progress gates (no dates—only definitions of done)

- **P0 – Harness ready:** One `ctcPredict` API, temporal stabilizer, logging, JSON+audio export.
- **P1 – Models plugged in:** A/B/C/D + Whisper teacher runners hooked; latency & ABX probes wired.
- **P2 – Micro‑pilot playable:** Game is fun; collects clean attempts reliably.
- **P3 – Offline scoring & calibration:** Rescoring worker (CLI+server), parent summary, **practice pack** export; optional forced alignment + GoP to benchmark our fast score.
- **P4 – Pick & personalize:** Choose the live engine under the latency cap; enable per‑child thresholds; keep cloud assist for tough cases.
- **P5 – Iterate:** Monthly data adds, retrain distilled student, re‑export models, rerun the same tests.

---

## Risks & mitigations

- **Sparse kid data →** run micro‑pilot + continuous collection; leverage Whisper teacher labels for breadth.
- **Noisy homes/classrooms →** denoise + augmentation + stability rules; cloud assist for edge cases.
- **Model drift →** fixed test harness (latency/ABX/accuracy) on held‑out clips; regressions block promotion.

---

## What “Theme done” looks like

- Apps call one foundation and get **instant, stable phoneme events**.
- Parents get **clear summaries** and **practice packs** overnight.
- We can **swap models**, retrain with new data, and compare results in the same harness—no app rewrites.