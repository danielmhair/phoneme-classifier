# Epic - Evaluation & Progress Gates

> **2026-07-11 status - partially absorbed:** The model-evaluation half is largely delivered by `evaluation/harness/` (one-command LOSO run, accuracy + known-target margin, persisted split manifests). Not yet built from this epic's scope: latency (p90) and ABX metrics, CI integration, and formal no-regression promotion gates. The child-progress-gates half is untouched product work.

Implement the test harness (latency, ABX, GoP, phoneme clarity metrics) and define the “no regression” rules for promoting new models.

## **Delivery Goal:**

Fixed test harness measures latency, ABX, and accuracy; models can’t ship if metrics regress.

## **Kill Criteria:**

If harness can’t detect regressions or tests become unreliable, stop and fix before promotion.

## **Personas:**

[Developer], [QA]

## **Metrics:**

[PhonemeAccuracy], [LatencyP90], [ABXScore]

## **Capabilities / User Stories:**

- As a developer, I can run one command to evaluate a model.

## **Context:**

- Scope: M
- Impact: High
- Risk: Low

## **What this Epic Delivers:**

Automated test harness with defined “no regression” gates for latency, phoneme accuracy, and ABX minimal-pair discrimination.

## **Active Tasks:**

- Build evaluation scripts and reports
- Set baseline metrics per model
- Integrate into CI/CD pipeline