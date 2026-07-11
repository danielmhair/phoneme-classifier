# Epic - Model Update & Export Pipeline

> **2026-07-11 status:** Unchanged, but its riskiest prerequisite bug is already fixed: ONNX export used to swallow failures silently (a 0-byte file could pass unnoticed); it now fails loudly (2026-07-10). INT8 quantization and one-click retrain-evaluate-export-promote remain future work.

Automate training, retraining, ONNX export (full + INT8), and deployment of models into the bake-off harness with fixed reproducible steps.

**Delivery Goal (Binary):**

One-click process retrains, evaluates, and exports ONNX models (full + INT8) ready for use.

**Kill Criteria (Binary):**

If pipeline can’t produce a ready-to-use ONNX without manual fixes, stop and rework.

**Personas:**

[Developer]

**Metrics:**

[DeploymentSpeed], [ModelStability]

**Capabilities / User Stories:**

- As a developer, I can retrain and deploy new models without manual steps.

**Estimate**

- Scope: M
- Impact: High
- Risk: Medium

**What this Epic Delivers:**

Automated pipeline for retraining, evaluation, ONNX export, quantization, and promotion of models into the bake-off harness.

**Active Tasks:**

- Script training + evaluation steps
- Automate ONNX export + quantization
- Deploy directly to game harness for testing