# Epic - Model Update & Export Pipeline

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