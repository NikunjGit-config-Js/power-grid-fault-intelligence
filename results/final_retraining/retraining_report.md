# Final Retraining Report

**Date:** 2026-08-15
**Phase:** Final Retraining
**Status:** COMPLETE

---

## 1. Objective

Retrain the selected LightGBM pipeline on the **complete available labelled dataset** (all 7861 samples) for production deployment. This phase does NOT produce new evaluation metrics — the locked test metrics from the Locked Final-Test Evaluation phase remain the final reported unbiased evaluation.

---

## 2. Pipeline Architecture

| Component | Class | Description |
|-----------|-------|-------------|
| Feature Transformer | `PhysicsWithCurrentSymmetry` | D1 feature set (20 features): Clarke transforms + guarded impedance + current symmetry |
| Classifier | `LGBMClassifier` | LightGBM gradient boosting |

**Pipeline:** `PhysicsWithCurrentSymmetry` -> `LGBMClassifier`

---

## 3. Hyperparameters

Loaded from `results/locked_final_test/final_test_results.json` (determined by 30-trial Optuna tuning on development set):

| Parameter | Value |
|-----------|-------|
| n_estimators | 144 |
| max_depth | 8 |
| learning_rate | 0.274228 |
| num_leaves | 13 |
| min_child_samples | 50 |

---

## 4. D1 Feature Set (20 Features)

Loaded from `PhysicsWithCurrentSymmetry.get_feature_names_out()`:

| # | Feature | Description |
|---|---------|-------------|
| 1 | Ia | Phase A current |
| 2 | Ib | Phase B current |
| 3 | Ic | Phase C current |
| 4 | Va | Phase A voltage |
| 5 | Vb | Phase B voltage |
| 6 | Vc | Phase C voltage |
| 7 | I0 | Zero-sequence current |
| 8 | Ialpha | Clarke alpha current |
| 9 | Ibeta | Clarke beta current |
| 10 | Imag | Current magnitude |
| 11 | V0 | Zero-sequence voltage |
| 12 | Valpha | Clarke alpha voltage |
| 13 | Vbeta | Clarke beta voltage |
| 14 | Vmag | Voltage magnitude |
| 15 | Za | Guarded impedance A |
| 16 | Zb | Guarded impedance B |
| 17 | Zc | Guarded impedance C |
| 18 | I_max | Max absolute current |
| 19 | I_mid | Mid absolute current |
| 20 | I_min | Min absolute current |


---

## 5. Training Data

| Property | Value |
|----------|-------|
| Dataset | classData.csv |
| Total samples | 7861 |
| Features | 6 raw (Ia, Ib, Ic, Va, Vb, Vc) |
| Classes | 6 (fault types) |
| Class mapping | Frozen from `final_test_results.json` |
| Random state | 42 |

### Class Distribution (Complete Dataset)

Computed directly from `classData.csv` using `FAULT_CODE_TO_LABEL` mapping:

| Class | Index | Count |
|-------|-------|-------|
| 3P-G | 0 | 1133 |
| LG A+B-G | 1 | 1134 |
| LL-G BC | 2 | 1004 |
| Line-to-Line AB | 3 | 1129 |
| No Fault | 4 | 2365 |
| Three-Phase | 5 | 1096 |

**Total:** 7861

---

## 6. Locked Test Metrics (From Previous Phase)

**These metrics are from the Locked Final-Test Evaluation phase and are NOT recomputed from the production model.**

| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Macro F1 | 1.0000 |
| Balanced Accuracy | 1.0000 |
| Cohen Kappa | 1.0000 |
| ROC AUC (OvR) | 1.0000 |
| PR AUC (OvR) | 1.0000 |
| Fault Detection Recall | 100.00% |
| False Alarm Rate | 0.00% |

---

## 7. Production Artifacts

| Artifact | Location | Size |
|----------|----------|------|
| Pipeline | `models/best_model_pipeline.joblib` | 581,709 bytes (0.55 MB) |
| Metadata | `models/model_metadata.json` | — |

### Artifact Integrity

| Property | Value |
|----------|-------|
| SHA-256 | `a771efb5f8b72f4b5e14d6734886a249fe31f0d7457f19c07cedbe367ae9c168` |
| sklearn version | 1.8.0 |
| lightgbm version | 4.7.0 |

---

## 8. Sanity Check Results

After saving, the pipeline was reloaded and tested:

| Check | Result |
|-------|--------|
| Pipeline loads | PASS |
| Prediction shape | (n_samples,) — PASS |
| Prediction dtype | int64 — PASS |
| Predicted indices in class_to_idx | PASS |
| predict_proba shape | (n_samples, 6) — PASS |
| **No new performance metrics computed** | CONFIRMED |

---

## 9. Non-Negotiable Rules Compliance

| Rule | Status |
|------|--------|
| Never fabricate metrics | PASS — all metrics loaded from `final_test_results.json` |
| Prevent data leakage | PASS — pipeline fitted on complete dataset, no held-out data |
| No new performance evaluation | PASS — only sanity check, no accuracy/F1 computed |
| Class mapping frozen | PASS — loaded from `final_test_results.json`, not newly fitted |
| Hyperparameters loaded | PASS — loaded from `final_test_results.json` |
| Metadata generated | PASS — all fields computed dynamically or loaded from source |
| D1 feature names loaded | PASS — from `PhysicsWithCurrentSymmetry.get_feature_names_out()` |
| Artifact integrity | PASS — SHA-256 hash computed |

---

## 10. Summary

The Final Retraining phase is complete. The production pipeline has been trained on the complete dataset (7861 samples) and saved as `models/best_model_pipeline.joblib`. All metadata is dynamically generated from source artifacts, with no hardcoded values. The locked test metrics from the previous phase remain the final reported unbiased evaluation.

**Next Phase:** Save Production Pipeline (artifact verification and packaging)
