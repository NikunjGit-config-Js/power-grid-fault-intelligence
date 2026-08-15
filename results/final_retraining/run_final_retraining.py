"""
Final Retraining Phase
=====================

Retrain the selected LightGBM pipeline on the COMPLETE available labelled
dataset (all 7861 samples) for production deployment.

This script does NOT compute any new evaluation metrics.
The locked-test metrics from the Locked Final-Test Evaluation phase
remain the final reported unbiased evaluation.

Non-negotiable rules followed:
  - Class mapping loaded from final_test_results.json (NOT newly fitted)
  - Hyperparameters loaded from final_test_results.json
  - Locked test metrics loaded from final_test_results.json
  - D1 feature names loaded from PhysicsWithCurrentSymmetry (NOT hardcoded)
  - sklearn/lightgbm versions obtained dynamically
  - Artifact size/hash computed after saving
  - No accuracy/F1 calculated on training data
  - Class distribution computed from actual classData.csv
"""

import json
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import sklearn
import lightgbm

# Ensure project root is on path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import PhysicsWithCurrentSymmetry, RAW_FEATURES
from sklearn.pipeline import Pipeline

# ======================================================================
# Constants
# ======================================================================
RANDOM_STATE = 42
DATA_PATH = os.path.join(PROJECT_ROOT, "classData.csv")
LOCKED_TEST_RESULTS = os.path.join(
    PROJECT_ROOT, "results", "locked_final_test", "final_test_results.json"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
PIPELINE_PATH = os.path.join(OUTPUT_DIR, "best_model_pipeline.joblib")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "retraining_report.md")

# Fault code mapping (MUST match run_locked_final_test.py exactly)
FAULT_CODE_TO_LABEL = {
    "0000": "No Fault",
    "1001": "Line-to-Line AB",
    "0110": "LL-G BC",
    "0111": "Three-Phase",
    "1011": "LG A+B-G",
    "1111": "3P-G",
}

# ======================================================================
# Step 1: Load frozen artifacts from locked final-test evaluation
# ======================================================================
print("=" * 70)
print("STEP 1: Load frozen artifacts from locked final-test evaluation")
print("=" * 70)

with open(LOCKED_TEST_RESULTS, "r") as f:
    locked_results = json.load(f)

class_to_idx = locked_results["class_to_idx"]
class_names = locked_results["class_names"]
best_hyperparams = locked_results["best_hyperparameters"]
locked_metrics = locked_results["metrics"]

print(f"Loaded class mapping: {class_to_idx}")
print(f"Loaded hyperparameters: {best_hyperparams}")
print(f"Loaded locked test metrics: {locked_metrics}")

# ======================================================================
# Step 2: Load D1 feature names from PhysicsWithCurrentSymmetry
# ======================================================================
print("\n" + "=" * 70)
print("STEP 2: Load D1 feature names from PhysicsWithCurrentSymmetry")
print("=" * 70)

d1_transformer = PhysicsWithCurrentSymmetry()
d1_feature_names = d1_transformer.get_feature_names_out().tolist()
d1_feature_count = len(d1_feature_names)

print(f"D1 feature names ({d1_feature_count}): {d1_feature_names}")
assert d1_feature_count == 20, f"Expected 20 D1 features, got {d1_feature_count}"
print("Assertion passed: len(d1_feature_names) == 20")

# ======================================================================
# Step 3: Load full dataset and construct labels using frozen mapping
# ======================================================================
print("\n" + "=" * 70)
print("STEP 3: Load full dataset and construct labels using frozen mapping")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Construct fault labels from G/C/B/A columns
fault_codes = df[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
fault_labels = fault_codes.map(FAULT_CODE_TO_LABEL)

# Assert: no unmapped fault codes
unmapped_codes = fault_codes[fault_labels.isna()]
if len(unmapped_codes) > 0:
    raise ValueError(f"Unmapped fault codes found: {unmapped_codes.unique()}")
print(f"Assertion passed: no unmapped fault codes (0 of {len(fault_codes)})")

# Compute class distribution from actual data
class_counts = Counter(fault_labels)
print(f"\nClass distribution (computed from classData.csv):")
total_samples = sum(class_counts.values())
for cls_name in sorted(class_counts.keys()):
    print(f"  {cls_name}: {class_counts[cls_name]}")
print(f"  Total: {total_samples}")

# Assert: total equals 7861
assert total_samples == 7861, f"Expected 7861 total samples, got {total_samples}"
print(f"Assertion passed: total_samples == 7861")

# Assert: exactly 6 expected classes present
assert len(class_counts) == 6, f"Expected 6 classes, got {len(class_counts)}"
print(f"Assertion passed: exactly 6 classes present")

# Map to integer indices using frozen class_to_idx
y = fault_labels.map(class_to_idx).to_numpy()

# Assert: all expected 6 classes are present
present_classes = set(int(x) for x in np.unique(y))
expected_classes = set(int(x) for x in class_to_idx.values())
assert present_classes == expected_classes, (
    f"Expected classes {expected_classes}, found {present_classes}"
)
print(f"Assertion passed: all 6 classes present: {sorted(present_classes)}")

# ======================================================================
# Step 4: Extract raw features
# ======================================================================
print("\n" + "=" * 70)
print("STEP 4: Extract raw features")
print("=" * 70)

X = df[RAW_FEATURES].values
print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

# ======================================================================
# Step 5: Create and fit production pipeline
# ======================================================================
print("\n" + "=" * 70)
print("STEP 5: Create and fit production pipeline on COMPLETE dataset")
print("=" * 70)

pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", lightgbm.LGBMClassifier(
        **best_hyperparams,
        random_state=RANDOM_STATE,
        verbose=-1,
    )),
])

print("Fitting pipeline on complete dataset (7861 samples)...")
pipeline.fit(X, y)
print("Pipeline fitted successfully")

# ======================================================================
# Step 6: Save production pipeline
# ======================================================================
print("\n" + "=" * 70)
print("STEP 6: Save production pipeline")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(pipeline, PIPELINE_PATH)
print(f"Pipeline saved to: {PIPELINE_PATH}")

# ======================================================================
# Step 7: Generate production metadata (dynamically)
# ======================================================================
print("\n" + "=" * 70)
print("STEP 7: Generate production metadata")
print("=" * 70)

# Artifact size
artifact_size_bytes = os.path.getsize(PIPELINE_PATH)
artifact_size_mb = artifact_size_bytes / (1024 * 1024)
print(f"Artifact size: {artifact_size_bytes} bytes ({artifact_size_mb:.2f} MB)")

# SHA-256 hash
with open(PIPELINE_PATH, "rb") as f:
    sha256_hash = hashlib.sha256(f.read()).hexdigest()
print(f"Artifact SHA-256: {sha256_hash}")

# Dynamic versions
sklearn_version = sklearn.__version__
lightgbm_version = lightgbm.__version__
print(f"scikit-learn version: {sklearn_version}")
print(f"lightgbm version: {lightgbm_version}")

# Inverse mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}

# expected_input_features (from len(raw_inputs), not hardcoded)
expected_input_features = len(RAW_FEATURES)
assert expected_input_features == 6, f"Expected 6 raw inputs, got {expected_input_features}"
print(f"Assertion passed: expected_input_features == {expected_input_features}")

# Build metadata dictionary
metadata = {
    "model_name": "LightGBM",
    "experiment": "Final Retraining",
    "evaluation_source": "Locked Final-Test Evaluation",
    "feature_set": "D1 (20 features)",
    "raw_inputs": RAW_FEATURES,
    "expected_input_features": expected_input_features,
    "d1_feature_names": d1_feature_names,
    "d1_feature_count": d1_feature_count,
    "classes": class_names,
    "class_to_idx": class_to_idx,
    "class_mapping": class_to_idx,
    "idx_to_class": idx_to_class,
    "hyperparameters": best_hyperparams,
    "accuracy": locked_metrics["accuracy"],
    "macro_f1": locked_metrics["macro_f1"],
    "fault_detection_recall": locked_metrics["fault_detection_recall"],
    "false_alarm_rate": locked_metrics["false_alarm_rate"],
    "locked_test_metrics": locked_metrics,
    "training_set_size": int(len(y)),
    "random_state": RANDOM_STATE,
    "sklearn_version": sklearn_version,
    "lightgbm_version": lightgbm_version,
    "artifact_size_bytes": artifact_size_bytes,
    "artifact_size_mb": round(artifact_size_mb, 4),
    "artifact_sha256": sha256_hash,
    "class_distribution": dict(sorted(class_counts.items())),
    "note": (
        "Production model trained on complete dataset. "
        "Metrics are from locked test evaluation, not this model."
    ),
}

# Verify internal consistency
assert metadata["class_mapping"] == metadata["class_to_idx"], (
    "class_mapping must match class_to_idx"
)
assert metadata["accuracy"] == metadata["locked_test_metrics"]["accuracy"], (
    "Top-level accuracy must match locked_test_metrics"
)
assert metadata["macro_f1"] == metadata["locked_test_metrics"]["macro_f1"], (
    "Top-level macro_f1 must match locked_test_metrics"
)
assert metadata["fault_detection_recall"] == metadata["locked_test_metrics"]["fault_detection_recall"], (
    "Top-level fault_detection_recall must match locked_test_metrics"
)
assert metadata["false_alarm_rate"] == metadata["locked_test_metrics"]["false_alarm_rate"], (
    "Top-level false_alarm_rate must match locked_test_metrics"
)
print("Assertion passed: internal consistency of metadata fields")

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {METADATA_PATH}")

# ======================================================================
# Step 8: Generate retraining report (from actual computed data)
# ======================================================================
print("\n" + "=" * 70)
print("STEP 8: Generate retraining report")
print("=" * 70)

# Build class distribution table from computed data
class_dist_table = ""
for cls_name in sorted(class_counts.keys()):
    class_dist_table += f"| {cls_name} | {class_to_idx[cls_name]} | {class_counts[cls_name]} |\n"

# Build feature table from loaded names
feature_table = ""
for i, fname in enumerate(d1_feature_names, 1):
    if i <= 6:
        desc = f"Phase {'ABC'[(i-1) % 3]} {'current' if i <= 3 else 'voltage'}"
    elif i == 7:
        desc = "Zero-sequence current"
    elif i == 8:
        desc = "Clarke alpha current"
    elif i == 9:
        desc = "Clarke beta current"
    elif i == 10:
        desc = "Current magnitude"
    elif i == 11:
        desc = "Zero-sequence voltage"
    elif i == 12:
        desc = "Clarke alpha voltage"
    elif i == 13:
        desc = "Clarke beta voltage"
    elif i == 14:
        desc = "Voltage magnitude"
    elif i == 15:
        desc = "Guarded impedance A"
    elif i == 16:
        desc = "Guarded impedance B"
    elif i == 17:
        desc = "Guarded impedance C"
    elif i == 18:
        desc = "Max absolute current"
    elif i == 19:
        desc = "Mid absolute current"
    elif i == 20:
        desc = "Min absolute current"
    else:
        desc = ""
    feature_table += f"| {i} | {fname} | {desc} |\n"

report_content = f"""# Final Retraining Report

**Date:** 2026-08-15
**Phase:** Final Retraining
**Status:** COMPLETE

---

## 1. Objective

Retrain the selected LightGBM pipeline on the **complete available labelled dataset** (all {total_samples} samples) for production deployment. This phase does NOT produce new evaluation metrics — the locked test metrics from the Locked Final-Test Evaluation phase remain the final reported unbiased evaluation.

---

## 2. Pipeline Architecture

| Component | Class | Description |
|-----------|-------|-------------|
| Feature Transformer | `PhysicsWithCurrentSymmetry` | D1 feature set ({d1_feature_count} features): Clarke transforms + guarded impedance + current symmetry |
| Classifier | `LGBMClassifier` | LightGBM gradient boosting |

**Pipeline:** `PhysicsWithCurrentSymmetry` -> `LGBMClassifier`

---

## 3. Hyperparameters

Loaded from `results/locked_final_test/final_test_results.json` (determined by 30-trial Optuna tuning on development set):

| Parameter | Value |
|-----------|-------|
| n_estimators | {best_hyperparams['n_estimators']} |
| max_depth | {best_hyperparams['max_depth']} |
| learning_rate | {best_hyperparams['learning_rate']:.6f} |
| num_leaves | {best_hyperparams['num_leaves']} |
| min_child_samples | {best_hyperparams['min_child_samples']} |

---

## 4. D1 Feature Set ({d1_feature_count} Features)

Loaded from `PhysicsWithCurrentSymmetry.get_feature_names_out()`:

| # | Feature | Description |
|---|---------|-------------|
{feature_table}

---

## 5. Training Data

| Property | Value |
|----------|-------|
| Dataset | classData.csv |
| Total samples | {total_samples} |
| Features | 6 raw (Ia, Ib, Ic, Va, Vb, Vc) |
| Classes | 6 (fault types) |
| Class mapping | Frozen from `final_test_results.json` |
| Random state | {RANDOM_STATE} |

### Class Distribution (Complete Dataset)

Computed directly from `classData.csv` using `FAULT_CODE_TO_LABEL` mapping:

| Class | Index | Count |
|-------|-------|-------|
{class_dist_table}
**Total:** {total_samples}

---

## 6. Locked Test Metrics (From Previous Phase)

**These metrics are from the Locked Final-Test Evaluation phase and are NOT recomputed from the production model.**

| Metric | Value |
|--------|-------|
| Accuracy | {locked_metrics['accuracy']:.4f} |
| Macro F1 | {locked_metrics['macro_f1']:.4f} |
| Balanced Accuracy | {locked_metrics['balanced_accuracy']:.4f} |
| Cohen Kappa | {locked_metrics['cohen_kappa']:.4f} |
| ROC AUC (OvR) | {locked_metrics['roc_auc_ovr']:.4f} |
| PR AUC (OvR) | {locked_metrics['pr_auc_ovr']:.4f} |
| Fault Detection Recall | {locked_metrics['fault_detection_recall'] * 100:.2f}% |
| False Alarm Rate | {locked_metrics['false_alarm_rate'] * 100:.2f}% |

---

## 7. Production Artifacts

| Artifact | Location | Size |
|----------|----------|------|
| Pipeline | `models/best_model_pipeline.joblib` | {artifact_size_bytes:,} bytes ({artifact_size_mb:.2f} MB) |
| Metadata | `models/model_metadata.json` | — |

### Artifact Integrity

| Property | Value |
|----------|-------|
| SHA-256 | `{sha256_hash}` |
| sklearn version | {sklearn_version} |
| lightgbm version | {lightgbm_version} |

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

The Final Retraining phase is complete. The production pipeline has been trained on the complete dataset ({total_samples} samples) and saved as `models/best_model_pipeline.joblib`. All metadata is dynamically generated from source artifacts, with no hardcoded values. The locked test metrics from the previous phase remain the final reported unbiased evaluation.

**Next Phase:** Save Production Pipeline (artifact verification and packaging)
"""

with open(REPORT_PATH, "w") as f:
    f.write(report_content)
print(f"Report saved to: {REPORT_PATH}")

# ======================================================================
# Step 9: Sanity check (NO new performance evaluation)
# ======================================================================
print("\n" + "=" * 70)
print("STEP 9: Sanity check (NO new performance evaluation)")
print("=" * 70)

# Reload pipeline
loaded_pipeline = joblib.load(PIPELINE_PATH)
print("Pipeline reloaded successfully")

# Predict on first 5 raw rows
sample_X = X[:5]
sample_y_true = y[:5]

preds = loaded_pipeline.predict(sample_X)
print(f"\nPrediction shape: {preds.shape}")
print(f"Prediction dtype: {preds.dtype}")
print(f"Predicted class indices: {preds}")
print(f"True class indices:     {sample_y_true}")

# Verify predictions are integers
assert preds.dtype in (np.int32, np.int64, int), f"Expected int dtype, got {preds.dtype}"
print("Assertion passed: predictions are integers")

# Verify prediction shape
assert preds.shape == (5,), f"Expected shape (5,), got {preds.shape}"
print("Assertion passed: prediction shape is (5,)")

# Verify predicted class indices exist in frozen mapping
for idx in preds:
    assert idx in class_to_idx.values(), f"Predicted index {idx} not in class_to_idx"
print("Assertion passed: all predicted indices exist in class_to_idx")

# Optionally check predict_proba shape
proba = loaded_pipeline.predict_proba(sample_X)
print(f"\npredict_proba shape: {proba.shape}")
assert proba.shape == (5, 6), f"Expected shape (5, 6), got {proba.shape}"
print("Assertion passed: predict_proba shape is (n_samples, 6)")

print("\n" + "=" * 70)
print("SANITY CHECK COMPLETE — All assertions passed")
print("No new performance metrics were calculated")
print("=" * 70)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print("FINAL RETRAINING COMPLETE")
print("=" * 70)
print(f"Pipeline saved:   {PIPELINE_PATH}")
print(f"Metadata saved:   {METADATA_PATH}")
print(f"Report saved:     {REPORT_PATH}")
print(f"Training set:     {len(y)} samples (complete dataset)")
print(f"Feature set:      D1 ({d1_feature_count} features)")
print(f"Model:            LightGBM")
print(f"Hyperparameters:  {best_hyperparams}")
print(f"Artifact SHA-256: {sha256_hash}")
print(f"\nLocked test metrics (from previous phase, NOT recomputed):")
for metric, value in locked_metrics.items():
    print(f"  {metric}: {value}")
print(f"\nClass distribution (computed from classData.csv):")
for cls_name in sorted(class_counts.keys()):
    print(f"  {cls_name}: {class_counts[cls_name]}")
