"""
Save Production Pipeline — Verification Script
===============================================

Sanity-check only. Do NOT compute accuracy, F1, recall, false-alarm rate,
ROC-AUC, PR-AUC, or any other new performance metric.

Source of truth:
  - models/model_metadata.json for artifact metadata
  - results/locked_final_test/final_test_results.json for hyperparameters
  - models/best_model_pipeline.joblib for the actual pipeline
"""

import json
import hashlib
import os
import sys

import numpy as np
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_pipeline.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
LOCKED_TEST_PATH = os.path.join(
    PROJECT_ROOT, "results", "locked_final_test", "final_test_results.json"
)

check_count = 0
pass_count = 0
fail_count = 0


def check(name, condition, detail=""):
    global check_count, pass_count, fail_count
    check_count += 1
    if condition:
        pass_count += 1
        print(f"  PASS  {name}")
    else:
        fail_count += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)


print("=" * 70)
print("SAVE PRODUCTION PIPELINE — VERIFICATION")
print("=" * 70)

# ======================================================================
# 1. Artifact existence
# ======================================================================
print("\n--- 1. Artifact existence ---")

check("best_model_pipeline.joblib exists", os.path.exists(PIPELINE_PATH))
check("model_metadata.json exists", os.path.exists(METADATA_PATH))

# ======================================================================
# 2. Load metadata and locked test results
# ======================================================================
print("\n--- 2. Load source artifacts ---")

with open(METADATA_PATH, "r") as f:
    meta = json.load(f)
print("  Loaded model_metadata.json")

with open(LOCKED_TEST_PATH, "r") as f:
    locked = json.load(f)
print("  Loaded final_test_results.json")

# ======================================================================
# 3. Integrity verification
# ======================================================================
print("\n--- 3. Integrity verification ---")

actual_size = os.path.getsize(PIPELINE_PATH)
check(
    "artifact_size_bytes matches",
    meta["artifact_size_bytes"] == actual_size,
    f"metadata={meta['artifact_size_bytes']}, actual={actual_size}",
)

actual_size_mb = actual_size / (1024 * 1024)
check(
    "artifact_size_mb consistent",
    abs(meta["artifact_size_mb"] - round(actual_size_mb, 4)) < 0.001,
    f"metadata={meta['artifact_size_mb']}, actual={round(actual_size_mb, 4)}",
)

with open(PIPELINE_PATH, "rb") as f:
    actual_sha256 = hashlib.sha256(f.read()).hexdigest()
check(
    "SHA-256 matches",
    meta["artifact_sha256"] == actual_sha256,
    f"metadata={meta['artifact_sha256']}, actual={actual_sha256}",
)

# ======================================================================
# 4. Metadata completeness
# ======================================================================
print("\n--- 4. Metadata completeness ---")

check("raw_inputs length == 6", len(meta["raw_inputs"]) == 6)
check("d1_feature_count == 20", meta["d1_feature_count"] == 20)
check("d1_feature_names length == 20", len(meta["d1_feature_names"]) == 20)
check("expected_input_features == 6", meta["expected_input_features"] == 6)
check("class_mapping == class_to_idx", meta["class_mapping"] == meta["class_to_idx"])
check("6 classes present", len(meta["classes"]) == 6)
check("class_to_idx has 6 entries", len(meta["class_to_idx"]) == 6)
check("idx_to_class has 6 entries", len(meta["idx_to_class"]) == 6)

# ======================================================================
# 5. Hyperparameter consistency
# ======================================================================
print("\n--- 5. Hyperparameter consistency ---")

check(
    "hyperparameters match locked test",
    meta["hyperparameters"] == locked["best_hyperparameters"],
    f"metadata={meta['hyperparameters']}, locked={locked['best_hyperparameters']}",
)

# ======================================================================
# 6. Locked test metrics consistency
# ======================================================================
print("\n--- 6. Locked test metrics consistency ---")

for metric in ["accuracy", "macro_f1", "fault_detection_recall", "false_alarm_rate"]:
    check(
        f"{metric} matches locked test",
        meta[metric] == locked["metrics"][metric],
        f"metadata={meta[metric]}, locked={locked['metrics'][metric]}",
    )

# ======================================================================
# 7. Pipeline architecture
# ======================================================================
print("\n--- 7. Pipeline architecture ---")

pipeline = joblib.load(PIPELINE_PATH)
print("  Pipeline loaded successfully")

step_names = [name for name, _ in pipeline.steps]
check("pipeline has steps", len(pipeline.steps) == 2, f"steps={step_names}")

check(
    "PhysicsWithCurrentSymmetry step present",
    any("PhysicsWithCurrentSymmetry" in type(s).__name__ for _, s in pipeline.steps),
    f"step_names={step_names}",
)

check(
    "LGBMClassifier step present",
    any("LGBMClassifier" in type(s).__name__ for _, s in pipeline.steps),
    f"step_names={step_names}",
)

# ======================================================================
# 8. Inference sanity checks
# ======================================================================
print("\n--- 8. Inference sanity checks ---")

# Generate sample raw input (first 5 rows of classData.csv)
import pandas as pd

df = pd.read_csv(os.path.join(PROJECT_ROOT, "classData.csv"))
X_raw = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].values[:5]

preds = pipeline.predict(X_raw)
check("predict() returns array", isinstance(preds, np.ndarray))
check("predict() shape == (5,)", preds.shape == (5,), f"shape={preds.shape}")
check("predict() dtype is integer", preds.dtype in (np.int32, np.int64), f"dtype={preds.dtype}")
check("predicted indices in range(6)", all(0 <= p <= 5 for p in preds), f"preds={preds}")

proba = pipeline.predict_proba(X_raw)
check("predict_proba() shape == (5, 6)", proba.shape == (5, 6), f"shape={proba.shape}")
check("predict_proba() rows sum to 1.0", np.allclose(proba.sum(axis=1), 1.0))

# Batch inference with more rows
X_batch = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].values[:20]
preds_batch = pipeline.predict(X_batch)
check("batch predict works", preds_batch.shape == (20,), f"shape={preds_batch.shape}")

proba_batch = pipeline.predict_proba(X_batch)
check("batch predict_proba works", proba_batch.shape == (20, 6), f"shape={proba_batch.shape}")

# ======================================================================
# 9. Summary
# ======================================================================
print("\n" + "=" * 70)
print(f"RESULTS: {pass_count}/{check_count} passed, {fail_count} failed")
print("=" * 70)

if fail_count > 0:
    print("\nSome checks FAILED. Review output above.")
    sys.exit(1)
else:
    print("\nAll checks PASSED.")
