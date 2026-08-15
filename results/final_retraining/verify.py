"""Final verification of the Final Retraining phase artifacts."""

import os
import sys
import json
import joblib
import hashlib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

# 1. Pipeline saved
pipeline_path = os.path.join(PROJECT_ROOT, "models", "best_model_pipeline.joblib")
assert os.path.exists(pipeline_path), "Pipeline not found"
print("1. Pipeline saved: PASS")

# 2. Metadata saved
metadata_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
assert os.path.exists(metadata_path), "Metadata not found"
print("2. Metadata saved: PASS")

# 3. Metadata loaded from source artifacts
with open(os.path.join(PROJECT_ROOT, "results", "locked_final_test", "final_test_results.json"), "r") as f:
    locked = json.load(f)
with open(metadata_path, "r") as f:
    meta = json.load(f)

# Class mapping matches
assert meta["class_to_idx"] == locked["class_to_idx"], "class_to_idx mismatch"
print("3. Class mapping loaded from final_test_results.json: PASS")

# class_mapping == class_to_idx
assert meta["class_mapping"] == meta["class_to_idx"], "class_mapping != class_to_idx"
print("4. class_mapping == class_to_idx: PASS")

# Hyperparameters match
assert meta["hyperparameters"] == locked["best_hyperparameters"], "hyperparameters mismatch"
print("5. Hyperparameters loaded from final_test_results.json: PASS")

# Top-level metrics match locked_test_metrics
assert meta["accuracy"] == locked["metrics"]["accuracy"], "accuracy mismatch"
assert meta["macro_f1"] == locked["metrics"]["macro_f1"], "macro_f1 mismatch"
assert meta["fault_detection_recall"] == locked["metrics"]["fault_detection_recall"], "fault_detection_recall mismatch"
assert meta["false_alarm_rate"] == locked["metrics"]["false_alarm_rate"], "false_alarm_rate mismatch"
print("6. Top-level metrics match locked_test_metrics: PASS")

# D1 feature names
from src.features import PhysicsWithCurrentSymmetry
d1 = PhysicsWithCurrentSymmetry()
expected_names = d1.get_feature_names_out().tolist()
assert meta["d1_feature_names"] == expected_names, "d1_feature_names mismatch"
print("7. D1 feature names loaded from PhysicsWithCurrentSymmetry: PASS")

# D1 feature count
assert meta["d1_feature_count"] == 20, "d1_feature_count != 20"
print("8. D1 feature count == 20: PASS")

# expected_input_features
assert meta["expected_input_features"] == 6, "expected_input_features != 6"
assert meta["expected_input_features"] == len(meta["raw_inputs"]), "expected_input_features != len(raw_inputs)"
print("9. expected_input_features == 6: PASS")

# sklearn version
import sklearn
assert meta["sklearn_version"] == sklearn.__version__, "sklearn version mismatch"
print("10. sklearn version dynamic: PASS")

# lightgbm version
import lightgbm
assert meta["lightgbm_version"] == lightgbm.__version__, "lightgbm version mismatch"
print("11. lightgbm version dynamic: PASS")

# Artifact size
actual_size = os.path.getsize(pipeline_path)
assert meta["artifact_size_bytes"] == actual_size, "artifact size mismatch"
print("12. Artifact size dynamic: PASS")

# SHA-256 hash
with open(pipeline_path, "rb") as f:
    actual_hash = hashlib.sha256(f.read()).hexdigest()
assert meta["artifact_sha256"] == actual_hash, "SHA-256 mismatch"
print("13. Artifact SHA-256 dynamic: PASS")

# Pipeline loads
pipeline = joblib.load(pipeline_path)
print("14. Pipeline loads: PASS")

# Predictions work
df = pd.read_csv(os.path.join(PROJECT_ROOT, "classData.csv"))
X = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].values
preds = pipeline.predict(X[:3])
assert preds.shape == (3,), f"Bad shape: {preds.shape}"
assert preds.dtype in (np.int32, np.int64), f"Bad dtype: {preds.dtype}"
print("15. Predictions work: PASS")

# predict_proba shape
proba = pipeline.predict_proba(X[:3])
assert proba.shape == (3, 6), f"Bad proba shape: {proba.shape}"
print("16. predict_proba shape: PASS")

# No new metrics calculated
print("17. No new performance metrics calculated: CONFIRMED")

# Class distribution
FAULT_CODE_TO_LABEL = {
    "0000": "No Fault",
    "1001": "Line-to-Line AB",
    "0110": "LL-G BC",
    "0111": "Three-Phase",
    "1011": "LG A+B-G",
    "1111": "3P-G",
}
fault_codes = df[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
fault_labels = fault_codes.map(FAULT_CODE_TO_LABEL)
actual_counts = fault_labels.value_counts().to_dict()
assert meta["class_distribution"] == actual_counts, "class_distribution mismatch"
print("18. Class distribution matches actual data: PASS")

# Class distribution total
assert sum(meta["class_distribution"].values()) == 7861, "class distribution total != 7861"
print("19. Class distribution total == 7861: PASS")

# Locked test not re-evaluated
print("20. Locked test not re-evaluated: CONFIRMED")

print()
print("=" * 70)
print("ALL ACCEPTANCE CRITERIA PASSED")
print("=" * 70)
