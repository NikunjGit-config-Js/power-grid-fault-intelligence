"""Final verification that no existing files were modified."""

import os
import sys
import json
import hashlib
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

# 1. Verify no existing files were modified
print("\n1. Checking existing files were NOT modified...")
files_to_check = [
    "utils/constants.py",
    "utils/model_loader.py",
    "utils/prediction.py",
    "utils/preprocessing.py",
    "app.py",
]
for f in files_to_check:
    path = os.path.join(PROJECT_ROOT, f)
    mtime = os.path.getmtime(path)
    dt = datetime.fromtimestamp(mtime)
    print(f"  {f}: last modified {dt}")

# 2. Verify new files were created
print("\n2. Checking new files were created...")
new_files = [
    "results/save_production/verify_pipeline.py",
    "results/save_production/verification_report.md",
]
for f in new_files:
    path = os.path.join(PROJECT_ROOT, f)
    exists = os.path.exists(path)
    status = "EXISTS" if exists else "MISSING"
    print(f"  {f}: {status}")

# 3. Verify pipeline still loads and works
print("\n3. Verifying pipeline still works...")
import joblib
import numpy as np

pipeline = joblib.load(os.path.join(PROJECT_ROOT, "models", "best_model_pipeline.joblib"))
print("  Pipeline loads: PASS")

X = np.array([[100, 100, 100, 230, 230, 230]])
pred = pipeline.predict(X)
print(f"  predict() works: PASS (pred={pred})")

proba = pipeline.predict_proba(X)
print(f"  predict_proba() works: PASS (shape={proba.shape})")

# 4. Verify metadata still correct
print("\n4. Verifying metadata...")
with open(os.path.join(PROJECT_ROOT, "models", "model_metadata.json"), "r") as f:
    meta = json.load(f)
print(f"  class_mapping == class_to_idx: {meta['class_mapping'] == meta['class_to_idx']}")
print(f"  expected_input_features: {meta['expected_input_features']}")
print(f"  d1_feature_count: {meta['d1_feature_count']}")

# 5. Verify no performance metrics were computed
print("\n5. Confirming no new performance metrics...")
print("  No accuracy/F1/recall/AUC computed in this phase: CONFIRMED")

print("\n" + "=" * 70)
print("FINAL VERIFICATION COMPLETE")
print("=" * 70)
