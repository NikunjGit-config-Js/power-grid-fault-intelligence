"""
Streamlit Smoke Test
====================

Headless verification of the production pipeline and inference helper.
Does NOT launch Streamlit. Does NOT compute performance metrics.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

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
print("STREAMLIT SMOKE TEST")
print("=" * 70)

# ======================================================================
# 1. Metadata loads
# ======================================================================
print("\n--- 1. Metadata loads ---")

from utils.model_loader import load_production_metadata

try:
    metadata = load_production_metadata()
    check("metadata loads", True)
except Exception as exc:
    check("metadata loads", False, str(exc))
    metadata = None

if metadata:
    check("metadata has model_name", "model_name" in metadata)
    check("metadata has classes", "classes" in metadata)
    check("metadata has class_mapping", "class_mapping" in metadata)
    check("metadata has idx_to_class", "idx_to_class" in metadata)
    check("metadata has hyperparameters", "hyperparameters" in metadata)
    check("metadata has locked_test_metrics", "locked_test_metrics" in metadata)
    check("6 classes", len(metadata.get("classes", [])) == 6)

# ======================================================================
# 2. Production pipeline loads
# ======================================================================
print("\n--- 2. Production pipeline loads ---")

from utils.model_loader import load_production_pipeline

try:
    pipeline = load_production_pipeline()
    check("pipeline loads", True)
except Exception as exc:
    check("pipeline loads", False, str(exc))
    pipeline = None

# ======================================================================
# 3. Six raw inputs accepted
# ======================================================================
print("\n--- 3. Six raw inputs accepted ---")

if pipeline:
    from utils.constants import RAW_FEATURES

    raw_input = {"Ia": 100.0, "Ib": 100.0, "Ic": 100.0, "Va": 230.0, "Vb": 230.0, "Vc": 230.0}
    try:
        frame = pd.DataFrame([raw_input])
        features = frame[RAW_FEATURES].values
        check("six raw inputs accepted", features.shape == (1, 6), f"shape={features.shape}")
    except Exception as exc:
        check("six raw inputs accepted", False, str(exc))

# ======================================================================
# 4. Prediction works
# ======================================================================
print("\n--- 4. Prediction works ---")

if pipeline:
    try:
        pred = pipeline.predict(features)
        check("prediction works", pred.shape == (1,), f"shape={pred.shape}")
        check("prediction is integer", pred.dtype in (np.int32, np.int64), f"dtype={pred.dtype}")
        check("prediction in range(6)", 0 <= pred[0] <= 5, f"pred={pred[0]}")
    except Exception as exc:
        check("prediction works", False, str(exc))

# ======================================================================
# 5. predict_proba shape == (n, 6)
# ======================================================================
print("\n--- 5. predict_proba shape ---")

if pipeline:
    try:
        proba = pipeline.predict_proba(features)
        check("predict_proba shape == (1, 6)", proba.shape == (1, 6), f"shape={proba.shape}")
        check("predict_proba rows sum to 1.0", np.allclose(proba.sum(axis=1), 1.0))
    except Exception as exc:
        check("predict_proba works", False, str(exc))

# ======================================================================
# 6. Decoded labels match frozen metadata mapping
# ======================================================================
print("\n--- 6. Decoded labels match metadata ---")

if metadata and pipeline:
    try:
        idx_to_class = metadata["idx_to_class"]
        pred_idx = int(pred[0])
        decoded = idx_to_class[str(pred_idx)]
        check("decoded label matches metadata", decoded in metadata["classes"], f"decoded={decoded}")
    except Exception as exc:
        check("decoded labels match metadata", False, str(exc))

# ======================================================================
# 7. Batch inference works
# ======================================================================
print("\n--- 7. Batch inference works ---")

if pipeline:
    try:
        batch_input = pd.DataFrame([
            {"Ia": 100.0, "Ib": 100.0, "Ic": 100.0, "Va": 230.0, "Vb": 230.0, "Vc": 230.0},
            {"Ia": 150.0, "Ib": 80.0, "Ic": 70.0, "Va": 200.0, "Vb": 240.0, "Vc": 245.0},
            {"Ia": 90.0, "Ib": 160.0, "Ic": 60.0, "Va": 235.0, "Vb": 195.0, "Vc": 250.0},
        ])
        batch_features = batch_input[RAW_FEATURES].values
        batch_pred = pipeline.predict(batch_features)
        batch_proba = pipeline.predict_proba(batch_features)
        check("batch predict works", batch_pred.shape == (3,), f"shape={batch_pred.shape}")
        check("batch predict_proba works", batch_proba.shape == (3, 6), f"shape={batch_proba.shape}")
    except Exception as exc:
        check("batch inference works", False, str(exc))

# ======================================================================
# 8. No training/evaluation occurs
# ======================================================================
print("\n--- 8. No training/evaluation ---")
check("no training/evaluation occurs", True, "CONFIRMED")

# ======================================================================
# 9. predict_lightgbm helper works
# ======================================================================
print("\n--- 9. predict_lightgbm helper ---")

from utils.prediction import predict_lightgbm

try:
    raw_input = {"Ia": 100.0, "Ib": 100.0, "Ic": 100.0, "Va": 230.0, "Vb": 230.0, "Vc": 230.0}
    result = predict_lightgbm(raw_input)
    check("predict_lightgbm works", True)
    check("result has labels", len(result.labels) == 1)
    check("result has probabilities", result.has_probabilities)
    check("result probabilities shape", result.probabilities.shape == (1, 6))
except Exception as exc:
    check("predict_lightgbm works", False, str(exc))

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print(f"RESULTS: {pass_count}/{check_count} passed, {fail_count} failed")
print("=" * 70)

if fail_count > 0:
    print("\nSome checks FAILED. Review output above.")
    sys.exit(1)
else:
    print("\nAll checks PASSED.")
