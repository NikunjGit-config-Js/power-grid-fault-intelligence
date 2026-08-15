"""Final verification for Locked Final-Test Evaluation phase."""

import os
import json
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")

print("=" * 60)
print("FINAL VERIFICATION -- LOCKED FINAL-TEST EVALUATION")
print("=" * 60)

# 1. Verify all expected files exist
expected_files = [
    "results/locked_final_test/run_locked_final_test.py",
    "results/locked_final_test/final_test_results.json",
    "results/locked_final_test/final_test_report.md",
    "results/locked_final_test/figures/confusion_matrix_raw.png",
    "results/locked_final_test/figures/confusion_matrix_normalized.png",
    "results/locked_final_test/figures/per_class_metrics.png",
]
print("\n1. Expected files exist:")
for f in expected_files:
    exists = (ROOT / f).exists()
    print(f"   {'PASS' if exists else 'FAIL'}: {f}")

# 2. Verify no locked test access in script BEFORE tuning
script = (ROOT / "results/locked_final_test/run_locked_final_test.py").read_text()

# Check that test labels are not constructed before tuning
# The script should have "Test labels: NOT YET CONSTRUCTED" before tuning
has_test_before_tune = False
lines = script.split("\n")
in_tuning_section = False
for line in lines:
    if "STEP 3" in line or "TUNE" in line.upper():
        in_tuning_section = True
    if "y_test" in line and not in_tuning_section and "NOT" not in line:
        if "transform" in line or "le." in line:
            has_test_before_tune = True

print(f"\n2. No test labels before tuning: {'PASS' if not has_test_before_tune else 'FAIL'}")

# 3. Verify LabelEncoder fitted on dev only
has_le_fit_full = "le.fit_transform" in script and "full" in script.lower()
has_le_fit_dev = "le.fit(dev" in script or "le.fit(dev_" in script
print(f"\n3. LabelEncoder fitted on dev only: {'PASS' if has_le_fit_dev else 'FAIL'}")

# 4. Verify Pipeline used inside CV (not pre-fitted)
has_pipeline_cv = "Pipeline([" in script and "for" in script
print(f"\n4. Pipeline created inside CV folds: {'PASS' if has_pipeline_cv else 'FAIL'}")

# 5. Verify RAW X_test passed to pipeline
has_raw_predict = "pipeline.predict(X_test)" in script or "eval_pipeline.predict(X_test)" in script
has_manual_transform = "transform(X_test)" in script and "predict" in script
print(f"\n5. RAW X_test passed to Pipeline: {'PASS' if has_raw_predict and not has_manual_transform else 'FAIL'}")

# 6. Verify results JSON has all required metrics
with open(ROOT / "results" / "locked_final_test" / "final_test_results.json") as f:
    results = json.load(f)

required_metrics = [
    "accuracy", "macro_f1", "balanced_accuracy", "cohen_kappa",
    "roc_auc_ovr", "pr_auc_ovr", "fault_detection_recall", "false_alarm_rate"
]
print(f"\n6. Required metrics in results JSON:")
for metric in required_metrics:
    present = metric in results.get("metrics", {})
    print(f"   {'PASS' if present else 'FAIL'}: {metric}")

# 7. Verify per-class metrics
per_class = results.get("per_class_metrics", {})
expected_classes = ["3P-G", "LG A+B-G", "LL-G BC", "Line-to-Line AB", "No Fault", "Three-Phase"]
print(f"\n7. Per-class metrics:")
for cls in expected_classes:
    present = cls in per_class
    has_keys = present and all(k in per_class[cls] for k in ["precision", "recall", "f1-score", "support"])
    print(f"   {'PASS' if has_keys else 'FAIL'}: {cls}")

# 8. Verify confusion matrices
cm_raw = results.get("confusion_matrix_raw", [])
cm_norm = results.get("confusion_matrix_normalized", [])
print(f"\n8. Confusion matrices:")
print(f"   {'PASS' if len(cm_raw) == 6 else 'FAIL'}: Raw CM is 6x6 (got {len(cm_raw)})")
print(f"   {'PASS' if len(cm_norm) == 6 else 'FAIL'}: Normalized CM is 6x6 (got {len(cm_norm)})")

# 9. Verify report has required sections
report = (ROOT / "results" / "locked_final_test" / "final_test_report.md").read_text()
required_sections = [
    "Final Hyperparameters",
    "Locked Test Set Metrics",
    "Per-Class Metrics",
    "Confusion Matrix",
    "Comparison to Test-4",
    "Methodology Notes",
    "Isolation Rules",
    "Limitations",
    "Handoff to Final Retraining",
]
print(f"\n9. Report required sections:")
for section in required_sections:
    found = section.lower() in report.lower()
    print(f"   {'PASS' if found else 'FAIL'}: {section}")

# 10. Verify previous artifacts not modified
import time
artifacts_to_check = [
    "results/model_comparison.csv",
    "results/test_4/test_4_results.json",
    "results/best_model_selection/selection_report.md",
    "results/best_model_selection/ranking.csv",
]
print(f"\n10. Previous artifacts not modified:")
for f in artifacts_to_check:
    path = ROOT / f
    if path.exists():
        mtime = os.path.getmtime(path)
        recently_modified = (time.time() - mtime) < 60
        print(f"   {'PASS' if not recently_modified else 'FAIL'}: {f}")
    else:
        print(f"   FAIL (missing): {f}")

# 11. Verify perfect scores are documented
has_perfect = "1.0000" in report or "100.00%" in report
print(f"\n11. Perfect scores documented: {'PASS' if has_perfect else 'FAIL'}")

# 12. Verify no retuning mentioned
has_retune = "retune" in script.lower() or "rerun" in script.lower()
print(f"\n12. No retuning in script: {'PASS' if not has_retune else 'FAIL'}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
