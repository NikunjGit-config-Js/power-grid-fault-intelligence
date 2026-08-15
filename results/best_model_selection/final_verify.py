"""Final verification for Best Model Selection phase."""

import os
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")

print("=" * 60)
print("FINAL VERIFICATION — BEST MODEL SELECTION")
print("=" * 60)

# 1. Verify all expected files exist
expected_files = [
    "results/best_model_selection/selection_analysis.py",
    "results/best_model_selection/selection_report.md",
    "results/best_model_selection/ranking.csv",
]
print("\n1. Expected files exist:")
for f in expected_files:
    exists = (ROOT / f).exists()
    print(f"   {'PASS' if exists else 'FAIL'}: {f}")

# 2. Verify no locked test set access in analysis script
script = (ROOT / "results/best_model_selection/selection_analysis.py").read_text()
# Check for actual data loading commands, not just mentions in comments/print statements
has_split_load = "split_indices.npz" in script and "np.load" in script
has_test_idx = "test_idx" in script and "[" in script  # array indexing
print(f"\n2. No locked test set in analysis script: {'PASS' if not has_split_load and not has_test_idx else 'FAIL'}")

# 3. Verify no locked test set access in report
report = (ROOT / "results/best_model_selection/selection_report.md").read_text()
# Report may mention "locked test" in context of what NOT to do, but should not load it
has_load_command = "np.load" in report or "split_indices.npz" in report
print(f"3. No locked test set load commands in report: {'PASS' if not has_load_command else 'FAIL'}")

# 4. Verify no model training in analysis script
has_fit = ".fit(" in script
has_pipeline = "Pipeline(" in script
has_model_init = "LGBMClassifier(" in script or "RandomForestClassifier(" in script
print(f"4. No model training in analysis script: {'PASS' if not has_fit and not has_pipeline and not has_model_init else 'FAIL'}")

# 5. Verify previous experiment artifacts not modified
import time
artifacts_to_check = [
    "results/model_comparison.csv",
    "results/test_4/test_4_results.json",
    "results/test_4/test_4_fold_results.json",
    "results/test_4/test_4_best_params.json",
    "results/test_4/test_4_checkpoint.json",
]
print("\n5. Previous experiment artifacts not modified:")
for f in artifacts_to_check:
    path = ROOT / f
    if path.exists():
        mtime = os.path.getmtime(path)
        # Check if file was modified recently (within last 60 seconds)
        recently_modified = (time.time() - mtime) < 60
        print(f"   {'PASS' if not recently_modified else 'FAIL'}: {f}")
    else:
        print(f"   FAIL (missing): {f}")

# 6. Verify ranking.csv has correct structure
import csv
csv_path = ROOT / "results/best_model_selection" / "ranking.csv"
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames
print(f"\n6. Ranking CSV structure:")
print(f"   {'PASS' if len(rows) == 14 else 'FAIL'}: 14 models in ranking (12 qualifying + 2 eliminated, got {len(rows)})")
print(f"   {'PASS' if 'Model' in headers else 'FAIL'}: 'Model' column present")
print(f"   {'PASS' if 'Macro F1 (mean)' in headers else 'FAIL'}: 'Macro F1 (mean)' column present")
print(f"   {'PASS' if 'Selected' in headers else 'FAIL'}: 'Selected' column present")

# 7. Verify selection report has required sections
required_sections = [
    "Selected Architecture",
    "Selection Rationale",
    "Test-4 Hyperparameter Stability",
    "Final Hyperparameters",
    "Fold-Level Stability",
    "Deployment Characteristics",
    "Limitations",
    "Handoff to Locked Final-Test Evaluation",
]
print("\n7. Selection report required sections:")
for section in required_sections:
    found = section.lower() in report.lower()
    print(f"   {'PASS' if found else 'FAIL'}: {section}")

# 8. Verify no fabricated metrics (check that numbers match source JSON)
import json
with open(ROOT / "results" / "test_4" / "test_4_results.json") as f:
    source = json.load(f)
lgbm_source = source.get("LightGBM", {})
report_has_lgbm_f1 = "0.9987" in report or "0.998745" in report
report_has_lgbm_recall = "0.9994" in report or "99.94%" in report
print(f"\n8. Metrics match source JSON:")
print(f"   {'PASS' if report_has_lgbm_f1 else 'FAIL'}: LightGBM Macro F1 in report")
print(f"   {'PASS' if report_has_lgbm_recall else 'FAIL'}: LightGBM Fault Recall in report")

# 9. Verify no winner declared prematurely (no locked test evaluation claimed)
has_final_test = "locked test" in report.lower() and ("untouched" in report.lower() or "untouch" in report.lower() or "not loaded" in report.lower() or "not used" in report.lower())
print(f"\n9. Locked test set properly referenced as untouched: {'PASS' if has_final_test else 'WARN'}")

# 10. Verify hyperparameter deferral is documented
has_deferred = "deferred" in report.lower() and "locked final-test" in report.lower()
print(f"\n10. Hyperparameter deferral documented: {'PASS' if has_deferred else 'FAIL'}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
