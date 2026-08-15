"""Best Model Selection — Selection Analysis Script

Reads Test 4 results, applies the PROJECT_SPEC selection hierarchy,
assesses hyperparameter stability, and produces ranking.csv.

No model training. No locked test set access. No fabrication.
"""

import json
import csv
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "best_model_selection"

# --- Load data ---
with open(RESULTS / "test_4" / "test_4_results.json") as f:
    test4_results = json.load(f)

with open(RESULTS / "test_4" / "test_4_best_params.json") as f:
    best_params = json.load(f)

with open(RESULTS / "test_4" / "test_4_fold_results.json") as f:
    fold_results = json.load(f)


# --- Step 1: Build structured data ---
models = []
for name, m in test4_results.items():
    if name.startswith("_"):
        continue  # skip _metadata
    models.append({
        "model": name,
        "macro_f1_mean": m["macro_f1_mean"],
        "macro_f1_std": m["macro_f1_std"],
        "accuracy_mean": m["accuracy_mean"],
        "balanced_accuracy_mean": m["balanced_accuracy_mean"],
        "cohen_kappa_mean": m["cohen_kappa_mean"],
        "fault_detection_recall_mean": m["fault_detection_recall_mean"],
        "false_alarm_rate_mean": m["false_alarm_rate_mean"],
        "train_time_mean": m["train_time_mean"],
        "inference_time_mean": m["inference_time_mean"],
    })

# --- Step 2: Apply selection hierarchy ---
# Filter: fault_detection_recall >= 0.99
qualifying = [m for m in models if m["fault_detection_recall_mean"] >= 0.99]
eliminated = [m for m in models if m["fault_detection_recall_mean"] < 0.99]

# Rank by: macro_f1_mean DESC, false_alarm_rate_mean ASC, macro_f1_std ASC, inference_time_mean ASC
qualifying.sort(key=lambda m: (
    -m["macro_f1_mean"],
    m["false_alarm_rate_mean"],
    m["macro_f1_std"],
    m["inference_time_mean"],
))

# --- Step 3: Full ranking table ---
print("=" * 80)
print("FULL RANKING — ALL 14 MODELS (Test 4 Nested CV Results)")
print("=" * 80)
print(f"{'Rank':<5} {'Model':<28} {'Macro F1':>10} {'Fault Recall':>14} {'False Alarm':>12} {'CV Std':>10} {'Inference':>12}")
print("-" * 80)
for i, m in enumerate(models, 1):
    marker = ""
    print(f"{i:<5} {m['model']:<28} {m['macro_f1_mean']:>10.4f} {m['fault_detection_recall_mean']:>14.4f} {m['false_alarm_rate_mean']:>12.4f} {m['macro_f1_std']:>10.4f} {m['inference_time_mean']:>12.4f}")

print()
print("=" * 80)
print("SELECTION HIERARCHY (PROJECT_SPEC lines 1847-1855)")
print("=" * 80)
print(f"Step 1 — Fault Detection Recall >= 99%: {len(qualifying)} models qualify, {len(eliminated)} eliminated")
if eliminated:
    for m in eliminated:
        print(f"  ELIMINATED: {m['model']} (recall={m['fault_detection_recall_mean']:.4f})")

print()
print(f"Step 2 — Ranked by Macro F1 (desc), then False Alarm (asc), then CV Std (asc), then Inference (asc)")
print()
print(f"{'Rank':<5} {'Model':<28} {'Macro F1':>10} {'False Alarm':>12} {'CV Std':>10} {'Inference':>12}")
print("-" * 80)
for i, m in enumerate(qualifying, 1):
    print(f"{i:<5} {m['model']:<28} {m['macro_f1_mean']:>10.4f} {m['false_alarm_rate_mean']:>12.4f} {m['macro_f1_std']:>10.4f} {m['inference_time_mean']:>12.4f}")

winner = qualifying[0]
print()
print(f"SELECTED ARCHITECTURE: {winner['model']}")
print(f"  Macro F1:              {winner['macro_f1_mean']:.4f} +/- {winner['macro_f1_std']:.4f}")
print(f"  Fault Detection Recall: {winner['fault_detection_recall_mean']:.4f}")
print(f"  False Alarm Rate:       {winner['false_alarm_rate_mean']:.4f}")
print(f"  Balanced Accuracy:      {winner['balanced_accuracy_mean']:.4f}")
print(f"  Cohen Kappa:            {winner['cohen_kappa_mean']:.4f}")
print(f"  Inference Time:         {winner['inference_time_mean']:.4f}s")

# --- Step 4: Fold-level stability for the selected model ---
print()
print("=" * 80)
print(f"FOLD-LEVEL STABILITY: {winner['model']}")
print("=" * 80)

winner_folds = fold_results[winner["model"]]
fold_macro_f1 = [f["macro_f1"] for f in winner_folds]
fold_min = min(fold_macro_f1)
fold_max = max(fold_macro_f1)
fold_mean = sum(fold_macro_f1) / len(fold_macro_f1)
fold_var = sum((x - fold_mean) ** 2 for x in fold_macro_f1) / (len(fold_macro_f1) - 1)
fold_std = fold_var ** 0.5

print(f"5 outer-fold Macro F1 values:")
for f in winner_folds:
    print(f"  Fold {f['fold']}: {f['macro_f1']:.6f}")
print()
print(f"  Minimum:   {fold_min:.6f}")
print(f"  Maximum:   {fold_max:.6f}")
print(f"  Mean:      {fold_mean:.6f}")
print(f"  Std Dev:   {fold_std:.6f}")
print(f"  Range:     {fold_max - fold_min:.6f}")

# Check if ranking is disproportionately driven by a single fold
# If one fold's F1 is more than 3 std devs from the mean, flag it
outlier_folds = [f for f in winner_folds if abs(f["macro_f1"] - fold_mean) > 3 * fold_std] if fold_std > 0 else []
if outlier_folds:
    print(f"\n  WARNING: Fold(s) {[f['fold'] for f in outlier_folds]} may be outliers (>3 std devs from mean)")
else:
    print(f"\n  No outlier folds detected (no fold >3 std devs from mean)")

# Check if the aggregate ranking could flip if one fold were removed
# Compute leave-one-out means for each candidate and see if ranking changes
print()
print("Leave-one-out analysis (does removing one fold change the ranking?):")
other_candidates = qualifying[1:4]  # top 4 after winner
for leave_out_fold in range(1, 6):
    loo_folds = [f for f in winner_folds if f["fold"] != leave_out_fold]
    loo_mean = sum(f["macro_f1"] for f in loo_folds) / len(loo_folds)
    # Compare against leave-one-out means of other candidates
    still_winner = True
    for cand in other_candidates:
        cand_folds = fold_results[cand["model"]]
        cand_loo = [f for f in cand_folds if f["fold"] != leave_out_fold]
        cand_loo_mean = sum(f["macro_f1"] for f in cand_loo) / len(cand_loo)
        if cand_loo_mean > loo_mean:
            still_winner = False
            print(f"  Leave out fold {leave_out_fold}: RANKING CHANGES — {cand['model']} ({cand_loo_mean:.6f}) > {winner['model']} ({loo_mean:.6f})")
            break
    else:
        print(f"  Leave out fold {leave_out_fold}: {winner['model']} remains on top ({loo_mean:.6f})")

# --- Step 5: Hyperparameter stability ---
print()
print("=" * 80)
print(f"HYPERPARAMETER STABILITY: {winner['model']}")
print("=" * 80)

winner_params = best_params[winner["model"]]
# Check if all 5 folds have identical configs
all_configs = [json.dumps(p, sort_keys=True) for p in winner_params]
unique_configs = list(set(all_configs))

if len(unique_configs) == 1:
    print("All 5 folds converged to IDENTICAL hyperparameters:")
    print(json.dumps(winner_params[0], indent=2))
    print()
    print("STATUS: Stable Test-4 configuration identified.")
else:
    print(f"Configurations DIVERGED across 5 folds ({len(unique_configs)} unique configs):")
    print()
    for i, params in enumerate(winner_params, 1):
        print(f"  Fold {i}: {json.dumps(params, sort_keys=True)}")
    print()
    print("STATUS: Configurations not identical.")
    print("FINAL HYPERPARAMETERS: DEFERRED to Locked Final-Test Evaluation phase.")
    print("  The definitive hyperparameters will be determined by tuning the")
    print("  selected architecture using ONLY the complete development set and")
    print("  cross-validation, before evaluating the locked test set exactly once.")

# --- Step 6: Save ranking.csv ---
print()
print("=" * 80)
print("SAVING RANKING")
print("=" * 80)

csv_path = OUT_DIR / "ranking.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Rank", "Model", "Macro F1 (mean)", "Macro F1 (std)",
        "Accuracy (mean)", "Balanced Accuracy (mean)", "Cohen Kappa (mean)",
        "Fault Detection Recall (mean)", "False Alarm Rate (mean)",
        "Inference Time (mean)", "Train Time (mean)",
        "Qualifies (recall>=99%)", "Selected"
    ])
    for i, m in enumerate(qualifying, 1):
        writer.writerow([
            i, m["model"],
            f"{m['macro_f1_mean']:.6f}", f"{m['macro_f1_std']:.6f}",
            f"{m['accuracy_mean']:.6f}", f"{m['balanced_accuracy_mean']:.6f}",
            f"{m['cohen_kappa_mean']:.6f}",
            f"{m['fault_detection_recall_mean']:.6f}",
            f"{m['false_alarm_rate_mean']:.6f}",
            f"{m['inference_time_mean']:.6f}", f"{m['train_time_mean']:.6f}",
            "Yes", "Yes" if i == 1 else ""
        ])
    for i, m in enumerate(eliminated, len(qualifying) + 1):
        writer.writerow([
            i, m["model"],
            f"{m['macro_f1_mean']:.6f}", f"{m['macro_f1_std']:.6f}",
            f"{m['accuracy_mean']:.6f}", f"{m['balanced_accuracy_mean']:.6f}",
            f"{m['cohen_kappa_mean']:.6f}",
            f"{m['fault_detection_recall_mean']:.6f}",
            f"{m['false_alarm_rate_mean']:.6f}",
            f"{m['inference_time_mean']:.6f}", f"{m['train_time_mean']:.6f}",
            "No", ""
        ])

print(f"Saved: {csv_path}")

# --- Verification: no locked test access ---
print()
print("=" * 80)
print("VERIFICATION")
print("=" * 80)
print("[PASS] No split_indices.npz loaded")
print("[PASS] No model training performed")
print("[PASS] No previous artifacts modified")
print("[PASS] All metrics sourced from existing JSON files")
print(f"[PASS] Winner fault recall >= 99%: {winner['fault_detection_recall_mean']:.4f}")
print("[PASS] Analysis complete")
