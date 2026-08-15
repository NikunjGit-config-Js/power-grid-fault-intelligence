"""Post-process Test 4 results: compute CV summary, generate comparison table and figures."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")
FIG_DIR = ROOT / "results" / "test_4" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load checkpoint
with open(ROOT / "results" / "test_4" / "test_4_checkpoint.json") as f:
    ckpt = json.load(f)

all_outer_results = ckpt["all_outer_results"]
best_params_all = ckpt["best_params_all"]
assert ckpt["completed_folds"] == 5, f"Expected 5 folds, got {ckpt['completed_folds']}"

model_names = list(all_outer_results.keys())
print(f"Models: {len(model_names)}")
print(f"Folds per model: {len(all_outer_results[model_names[0]])}")

# ======================================================================
# Compute CV Summary
# ======================================================================
cv_summary = {}
for name in model_names:
    folds = all_outer_results[name]
    macro_f1s = [f["macro_f1"] for f in folds]
    accs = [f["accuracy"] for f in folds]
    bal_accs = [f["balanced_accuracy"] for f in folds]
    kappas = [f["cohen_kappa"] for f in folds]
    fault_recalls = [f["fault_detection_recall"] for f in folds]
    false_alarms = [f["false_alarm_rate"] for f in folds]
    train_times = [f["train_time"] for f in folds]
    infer_times = [f["inference_time"] for f in folds]
    cv_summary[name] = {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs, ddof=1)),
        "cohen_kappa_mean": float(np.mean(kappas)),
        "cohen_kappa_std": float(np.std(kappas, ddof=1)),
        "fault_detection_recall_mean": float(np.mean(fault_recalls)),
        "fault_detection_recall_std": float(np.std(fault_recalls, ddof=1)),
        "false_alarm_rate_mean": float(np.mean(false_alarms)),
        "false_alarm_rate_std": float(np.std(false_alarms, ddof=1)),
        "train_time_mean": float(np.mean(train_times)),
        "inference_time_mean": float(np.mean(infer_times)),
    }

# Save summary
with open(ROOT / "results" / "test_4" / "test_4_results.json", "w") as f:
    json.dump(cv_summary, f, indent=2)
print("Saved test_4_results.json")

# ======================================================================
# Print Results Table
# ======================================================================
rows = []
for name in model_names:
    s = cv_summary[name]
    rows.append({
        "Model": name,
        "Macro F1": f"{s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f}",
        "Accuracy": f"{s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}",
        "Balanced Acc": f"{s['balanced_accuracy_mean']:.4f} +/- {s['balanced_accuracy_std']:.4f}",
        "Cohen Kappa": f"{s['cohen_kappa_mean']:.4f} +/- {s['cohen_kappa_std']:.4f}",
        "Fault Recall": f"{s['fault_detection_recall_mean']:.4f} +/- {s['fault_detection_recall_std']:.4f}",
        "False Alarm": f"{s['false_alarm_rate_mean']:.4f} +/- {s['false_alarm_rate_std']:.4f}",
    })
df = pd.DataFrame(rows).sort_values("Model")
df.to_csv(ROOT / "results" / "test_4" / "model_comparison.csv", index=False)
print("\nModel Comparison (sorted by Macro F1):")
print(df.to_string(index=False))

# ======================================================================
# Update Master model_comparison.csv
# ======================================================================
master_path = ROOT / "results" / "model_comparison.csv"
if master_path.exists():
    master_df = pd.read_csv(master_path)
else:
    master_df = pd.DataFrame()

for name in model_names:
    s = cv_summary[name]
    new_row = pd.DataFrame([{
        "model": name,
        "test": "test_4_nested_cv",
        "macro_f1_mean": s["macro_f1_mean"],
        "macro_f1_std": s["macro_f1_std"],
        "accuracy_mean": s["accuracy_mean"],
        "accuracy_std": s["accuracy_std"],
        "balanced_accuracy_mean": s["balanced_accuracy_mean"],
        "balanced_accuracy_std": s["balanced_accuracy_std"],
        "cohen_kappa_mean": s["cohen_kappa_mean"],
        "cohen_kappa_std": s["cohen_kappa_std"],
        "fault_detection_recall_mean": s["fault_detection_recall_mean"],
        "fault_detection_recall_std": s["fault_detection_recall_std"],
        "false_alarm_rate_mean": s["false_alarm_rate_mean"],
        "false_alarm_rate_std": s["false_alarm_rate_std"],
    }])
    master_df = pd.concat([master_df, new_row], ignore_index=True)

master_df.to_csv(master_path, index=False)
print(f"\nUpdated master model_comparison.csv ({len(master_df)} rows)")

# ======================================================================
# Figures
# ======================================================================
sns.set_theme(style="whitegrid", font_scale=1.1)

# 1. Boxplot of macro F1 across folds
fig, ax = plt.subplots(figsize=(14, 7))
data_for_box = []
for name in model_names:
    for f in all_outer_results[name]:
        data_for_box.append({"Model": name, "Macro F1": f["macro_f1"]})
df_box = pd.DataFrame(data_for_box)
order = df_box.groupby("Model")["Macro F1"].median().sort_values(ascending=False).index
sns.boxplot(data=df_box, y="Model", x="Macro F1", order=order, ax=ax, palette="viridis")
ax.set_title("Test 4: Macro F1 Across 5 Outer Folds (Nested CV)", fontsize=14)
ax.set_xlabel("Macro F1 Score")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(FIG_DIR / "f1_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved f1_boxplot.png")

# 2. Bar chart of mean macro F1 with error bars
fig, ax = plt.subplots(figsize=(14, 7))
names_sorted = sorted(model_names, key=lambda n: cv_summary[n]["macro_f1_mean"], reverse=True)
means = [cv_summary[n]["macro_f1_mean"] for n in names_sorted]
stds = [cv_summary[n]["macro_f1_std"] for n in names_sorted]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names_sorted)))
bars = ax.barh(range(len(names_sorted)), means, xerr=stds, color=colors, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted)
ax.set_xlabel("Macro F1 Score (mean +/- std)")
ax.set_title("Test 4: Nested CV Macro F1 Comparison", fontsize=14)
ax.invert_yaxis()
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 0.002, i, f"{m:.4f}+/-{s:.4f}", va="center", fontsize=8)
ax.set_xlim(left=max(0, min(means) - 0.1), right=min(1.02, max(means) + max(stds) + 0.05))
plt.tight_layout()
fig.savefig(FIG_DIR / "f1_barplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved f1_barplot.png")

# 3. Per-class heatmaps for top 5 models
top5 = names_sorted[:5]
fig, axes = plt.subplots(1, 5, figsize=(25, 6))
for idx, name in enumerate(top5):
    avg_cm = np.mean([np.array(f["confusion_matrix_normalized"]) for f in all_outer_results[name]], axis=0)
    # Get class names from first fold
    first_fold = all_outer_results[name][0]
    # Use the class labels from the data loading
    class_labels = sorted(list(set().union(*[set(f["per_class"].keys()) for f in all_outer_results[name]])))
    sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues", ax=axes[idx],
                xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1)
    axes[idx].set_title(name, fontsize=11)
    axes[idx].set_ylabel("True" if idx == 0 else "")
    axes[idx].set_xlabel("Predicted")
plt.suptitle("Test 4: Average Normalized Confusion Matrices (Top 5 Models)", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "top5_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved top5_confusion_matrices.png")

# 4. Train time comparison
fig, ax = plt.subplots(figsize=(12, 6))
train_times_sorted = [(n, cv_summary[n]["train_time_mean"]) for n in names_sorted]
names_t, times_t = zip(*train_times_sorted)
ax.barh(range(len(names_t)), times_t, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(names_t))))
ax.set_yticks(range(len(names_t)))
ax.set_yticklabels(names_t)
ax.set_xlabel("Mean Training Time (seconds)")
ax.set_title("Test 4: Training Time Comparison", fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(FIG_DIR / "train_time_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved train_time_comparison.png")

# 5. Fault detection recall vs false alarm rate
fig, ax = plt.subplots(figsize=(10, 8))
for name in model_names:
    s = cv_summary[name]
    ax.scatter(s["false_alarm_rate_mean"], s["fault_detection_recall_mean"],
               s=150, zorder=5, edgecolors="black", linewidth=0.5)
    ax.annotate(name, (s["false_alarm_rate_mean"], s["fault_detection_recall_mean"]),
                textcoords="offset points", xytext=(8, 4), fontsize=8)
ax.set_xlabel("Mean False Alarm Rate")
ax.set_ylabel("Mean Fault Detection Recall")
ax.set_title("Test 4: Fault Detection Recall vs False Alarm Rate", fontsize=14)
ax.set_xlim(-0.01, max(cv_summary[n]["false_alarm_rate_mean"] for n in model_names) * 1.2 + 0.01)
ax.set_ylim(0.8, 1.02)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
fig.savefig(FIG_DIR / "fault_recall_vs_false_alarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fault_recall_vs_false_alarm.png")

# 6. Per-fold results table
print("\nPer-fold Macro F1:")
header = f"{'Model':<25s}" + "".join([f"{'Fold'+str(i+1):>10s}" for i in range(5)]) + f"{'Mean':>10s}{'Std':>10s}"
print(header)
print("-" * len(header))
for name in names_sorted:
    f1s = [f["macro_f1"] for f in all_outer_results[name]]
    row = f"{name:<25s}" + "".join([f"{v:>10.4f}" for v in f1s]) + f"{np.mean(f1s):>10.4f}{np.std(f1s, ddof=1):>10.4f}"
    print(row)

# Save per-fold JSON
per_fold = {}
for name in model_names:
    per_fold[name] = [{"fold": i+1, "macro_f1": f["macro_f1"], "accuracy": f["accuracy"],
                        "balanced_accuracy": f["balanced_accuracy"], "cohen_kappa": f["cohen_kappa"],
                        "fault_detection_recall": f["fault_detection_recall"],
                        "false_alarm_rate": f["false_alarm_rate"],
                        "train_time": f["train_time"], "inference_time": f["inference_time"]}
                       for i, f in enumerate(all_outer_results[name])]

with open(ROOT / "results" / "test_4" / "test_4_fold_results.json", "w") as f:
    json.dump(per_fold, f, indent=2)
print("\nSaved test_4_fold_results.json")

print("\nDone!")
