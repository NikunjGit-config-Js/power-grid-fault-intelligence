import os
os.environ['KERAS_BACKEND'] = 'numpy'

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")

# Load results
results_path = ROOT / "results" / "test_2" / "test_2_results.json"
with open(results_path, "r") as f:
    results = json.load(f)

fig_dir = ROOT / "results" / "test_2" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

class_names = ["3P-G", "LG A+B-G", "LL-G BC", "Line-to-Line AB", "No Fault", "Three-Phase"]

# --- 1. Model comparison bar chart ---
models = list(results.keys())
macro_f1s = [results[m]["macro_f1"] for m in models]
balanced_accs = [results[m]["balanced_accuracy"] for m in models]
accuracies = [results[m]["accuracy"] for m in models]

sorted_idx = np.argsort(macro_f1s)[::-1]
models_sorted = [models[i] for i in sorted_idx]
f1_sorted = [macro_f1s[i] for i in sorted_idx]
acc_sorted = [accuracies[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(models_sorted))
width = 0.35
bars1 = ax.bar(x - width/2, f1_sorted, width, label='Macro F1', color='#2196F3', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, acc_sorted, width, label='Accuracy', color='#4CAF50', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Test 2 — Hyperparameter-Tuned Models: Macro F1 vs Accuracy (sorted by Macro F1)')
ax.set_xticks(x)
ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=9)
ax.legend()
ax.set_ylim(0.0, 1.05)
ax.axhline(y=0.99, color='red', linestyle='--', alpha=0.3, label='0.99 threshold')
for bar, val in zip(bars1, f1_sorted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(fig_dir / "model_comparison_macro_f1.png", dpi=150)
plt.close()
print("Saved: model_comparison_macro_f1.png")

# --- 2. Confusion matrices for top models ---
top_models = ["Gradient Boosting", "HistGradientBoosting", "Random Forest", "CatBoost", "Bagging", "Extra Trees"]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for idx, model_name in enumerate(top_models):
    ax = axes[idx // 3, idx % 3]
    cm = np.array(results[model_name]["confusion_matrix_normalized"])
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax, vmin=0, vmax=1)
    ax.set_title(f'{model_name}\nMacro F1 = {results[model_name]["macro_f1"]:.4f}', fontsize=11)
    ax.set_ylabel('True' if idx % 3 == 0 else '')
    ax.set_xlabel('Predicted' if idx // 3 == 1 else '')
plt.suptitle('Test 2 — Normalized Confusion Matrices (Top 6 Models)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(fig_dir / "confusion_matrices_top6.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices_top6.png")

# --- 3. Per-class F1 for all models ---
per_class_data = []
for model_name in results:
    for cls_name in class_names:
        per_class_data.append({
            "Model": model_name,
            "Class": cls_name,
            "F1": results[model_name]["per_class"][cls_name]["f1-score"],
            "Precision": results[model_name]["per_class"][cls_name]["precision"],
            "Recall": results[model_name]["per_class"][cls_name]["recall"],
        })
df_per_class = pd.DataFrame(per_class_data)

fig, ax = plt.subplots(figsize=(14, 8))
pivot_f1 = df_per_class.pivot_table(index="Model", columns="Class", values="F1")
pivot_f1 = pivot_f1.loc[models_sorted]
sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlGn', ax=ax, vmin=0.8, vmax=1.0, linewidths=0.5)
ax.set_title('Test 2 — Per-Class F1 Score by Model', fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "per_class_f1_heatmap.png", dpi=150)
plt.close()
print("Saved: per_class_f1_heatmap.png")

# --- 4. Fault detection recall and false alarm rate ---
fault_recall = [results[m]["fault_detection_recall"] for m in models_sorted]
false_alarm = [results[m]["false_alarm_rate"] for m in models_sorted]

fig, ax1 = plt.subplots(figsize=(12, 6))
x = np.arange(len(models_sorted))
ax1.bar(x - 0.2, fault_recall, 0.4, label='Fault Detection Recall', color='#2196F3', alpha=0.8)
ax1.bar(x + 0.2, false_alarm, 0.4, label='False Alarm Rate', color='#F44336', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('Rate')
ax1.set_title('Test 2 — Fault Detection Recall vs False Alarm Rate')
ax1.set_xticks(x)
ax1.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(fig_dir / "fault_recall_false_alarm.png", dpi=150)
plt.close()
print("Saved: fault_recall_false_alarm.png")

# --- 5. Feature importance / best hyperparameters summary ---
print("\n--- Best Hyperparameters Summary ---")
for model_name in ["Gradient Boosting", "HistGradientBoosting", "Random Forest", "CatBoost", "LightGBM", "XGBoost"]:
    bp = results[model_name].get("best_params", {})
    print(f"\n{model_name} (Macro F1 = {results[model_name]['macro_f1']:.4f}):")
    for k, v in bp.items():
        print(f"  {k}: {v}")

# Keras MLP
if "MLP Neural Network" in results:
    keras_hp = results["MLP Neural Network"].get("best_hyperparameters", {})
    print(f"\nMLP Neural Network (Macro F1 = {results['MLP Neural Network']['macro_f1']:.4f}):")
    for k, v in keras_hp.items():
        print(f"  {k}: {v}")

print("\nAll figures saved to:", fig_dir)
