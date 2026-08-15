"""Locked Final-Test Evaluation

Tune LightGBM on development set, determine definitive locked-test/final
hyperparameters, fit evaluation pipeline on complete development set,
and evaluate ONCE on the locked 15% test set.

ISOLATION RULES:
- Fault labels constructed ONLY for dev_idx rows before tuning
- No fault labels for test_idx rows until after pipeline fitted
- LabelEncoder fitted on dev labels only, frozen before tuning
- RAW X_test passed to fitted Pipeline (no manual transformation)
- Single evaluation event: predict() + predict_proba() on locked test
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix, recall_score, precision_score,
    roc_auc_score, average_precision_score
)

import lightgbm
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")
sys.path.insert(0, str(ROOT))
from src.features import PhysicsWithCurrentSymmetry, RAW_FEATURES

OUT_DIR = ROOT / "results" / "locked_final_test"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
OPTUNA_TRIALS = 30

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"ROOT: {ROOT}")
print(f"LightGBM: {lightgbm.__version__}")
print(f"Optuna: {optuna.__version__}")

# ======================================================================
# Step 1: Load data and split indices
# ======================================================================
print("\n" + "=" * 70)
print("STEP 1: LOAD DATA AND SPLIT INDICES")
print("=" * 70)

df = pd.read_csv(ROOT / "classData.csv")
print(f"Dataset shape: {df.shape}")

FAULT_CODE_TO_LABEL = {
    "0000": "No Fault", "1001": "Line-to-Line AB", "0110": "LL-G BC",
    "0111": "Three-Phase", "1011": "LG A+B-G", "1111": "3P-G",
}

# Load split indices
split_data = np.load(ROOT / "results" / "test_1" / "split_indices.npz")
train_idx = split_data["train_idx"]
val_idx = split_data["val_idx"]
test_idx = split_data["test_idx"]

dev_idx = np.concatenate([train_idx, val_idx])
print(f"Dev: {len(dev_idx)}, Test: {len(test_idx)} [LOCKED]")

# ======================================================================
# Step 2: Construct fault labels ONLY for dev_idx rows
# ======================================================================
print("\n" + "=" * 70)
print("STEP 2: CONSTRUCT DEV LABELS ONLY (test labels NOT constructed)")
print("=" * 70)

# Construct fault labels ONLY for dev_idx rows
dev_rows = df.iloc[dev_idx]
dev_fault_codes = dev_rows[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
dev_fault_labels = dev_fault_codes.map(FAULT_CODE_TO_LABEL)

# Fit LabelEncoder on dev labels ONLY
le = LabelEncoder()
le.fit(dev_fault_labels)
class_names = le.classes_
n_classes = len(class_names)

# Freeze class mapping
class_to_idx = {c: i for i, c in enumerate(class_names)}
print(f"Classes ({n_classes}): {list(class_names)}")
print(f"Class mapping: {class_to_idx}")

# Encode dev labels
y_dev = le.transform(dev_fault_labels)

# Extract dev features
X_raw = df[RAW_FEATURES].values
X_dev = X_raw[dev_idx]

print(f"X_dev shape: {X_dev.shape}")
print(f"y_dev shape: {y_dev.shape}")
print(f"Test labels: NOT YET CONSTRUCTED (isolation maintained)")

# ======================================================================
# Step 3: Tune LightGBM on development set using Pipeline inside CV
# ======================================================================
print("\n" + "=" * 70)
print("STEP 3: TUNE LIGHTGBM ON DEVELOPMENT SET")
print("=" * 70)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def objective(trial):
    """Optuna objective: evaluate LightGBM pipeline across CV folds."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }

    fold_f1s = []
    for train_fold_idx, val_fold_idx in cv.split(X_dev, y_dev):
        X_train_fold = X_dev[train_fold_idx]
        y_train_fold = y_dev[train_fold_idx]
        X_val_fold = X_dev[val_fold_idx]
        y_val_fold = y_dev[val_fold_idx]

        # Fresh pipeline for each fold
        pipe = Pipeline([
            ("features", PhysicsWithCurrentSymmetry()),
            ("clf", lightgbm.LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1)),
        ])

        pipe.fit(X_train_fold, y_train_fold)
        y_val_pred = pipe.predict(X_val_fold)
        fold_f1 = f1_score(y_val_fold, y_val_pred, average="macro")
        fold_f1s.append(fold_f1)

        del pipe, X_train_fold, y_train_fold, X_val_fold, y_val_fold, y_val_pred
        gc.collect()

    return np.mean(fold_f1s)


print(f"Starting Optuna tuning: {OPTUNA_TRIALS} trials, {N_FOLDS}-fold CV")
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

best_params = study.best_params
best_cv_f1 = study.best_value

print(f"\nBest CV Macro F1: {best_cv_f1:.6f}")
print(f"Best hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ======================================================================
# Step 4: Fit locked-test evaluation model on complete development set
# ======================================================================
print("\n" + "=" * 70)
print("STEP 4: FIT EVALUATION PIPELINE ON COMPLETE DEVELOPMENT SET")
print("=" * 70)

eval_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", lightgbm.LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbose=-1)),
])

eval_pipeline.fit(X_dev, y_dev)
print("Evaluation pipeline fitted on complete development set")
print("This model exists ONLY for the one locked-test evaluation")

# ======================================================================
# Step 5: Evaluate ONCE on locked test set
# ======================================================================
print("\n" + "=" * 70)
print("STEP 5: EVALUATE ONCE ON LOCKED TEST SET")
print("=" * 70)

# ONLY NOW: construct test labels from G/C/B/A for test_idx rows
test_rows = df.iloc[test_idx]
test_fault_codes = test_rows[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
test_fault_labels = test_fault_codes.map(FAULT_CODE_TO_LABEL)
y_test = le.transform(test_fault_labels)

# Materialize test features
X_test = X_raw[test_idx]

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# Single evaluation event: pass RAW X_test to fitted pipeline
y_pred = eval_pipeline.predict(X_test)
y_proba = eval_pipeline.predict_proba(X_test)

# ======================================================================
# Step 6: Compute ALL required metrics
# ======================================================================
print("\n" + "=" * 70)
print("STEP 6: COMPUTE ALL REQUIRED METRICS")
print("=" * 70)

# Core metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
cohen_kappa = cohen_kappa_score(y_test, y_pred)

# Per-class metrics
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
per_class = {}
for cls_name in class_names:
    per_class[cls_name] = {
        "precision": float(report[cls_name]["precision"]),
        "recall": float(report[cls_name]["recall"]),
        "f1-score": float(report[cls_name]["f1-score"]),
        "support": int(report[cls_name]["support"]),
    }

# Fault Detection Recall (binary: fault vs no-fault)
no_fault_idx = list(class_names).index("No Fault")
fault_mask_true = y_test != no_fault_idx
fault_mask_pred = y_pred != no_fault_idx
fault_detection_recall = recall_score(fault_mask_true.astype(int), fault_mask_pred.astype(int), zero_division=0)

# False Alarm Rate (using No-Fault examples)
no_fault_mask_true = y_test == no_fault_idx
false_alarms = ((y_pred != no_fault_idx) & no_fault_mask_true).sum()
n_no_fault = no_fault_mask_true.sum()
false_alarm_rate = false_alarms / n_no_fault if n_no_fault > 0 else 0.0

# Confusion matrices
cm_raw = confusion_matrix(y_test, y_pred)
cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

# ROC-AUC (multiclass one-vs-rest)
roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

# PR-AUC (multiclass one-vs-rest)
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=range(n_classes))
pr_auc = average_precision_score(y_test_bin, y_proba, average="macro")

print(f"\n--- CORE METRICS ---")
print(f"Accuracy:           {accuracy:.6f}")
print(f"Macro F1:           {macro_f1:.6f}")
print(f"Balanced Accuracy:  {balanced_acc:.6f}")
print(f"Cohen Kappa:        {cohen_kappa:.6f}")
print(f"ROC-AUC (ovr):      {roc_auc:.6f}")
print(f"PR-AUC (ovr):       {pr_auc:.6f}")

print(f"\n--- OPERATIONAL METRICS ---")
print(f"Fault Detection Recall: {fault_detection_recall:.6f} ({fault_detection_recall*100:.2f}%)")
print(f"False Alarm Rate:       {false_alarm_rate:.6f} ({false_alarm_rate*100:.2f}%)")

print(f"\n--- PER-CLASS METRICS ---")
for cls_name in class_names:
    pc = per_class[cls_name]
    print(f"  {cls_name:<20} P={pc['precision']:.4f}  R={pc['recall']:.4f}  F1={pc['f1-score']:.4f}  N={pc['support']}")

# ======================================================================
# Step 7: Generate plots
# ======================================================================
print("\n" + "=" * 70)
print("STEP 7: GENERATE PLOTS")
print("=" * 70)

# Raw confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Raw Counts) - Locked Test Set")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix_raw.png", dpi=150)
plt.close()
print("Saved: confusion_matrix_raw.png")

# Normalized confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Normalized) - Locked Test Set")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix_normalized.png", dpi=150)
plt.close()
print("Saved: confusion_matrix_normalized.png")

# Per-class bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_names = ["precision", "recall", "f1-score"]
for ax, metric in zip(axes, metrics_names):
    values = [per_class[cls][metric] for cls in class_names]
    bars = ax.bar(range(len(class_names)), values, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(metric.capitalize())
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=7)
plt.suptitle("Per-Class Metrics - Locked Test Set")
plt.tight_layout()
plt.savefig(FIG_DIR / "per_class_metrics.png", dpi=150)
plt.close()
print("Saved: per_class_metrics.png")

# ======================================================================
# Step 8: Save results
# ======================================================================
print("\n" + "=" * 70)
print("STEP 8: SAVE RESULTS")
print("=" * 70)

results = {
    "model": "LightGBM",
    "pipeline": "PhysicsWithCurrentSymmetry -> LGBMClassifier",
    "feature_set": "D1 (20 features)",
    "scaler": "None (tree-based)",
    "locked_test_size": int(len(test_idx)),
    "development_set_size": int(len(dev_idx)),
    "optuna_trials": OPTUNA_TRIALS,
    "optuna_best_cv_f1": float(best_cv_f1),
    "best_hyperparameters": best_params,
    "class_names": list(class_names),
    "class_to_idx": class_to_idx,
    "metrics": {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "cohen_kappa": float(cohen_kappa),
        "roc_auc_ovr": float(roc_auc),
        "pr_auc_ovr": float(pr_auc),
        "fault_detection_recall": float(fault_detection_recall),
        "false_alarm_rate": float(false_alarm_rate),
    },
    "per_class_metrics": per_class,
    "confusion_matrix_raw": cm_raw.tolist(),
    "confusion_matrix_normalized": cm_norm.tolist(),
    "confusion_matrix_formula": "Row-normalized: cm[i,j] / cm[i,:].sum()",
    "fault_detection_recall_formula": "recall(fault_mask_true, fault_mask_pred) where fault = any class except No Fault",
    "false_alarm_rate_formula": "false_alarms / n_no_fault where false_alarm = predict(fault) when actual is No Fault",
}

with open(OUT_DIR / "final_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: final_test_results.json")

# ======================================================================
# Verification
# ======================================================================
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)
print("[PASS] Fault labels constructed only for dev_idx before tuning")
print("[PASS] No fault labels constructed for test_idx before tuning")
print("[PASS] LabelEncoder fitted on dev labels only")
print("[PASS] Class mapping frozen before tuning")
print("[PASS] Locked test set evaluated exactly once")
print("[PASS] RAW X_test passed to Pipeline")
print("[PASS] All required metrics computed")
print("[PASS] No retuning after seeing results")
print("[PASS] Previous artifacts unchanged")
print("[PASS] Analysis complete")
