import os
os.environ['KERAS_BACKEND'] = 'numpy'

import sys
import json
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix, recall_score, precision_score
)

from sklearn.svm import SVC

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")
sys.path.insert(0, str(ROOT))
from src.features import PhysicsWithCurrentSymmetry, RAW_FEATURES

FIG_DIR = ROOT / "results" / "test_2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OPTUNA_TRIALS = 3  # Very few trials for SVM

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("ROOT:", ROOT)
print("Optuna:", optuna.__version__)
print("Optuna trials:", OPTUNA_TRIALS)

# Load data
df = pd.read_csv(ROOT / "classData.csv")
print("\nDataset shape:", df.shape)

FAULT_CODE_TO_LABEL = {
    "0000": "No Fault",
    "1001": "Line-to-Line AB",
    "0110": "LL-G BC",
    "0111": "Three-Phase",
    "1011": "LG A+B-G",
    "1111": "3P-G",
}

fault_codes = df[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
df["fault_label"] = fault_codes.map(FAULT_CODE_TO_LABEL)

X_raw = df[RAW_FEATURES].values
y_raw = df["fault_label"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
n_classes = len(class_names)

print("Features:", RAW_FEATURES)
print("Classes (" + str(n_classes) + "):", list(class_names))

# Split
X_dev, X_test, y_dev, y_test = train_test_split(
    X_raw, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.15/0.85, random_state=RANDOM_STATE, stratify=y_dev
)

print("\nTotal:     " + str(len(X_raw)) + " samples")
print("Dev:       " + str(len(X_dev)) + " (" + str(100*len(X_dev)/len(X_raw)) + "%)")
print("  Train:   " + str(len(X_train)) + " (" + str(100*len(X_train)/len(X_raw)) + "%)")
print("  Val:     " + str(len(X_val)) + " (" + str(100*len(X_val)/len(X_raw)) + "%)")
print("Test:      " + str(len(X_test)) + " (" + str(100*len(X_test)/len(X_raw)) + "%) [LOCKED]")

# Helper function
def evaluate_model(pipeline, X_val, y_val, class_names, name):
    t0 = time.time()
    y_val_pred = pipeline.predict(X_val)
    infer_time = time.time() - t0
    
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_val, y_val_pred)
    kappa = cohen_kappa_score(y_val, y_val_pred)
    
    report = classification_report(y_val, y_val_pred, target_names=class_names, output_dict=True)
    per_class = {}
    for cls_name in class_names:
        per_class[cls_name] = {
            "precision": report[cls_name]["precision"],
            "recall": report[cls_name]["recall"],
            "f1-score": report[cls_name]["f1-score"],
            "support": int(report[cls_name]["support"]),
        }
    
    no_fault_idx = list(class_names).index("No Fault")
    fault_mask_val = y_val != no_fault_idx
    fault_mask_pred = y_val_pred != no_fault_idx
    fault_recall = recall_score(fault_mask_val.astype(int), fault_mask_pred.astype(int), zero_division=0)
    
    no_fault_mask_val = y_val == no_fault_idx
    false_alarms = ((y_val_pred != no_fault_idx) & no_fault_mask_val).sum()
    n_no_fault = no_fault_mask_val.sum()
    false_alarm_rate = false_alarms / n_no_fault if n_no_fault > 0 else 0.0
    
    cm = confusion_matrix(y_val, y_val_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "cohen_kappa": float(kappa),
        "per_class": per_class,
        "fault_detection_recall": float(fault_recall),
        "false_alarm_rate": float(false_alarm_rate),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "train_time": 0.0,
        "inference_time": float(infer_time),
        "n_features": 20,
        "feature_set": "D1_physics_current_sym",
    }

# Load existing results
checkpoint_path = ROOT / "results" / "test_2" / "test_2_checkpoint.json"
with open(checkpoint_path, "r") as f:
    results = json.load(f)

print("\nLoaded existing results with", len(results), "models")

# SVM with very simple search space
print("\n--- SVM Optuna (3 trials) ---")
def objective_svm(trial):
    # Simple search space
    C = trial.suggest_float("C", 0.1, 10, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf"])
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", SVC(
            random_state=RANDOM_STATE,
            C=C,
            kernel=kernel,
            gamma="scale",
            class_weight="balanced",
            probability=True,
        )),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_svm = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_svm.optimize(objective_svm, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

best_params_svm = study_svm.best_trial.params
print("Best SVM Macro F1:", study_svm.best_trial.value)

svm_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("scaler", RobustScaler()),
    ("clf", SVC(
        random_state=RANDOM_STATE,
        **best_params_svm,
        gamma="scale",
        class_weight="balanced",
        probability=True,
    )),
])
t0 = time.time()
svm_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_svm = evaluate_model(svm_pipeline, X_val, y_val, class_names, "SVM")
results_svm["train_time"] = float(train_time)
results_svm["n_trials"] = OPTUNA_TRIALS
results_svm["best_params"] = best_params_svm
results["SVM"] = results_svm

# Save updated results
out_path = ROOT / "results" / "test_2" / "test_2_results_optuna.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nOptuna ML results saved to:", out_path)

# Save best params
best_params_dict = {}
for name, r in results.items():
    if "best_params" in r:
        best_params_dict[name] = r["best_params"]

params_path = ROOT / "results" / "test_2" / "optuna_best_params.json"
with open(params_path, "w") as f:
    json.dump(best_params_dict, f, indent=2)
print("Best params saved to:", params_path)

# Build summary table
summary_rows = []
for name in results:
    r = results[name]
    summary_rows.append({
        "Model": name,
        "Accuracy": r["accuracy"],
        "Macro F1": r["macro_f1"],
        "Balanced Acc": r["balanced_accuracy"],
        "Cohen Kappa": r["cohen_kappa"],
        "Fault Recall": r["fault_detection_recall"],
        "False Alarm": r["false_alarm_rate"],
        "Train Time (s)": r["train_time"],
        "Trials": r.get("n_trials", "N/A"),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Macro F1", ascending=False)
print("\nTest 2 — Optuna Tuned ML Models (sorted by Macro F1):")
print("=" * 130)
print(summary_df.to_string(index=False, float_format="{:.4f}".format))

print("\nOptuna ML optimization complete!")
print("Total models:", len(results))
