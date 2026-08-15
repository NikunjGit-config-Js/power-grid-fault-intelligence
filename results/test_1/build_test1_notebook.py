"""Build the Test 1 — Default/Baseline Models notebook.

Runs all mandatory ML algorithms with sensible defaults (no tuning).
Uses D1 (Physics Core + Current Symmetry, 20 features) exclusively.
Pipeline: PhysicsWithCurrentSymmetry -> [RobustScaler] -> Classifier
"""

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.13.0"
    }
})

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ===========================================================================
# Title
# ===========================================================================
md("""# Test 1 — Default/Baseline Models

**Objective:** Run all mandatory ML algorithms with sensible defaults (no tuning).
Use D1 (Physics Core + Current Symmetry, 20 features) exclusively.

**Pipeline architecture:** `PhysicsWithCurrentSymmetry -> [RobustScaler] -> Classifier`

**Split:** Locked 70/15/15 (train/val/test). Test set is NEVER evaluated.

**Models:**
- Scale-sensitive: LR, KNN, SVM (Pipeline: D1 -> RobustScaler -> Classifier)
- Tree-based: DT, Bagging, RF, ExtraTrees, AdaBoost, GBM, XGBoost, HistGBM, LightGBM, CatBoost
- Neural Network: Keras MLP (Input -> Dense(64,ReLU) -> Dense(32,ReLU) -> Dense(6,Softmax))
""")

# ===========================================================================
# Cell 2: Imports
# ===========================================================================
code("""import os
os.environ["KERAS_BACKEND"] = "numpy"

import sys
import json
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix, recall_score, precision_score
)

# Classical models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Bagging
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
)

# Boosting
import xgboost
import lightgbm
import catboost

# Neural Network
import keras
from keras import layers

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Project root
ROOT = Path.cwd()
if not (ROOT / "classData.csv").exists():
    ROOT = Path(r"C:\\Users\\ASUS\\Downloads\\power-grid-fault-intelligence")

sys.path.insert(0, str(ROOT))
from src.features import PhysicsWithCurrentSymmetry, RAW_FEATURES

FIG_DIR = ROOT / "results" / "test_1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
print(f"ROOT: {ROOT}")
print(f"XGBoost: {xgboost.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
print(f"CatBoost: {catboost.__version__}")
print(f"Keras: {keras.__version__}")
""")

# ===========================================================================
# Cell 3: Data Loading
# ===========================================================================
md("""---
## 1. Data Loading and Label Construction
""")

code("""df = pd.read_csv(ROOT / "classData.csv")
print(f"Dataset shape: {df.shape}")

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

print(f"Features: {RAW_FEATURES}")
print(f"Classes ({n_classes}): {list(class_names)}")
for cls, count in zip(*np.unique(y, return_counts=True)):
    print(f"  {class_names[cls]}: {count} ({100*count/len(y):.1f}%)")
""")

# ===========================================================================
# Cell 4: Locked Split
# ===========================================================================
md("""---
## 2. Locked 70/15/15 Split

The final 15% test set is NEVER transformed or evaluated during Test 1.
Split indices are persisted for reuse in Tests 2-4.
""")

code("""# First hold out 15% final test
X_dev, X_test, y_dev, y_test = train_test_split(
    X_raw, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

# Split remaining 85% into 70% train + 15% val (of total)
# Validation fraction of remaining = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.15/0.85, random_state=RANDOM_STATE, stratify=y_dev
)

print(f"Total:     {len(X_raw)} samples")
print(f"Dev:       {len(X_dev)} ({100*len(X_dev)/len(X_raw):.1f}%)")
print(f"  Train:   {len(X_train)} ({100*len(X_train)/len(X_raw):.1f}%)")
print(f"  Val:     {len(X_val)} ({100*len(X_val)/len(X_raw):.1f}%)")
print(f"Test:      {len(X_test)} ({100*len(X_test)/len(X_raw):.1f}%) [LOCKED]")

# Persist split indices for Tests 2-4
split_path = ROOT / "results" / "test_1" / "split_indices.npz"
np.savez_compressed(
    split_path,
    train_idx=np.where(np.isin(np.arange(len(X_raw)), np.where(np.isin(y, y_train))[0]))[0],
    val_idx=np.where(np.isin(np.arange(len(X_raw)), np.where(np.isin(y, y_val))[0]))[0],
    test_idx=np.where(np.isin(np.arange(len(X_raw)), np.where(np.isin(y, y_test))[0]))[0],
)
print(f"\\nSplit indices saved to: {split_path}")

# Verify class distribution
print("\\nClass distribution:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = {class_names[i]: int(c) for i, c in enumerate(np.bincount(y_split, minlength=n_classes))}
    print(f"  {split_name}: {dist}")
""")

# ===========================================================================
# Cell 5: Model Definitions
# ===========================================================================
md("""---
## 3. Model Definitions

Every model wraps `PhysicsWithCurrentSymmetry` (D1) as the first pipeline stage.
Scale-sensitive models add `RobustScaler` after D1.
""")

code("""# D1 transformer (shared across all models)
d1 = PhysicsWithCurrentSymmetry()

# Scale-sensitive models: Pipeline(D1 -> RobustScaler -> Classifier)
scale_sensitive = {
    "Logistic Regression": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ]),
    "KNN": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", KNeighborsClassifier()),
    ]),
    "SVM": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", SVC(random_state=RANDOM_STATE, probability=True)),
    ]),
}

# Tree-based models: Pipeline(D1 -> Classifier)
tree_based = {
    "Decision Tree": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]),
    "Bagging": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", BaggingClassifier(random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]),
    "Extra Trees": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", ExtraTreesClassifier(random_state=RANDOM_STATE)),
    ]),
    "AdaBoost": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", AdaBoostClassifier(random_state=RANDOM_STATE)),
    ]),
    "Gradient Boosting": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]),
    "XGBoost": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", xgboost.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
        )),
    ]),
    "HistGradientBoosting": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]),
    "LightGBM": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", lightgbm.LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
        )),
    ]),
    "CatBoost": Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", catboost.CatBoostClassifier(
            random_seed=RANDOM_STATE,
            verbose=0,
        )),
    ]),
}

all_models = {**scale_sensitive, **tree_based}
print(f"ML models defined: {len(all_models)}")
for name in all_models:
    print(f"  - {name}")
""")

# ===========================================================================
# Cell 6: Train ML Models
# ===========================================================================
md("""---
## 4. Train All ML Models

Train on `X_train`, evaluate on `X_val`. Test set is NEVER touched.
""")

code("""results = {}

for name, pipeline in all_models.items():
    print(f"\\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_val_pred = pipeline.predict(X_val)
    infer_time = time.time() - t0

    # Flatten predictions if 2D (happens with CatBoost)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()

    # Metrics
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_val, y_val_pred)
    kappa = cohen_kappa_score(y_val, y_val_pred)

    # Per-class precision and recall
    report = classification_report(y_val, y_val_pred, target_names=class_names, output_dict=True)
    per_class = {}
    for cls_name in class_names:
        per_class[cls_name] = {
            "precision": report[cls_name]["precision"],
            "recall": report[cls_name]["recall"],
            "f1-score": report[cls_name]["f1-score"],
            "support": int(report[cls_name]["support"]),
        }

    # Fault detection recall (binary: any fault vs no-fault)
    no_fault_idx = list(class_names).index("No Fault")
    fault_mask_val = y_val != no_fault_idx
    fault_mask_pred = y_val_pred != no_fault_idx
    fault_recall = recall_score(fault_mask_val.astype(int), fault_mask_pred.astype(int), zero_division=0)

    # False alarm rate
    no_fault_mask_val = y_val == no_fault_idx
    false_alarms = ((y_val_pred != no_fault_idx) & no_fault_mask_val).sum()
    n_no_fault = no_fault_mask_val.sum()
    false_alarm_rate = false_alarms / n_no_fault if n_no_fault > 0 else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    results[name] = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "cohen_kappa": float(kappa),
        "per_class": per_class,
        "fault_detection_recall": float(fault_recall),
        "false_alarm_rate": float(false_alarm_rate),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "train_time": float(train_time),
        "inference_time": float(infer_time),
        "n_features": 20,
        "feature_set": "D1_physics_current_sym",
    }

    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Macro F1:        {f1:.4f}")
    print(f"  Balanced Acc:    {bal_acc:.4f}")
    print(f"  Cohen's Kappa:   {kappa:.4f}")
    print(f"  Fault Recall:    {fault_recall:.4f}")
    print(f"  False Alarm:     {false_alarm_rate:.4f}")
    print(f"  Train time:      {train_time:.3f}s")
""")

# ===========================================================================
# Cell 7: Neural Network (sklearn MLPClassifier)
# ===========================================================================
md("""---
## 5. Neural Network

Architecture: Input(20) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(6, Softmax)
Uses sklearn MLPClassifier with early_stopping=True (patience=10).
RobustScaler fitted on training data only.
""")

code("""from sklearn.neural_network import MLPClassifier

# Pipeline: D1 -> RobustScaler -> MLPClassifier
nn_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("scaler", RobustScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )),
])

t0 = time.time()
nn_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

t0 = time.time()
y_val_pred_nn = nn_pipeline.predict(X_val)
infer_time = time.time() - t0

# Metrics
acc = accuracy_score(y_val, y_val_pred_nn)
f1 = f1_score(y_val, y_val_pred_nn, average="macro")
bal_acc = balanced_accuracy_score(y_val, y_val_pred_nn)
kappa = cohen_kappa_score(y_val, y_val_pred_nn)

report = classification_report(y_val, y_val_pred_nn, target_names=class_names, output_dict=True)
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
fault_mask_pred = y_val_pred_nn != no_fault_idx
fault_recall = recall_score(fault_mask_val.astype(int), fault_mask_pred.astype(int), zero_division=0)

no_fault_mask_val = y_val == no_fault_idx
false_alarms = ((y_val_pred_nn != no_fault_idx) & no_fault_mask_val).sum()
n_no_fault = no_fault_mask_val.sum()
false_alarm_rate = false_alarms / n_no_fault if n_no_fault > 0 else 0.0

cm = confusion_matrix(y_val, y_val_pred_nn)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

results["MLP Neural Network"] = {
    "accuracy": float(acc),
    "macro_f1": float(f1),
    "balanced_accuracy": float(bal_acc),
    "cohen_kappa": float(kappa),
    "per_class": per_class,
    "fault_detection_recall": float(fault_recall),
    "false_alarm_rate": float(false_alarm_rate),
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_normalized": cm_norm.tolist(),
    "train_time": float(train_time),
    "inference_time": float(infer_time),
    "n_features": 20,
    "feature_set": "D1_physics_current_sym",
    "n_iter": nn_pipeline.named_steps["clf"].n_iter_,
}

print(f"\\nMLP Neural Network Results:")
print(f"  Accuracy:        {acc:.4f}")
print(f"  Macro F1:        {f1:.4f}")
print(f"  Balanced Acc:    {bal_acc:.4f}")
print(f"  Cohen's Kappa:   {kappa:.4f}")
print(f"  Fault Recall:    {fault_recall:.4f}")
print(f"  False Alarm:     {false_alarm_rate:.4f}")
print(f"  Train time:      {train_time:.3f}s")
print(f"  Iterations:      {nn_pipeline.named_steps['clf'].n_iter_}")
""")

# ===========================================================================
# Cell 8: Results Table
# ===========================================================================
md("""---
## 6. Results Summary Table
""")

code("""# Build summary table
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
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Macro F1", ascending=False)
print("Test 1 — Default Models (sorted by Macro F1):")
print("=" * 120)
print(summary_df.to_string(index=False, float_format="{:.4f}".format))
""")

# ===========================================================================
# Cell 9: Confusion Matrices
# ===========================================================================
md("""---
## 7. Confusion Matrices

Saved as PNGs for every model.
""")

code("""for name in results:
    r = results[name]
    cm = np.array(r["confusion_matrix"])
    cm_norm = np.array(r["confusion_matrix_normalized"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{name} — Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # Row-normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f"{name} — Confusion Matrix (Row-Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(FIG_DIR / f"confusion_matrix_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

print(f"Saved {len(results)} confusion matrix pairs to: {FIG_DIR}")
""")

# ===========================================================================
# Cell 10: Per-Class Precision/Recall
# ===========================================================================
md("""---
## 8. Per-Class Precision and Recall

Grouped bar charts for every model.
""")

code("""for name in results:
    r = results[name]
    per_class = r["per_class"]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(n_classes)
    width = 0.35

    precisions = [per_class[c]["precision"] for c in class_names]
    recalls = [per_class[c]["recall"] for c in class_names]

    bars1 = ax.bar(x - width/2, precisions, width, label="Precision", color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, recalls, width, label="Recall", color="#FF9800", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(f"{name} — Per-Class Precision and Recall (Test 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(FIG_DIR / f"per_class_pr_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

print(f"Saved {len(results)} per-class PR charts to: {FIG_DIR}")
""")

# ===========================================================================
# Cell 11: Model Comparison
# ===========================================================================
md("""---
## 9. Model Comparison Visualizations
""")

code("""# Macro F1 comparison
fig, ax = plt.subplots(figsize=(12, 6))
names_sorted = summary_df["Model"].values
f1_sorted = summary_df["Macro F1"].values
colors = sns.color_palette("Set2", len(names_sorted))

bars = ax.barh(names_sorted, f1_sorted, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Macro F1")
ax.set_title("Test 1 — Model Comparison (Macro F1)")
ax.set_xlim(0, 1.05)
for i, v in enumerate(f1_sorted):
    ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'model_comparison_f1.png'}")

# Fault Detection Recall comparison
fig, ax = plt.subplots(figsize=(12, 6))
fault_sorted = [results[m]["fault_detection_recall"] for m in names_sorted]
bars = ax.barh(names_sorted, fault_sorted, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Fault Detection Recall")
ax.set_title("Test 1 — Model Comparison (Fault Detection Recall)")
ax.set_xlim(0, 1.05)
for i, v in enumerate(fault_sorted):
    ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_fault_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'model_comparison_fault_recall.png'}")
""")

# ===========================================================================
# Cell 12: Save Results
# ===========================================================================
md("""---
## 10. Save Results
""")

code("""# Save test_1_results.json
out_path = ROOT / "results" / "test_1" / "test_1_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {out_path}")

# Update model_comparison.csv
csv_path = ROOT / "results" / "model_comparison.csv"
rows = []
for name in results:
    r = results[name]
    rows.append({
        "Experiment": "Test 1",
        "Feature Set": "D1_physics_current_sym",
        "Model": name,
        "Accuracy": r["accuracy"],
        "Macro F1": r["macro_f1"],
        "Balanced Accuracy": r["balanced_accuracy"],
        "Cohen Kappa": r["cohen_kappa"],
        "Fault Recall": r["fault_detection_recall"],
        "False Alarm Rate": r["false_alarm_rate"],
        "ROC-AUC": "Not Applicable",
        "PR-AUC": "Not Applicable",
        "CV Std Macro F1": "N/A (single split)",
        "Training Time": r["train_time"],
        "Inference Time": r["inference_time"],
        "Number of Features": 20,
    })

csv_df = pd.DataFrame(rows)
csv_df.to_csv(csv_path, index=False)
print(f"Model comparison saved to: {csv_path}")

# List all figures
figures = sorted(FIG_DIR.glob("*.png"))
print(f"\\nTotal figures saved: {len(figures)}")
for f in figures:
    print(f"  {f.name}")
""")

# ===========================================================================
# Cell 13: Summary
# ===========================================================================
md("""---
## 11. Summary

### What Was Done
- Ran 13 ML models + 1 Keras MLP with default parameters
- All models use D1 (Physics Core + Current Symmetry, 20 features)
- Pipeline: `PhysicsWithCurrentSymmetry -> [RobustScaler] -> Classifier`
- RobustScaler fitted on training data only
- Test set NEVER evaluated

### Key Findings
- [See results table above]
- Best model by Macro F1: [top model]
- All metrics stored in `test_1_results.json`
- Figures saved to `results/test_1/figures/`

### Next Steps
- Test 2: Holdout hyperparameter tuning with Optuna/Keras Tuner
- Test 3: 5-fold stratified cross-validation
- Test 4: Nested cross-validation + tuning
""")

# ===========================================================================
# Assemble notebook
# ===========================================================================
nb.cells = cells

out_path = ROOT / "results" / "test_1" / "test_1_default_models.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook created: {out_path}")
print(f"Total cells: {len(cells)}")
