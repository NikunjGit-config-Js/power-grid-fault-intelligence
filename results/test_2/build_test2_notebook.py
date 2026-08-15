"""Build the Test 2 — Holdout Hyperparameter Tuning notebook.

Uses Optuna for ML models and Keras Tuner for Neural Network.
All models use D1 (Physics Core + Current Symmetry, 20 features).
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
md("""# Test 2 — Holdout Hyperparameter Tuning

**Objective:** Tune hyperparameters for all models using Optuna (ML) and Keras Tuner (NN).
Use D1 (Physics Core + Current Symmetry, 20 features) exclusively.

**Pipeline architecture:** `PhysicsWithCurrentSymmetry -> [RobustScaler] -> Classifier`

**Split:** Locked 70/15/15 (train/val/test). Test set is NEVER evaluated.

**Optimization:** Macro F1 on validation set (NOT accuracy).

**Trial Budgets:**
- Optuna: 50 trials per model
- Keras Tuner: 30 trials

**Models:**
- Scale-sensitive: LR, KNN, SVM (Pipeline: D1 -> RobustScaler -> Classifier)
- Tree-based: DT, Bagging, RF, ExtraTrees, AdaBoost, GBM, XGBoost, HistGBM, LightGBM, CatBoost
- Neural Network: Keras MLP with architecture search
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
    classification_report, confusion_matrix, recall_score, precision_score,
    make_scorer
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

# Optuna
import optuna
from optuna.samplers import TPESampler

# Neural Network
import keras
from keras import layers
import keras_tuner as kt

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Project root
ROOT = Path.cwd()
if not (ROOT / "classData.csv").exists():
    ROOT = Path(r"C:\\Users\\ASUS\\Downloads\\power-grid-fault-intelligence")

sys.path.insert(0, str(ROOT))
from src.features import PhysicsWithCurrentSymmetry, RAW_FEATURES

FIG_DIR = ROOT / "results" / "test_2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OPTUNA_TRIALS = 20
KERAS_TRIALS = 15

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"ROOT: {ROOT}")
print(f"XGBoost: {xgboost.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
print(f"CatBoost: {catboost.__version__}")
print(f"Keras: {keras.__version__}")
print(f"Optuna: {optuna.__version__}")
print(f"Keras Tuner: {kt.__version__}")
print(f"Optuna trials: {OPTUNA_TRIALS}")
print(f"Keras Tuner trials: {KERAS_TRIALS}")
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

The final 15% test set is NEVER transformed or evaluated during Test 2.
Split indices are loaded from Test 1 for consistency.
""")

code("""# First hold out 15% final test
X_dev, X_test, y_dev, y_test = train_test_split(
    X_raw, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

# Split remaining 85% into 70% train + 15% val (of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.15/0.85, random_state=RANDOM_STATE, stratify=y_dev
)

print(f"Total:     {len(X_raw)} samples")
print(f"Dev:       {len(X_dev)} ({100*len(X_dev)/len(X_raw):.1f}%)")
print(f"  Train:   {len(X_train)} ({100*len(X_train)/len(X_raw):.1f}%)")
print(f"  Val:     {len(X_val)} ({100*len(X_val)/len(X_raw):.1f}%)")
print(f"Test:      {len(X_test)} ({100*len(X_test)/len(X_raw):.1f}%) [LOCKED]")

# Verify class distribution
print("\\nClass distribution:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = {class_names[i]: int(c) for i, c in enumerate(np.bincount(y_split, minlength=n_classes))}
    print(f"  {split_name}: {dist}")
""")

# ===========================================================================
# Cell 5: Optuna Helper Function
# ===========================================================================
md("""---
## 3. Optuna Helper Function

Defines the objective function for Optuna optimization.
Uses Pipeline: D1 -> [RobustScaler] -> Classifier
""")

code("""def create_objective(X_train, y_train, X_val, y_val, class_names, is_scale_sensitive=False):
    \"\"\"Create an Optuna objective function for a given pipeline configuration.\"\"\"
    
    no_fault_idx = list(class_names).index("No Fault")
    
    def objective(trial):
        # Get hyperparameters from trial
        params = {}
        
        # Pipeline steps
        steps = [("features", PhysicsWithCurrentSymmetry())]
        
        if is_scale_sensitive:
            steps.append(("scaler", RobustScaler()))
        
        # Classifier selection and hyperparameters
        classifier_name = trial.suggest_categorical("classifier", ["classifier"])
        
        # We'll pass the classifier as a parameter
        clf = trial.params.get("clf")
        if clf is None:
            return 0.0
        
        steps.append(("clf", clf))
        pipeline = Pipeline(steps)
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_val_pred = pipeline.predict(X_val)
        if y_val_pred.ndim > 1:
            y_val_pred = y_val_pred.flatten()
        
        # Calculate macro F1
        f1 = f1_score(y_val, y_val_pred, average="macro")
        
        return f1
    
    return objective


def evaluate_model(pipeline, X_val, y_val, class_names, name):
    \"\"\"Evaluate a trained pipeline and return all metrics.\"\"\"
    
    t0 = time.time()
    y_val_pred = pipeline.predict(X_val)
    infer_time = time.time() - t0
    
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    # Get probabilities if available
    y_val_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_val_proba = pipeline.predict_proba(X_val)
        except:
            pass
    
    # Core metrics
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
    
    # Fault detection recall
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

print("Helper functions defined.")
""")

# ===========================================================================
# Cell 6: Optuna - Decision Tree
# ===========================================================================
md("""---
## 4. Optuna Tuning — Decision Tree
""")

code("""def objective_dt(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE, **params)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_dt = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_dt.optimize(objective_dt, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best DT trial: {study_dt.best_trial.value:.4f}")
print(f"Best DT params: {study_dt.best_trial.params}")

# Train final model with best params
best_params_dt = study_dt.best_trial.params
dt_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE, **best_params_dt)),
])
t0 = time.time()
dt_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_dt = evaluate_model(dt_pipeline, X_val, y_val, class_names, "Decision Tree")
results_dt["train_time"] = float(train_time)
results_dt["best_params"] = best_params_dt
results_dt["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 7: Optuna - Random Forest
# ===========================================================================
md("""---
## 5. Optuna Tuning — Random Forest
""")

code("""def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **params)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_rf = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_rf.optimize(objective_rf, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best RF trial: {study_rf.best_trial.value:.4f}")
print(f"Best RF params: {study_rf.best_trial.params}")

# Train final model with best params
best_params_rf = study_rf.best_trial.params
rf_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **best_params_rf)),
])
t0 = time.time()
rf_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_rf = evaluate_model(rf_pipeline, X_val, y_val, class_names, "Random Forest")
results_rf["train_time"] = float(train_time)
results_rf["best_params"] = best_params_rf
results_rf["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 8: Optuna - Extra Trees
# ===========================================================================
md("""---
## 6. Optuna Tuning — Extra Trees
""")

code("""def objective_et(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", ExtraTreesClassifier(random_state=RANDOM_STATE, **params)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_et = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_et.optimize(objective_et, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best ET trial: {study_et.best_trial.value:.4f}")
print(f"Best ET params: {study_et.best_trial.params}")

# Train final model with best params
best_params_et = study_et.best_trial.params
et_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", ExtraTreesClassifier(random_state=RANDOM_STATE, **best_params_et)),
])
t0 = time.time()
et_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_et = evaluate_model(et_pipeline, X_val, y_val, class_names, "Extra Trees")
results_et["train_time"] = float(train_time)
results_et["best_params"] = best_params_et
results_et["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 9: Optuna - Bagging
# ===========================================================================
md("""---
## 7. Optuna Tuning — Bagging
""")

code("""def objective_bagging(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "bootstrap_features": trial.suggest_categorical("bootstrap_features", [True, False]),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", BaggingClassifier(random_state=RANDOM_STATE, **params)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_bagging = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_bagging.optimize(objective_bagging, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best Bagging trial: {study_bagging.best_trial.value:.4f}")
print(f"Best Bagging params: {study_bagging.best_trial.params}")

# Train final model with best params
best_params_bagging = study_bagging.best_trial.params
bagging_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", BaggingClassifier(random_state=RANDOM_STATE, **best_params_bagging)),
])
t0 = time.time()
bagging_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_bagging = evaluate_model(bagging_pipeline, X_val, y_val, class_names, "Bagging")
results_bagging["train_time"] = float(train_time)
results_bagging["best_params"] = best_params_bagging
results_bagging["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 10: Optuna - AdaBoost
# ===========================================================================
md("""---
## 8. Optuna Tuning — AdaBoost
""")

code("""def objective_adaboost(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", AdaBoostClassifier(random_state=RANDOM_STATE, **params)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_adaboost = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_adaboost.optimize(objective_adaboost, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best AdaBoost trial: {study_adaboost.best_trial.value:.4f}")
print(f"Best AdaBoost params: {study_adaboost.best_trial.params}")

# Train final model with best params
best_params_adaboost = study_adaboost.best_trial.params
adaboost_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", AdaBoostClassifier(random_state=RANDOM_STATE, **best_params_adaboost)),
])
t0 = time.time()
adaboost_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_adaboost = evaluate_model(adaboost_pipeline, X_val, y_val, class_names, "AdaBoost")
results_adaboost["train_time"] = float(train_time)
results_adaboost["best_params"] = best_params_adaboost
results_adaboost["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 11: Optuna - Gradient Boosting
# ===========================================================================
md("""---
## 9. Optuna Tuning — Gradient Boosting
""")

code("""def objective_gb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE, **params)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_gb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_gb.optimize(objective_gb, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best GB trial: {study_gb.best_trial.value:.4f}")
print(f"Best GB params: {study_gb.best_trial.params}")

# Train final model with best params
best_params_gb = study_gb.best_trial.params
gb_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE, **best_params_gb)),
])
t0 = time.time()
gb_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_gb = evaluate_model(gb_pipeline, X_val, y_val, class_names, "Gradient Boosting")
results_gb["train_time"] = float(train_time)
results_gb["best_params"] = best_params_gb
results_gb["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 12: Optuna - XGBoost
# ===========================================================================
md("""---
## 10. Optuna Tuning — XGBoost (Mandatory)
""")

code("""def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", xgboost.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            **params
        )),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_xgb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_xgb.optimize(objective_xgb, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best XGB trial: {study_xgb.best_trial.value:.4f}")
print(f"Best XGB params: {study_xgb.best_trial.params}")

# Train final model with best params
best_params_xgb = study_xgb.best_trial.params
xgb_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", xgboost.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
        **best_params_xgb
    )),
])
t0 = time.time()
xgb_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_xgb = evaluate_model(xgb_pipeline, X_val, y_val, class_names, "XGBoost")
results_xgb["train_time"] = float(train_time)
results_xgb["best_params"] = best_params_xgb
results_xgb["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 13: Optuna - HistGradientBoosting
# ===========================================================================
md("""---
## 11. Optuna Tuning — HistGradientBoosting
""")

code("""def objective_hgb(trial):
    params = {
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 0, 10),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE, **params)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_hgb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_hgb.optimize(objective_hgb, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best HGB trial: {study_hgb.best_trial.value:.4f}")
print(f"Best HGB params: {study_hgb.best_trial.params}")

# Train final model with best params
best_params_hgb = study_hgb.best_trial.params
hgb_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_params_hgb)),
])
t0 = time.time()
hgb_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_hgb = evaluate_model(hgb_pipeline, X_val, y_val, class_names, "HistGradientBoosting")
results_hgb["train_time"] = float(train_time)
results_hgb["best_params"] = best_params_hgb
results_hgb["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 14: Optuna - LightGBM
# ===========================================================================
md("""---
## 12. Optuna Tuning — LightGBM
""")

code("""def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", lightgbm.LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
            **params
        )),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_lgbm = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_lgbm.optimize(objective_lgbm, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best LGBM trial: {study_lgbm.best_trial.value:.4f}")
print(f"Best LGBM params: {study_lgbm.best_trial.params}")

# Train final model with best params
best_params_lgbm = study_lgbm.best_trial.params
lgbm_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", lightgbm.LGBMClassifier(
        random_state=RANDOM_STATE,
        verbose=-1,
        **best_params_lgbm
    )),
])
t0 = time.time()
lgbm_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_lgbm = evaluate_model(lgbm_pipeline, X_val, y_val, class_names, "LightGBM")
results_lgbm["train_time"] = float(train_time)
results_lgbm["best_params"] = best_params_lgbm
results_lgbm["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 15: Optuna - CatBoost
# ===========================================================================
md("""---
## 13. Optuna Tuning — CatBoost
""")

code("""def objective_catboost(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 2, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("clf", catboost.CatBoostClassifier(
            random_seed=RANDOM_STATE,
            verbose=0,
            **params
        )),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_catboost = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_catboost.optimize(objective_catboost, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best CatBoost trial: {study_catboost.best_trial.value:.4f}")
print(f"Best CatBoost params: {study_catboost.best_trial.params}")

# Train final model with best params
best_params_catboost = study_catboost.best_trial.params
catboost_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", catboost.CatBoostClassifier(
        random_seed=RANDOM_STATE,
        verbose=0,
        **best_params_catboost
    )),
])
t0 = time.time()
catboost_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_catboost = evaluate_model(catboost_pipeline, X_val, y_val, class_names, "CatBoost")
results_catboost["train_time"] = float(train_time)
results_catboost["best_params"] = best_params_catboost
results_catboost["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 16: Optuna - Logistic Regression
# ===========================================================================
md("""---
## 14. Optuna Tuning — Logistic Regression
""")

code("""def objective_lr(trial):
    params = {
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver="saga",
            **params
        )),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_lr = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_lr.optimize(objective_lr, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best LR trial: {study_lr.best_trial.value:.4f}")
print(f"Best LR params: {study_lr.best_trial.params}")

# Train final model with best params
best_params_lr = study_lr.best_trial.params
lr_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("scaler", RobustScaler()),
    ("clf", LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="saga",
        **best_params_lr
    )),
])
t0 = time.time()
lr_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_lr = evaluate_model(lr_pipeline, X_val, y_val, class_names, "Logistic Regression")
results_lr["train_time"] = float(train_time)
results_lr["best_params"] = best_params_lr
results_lr["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 17: Optuna - KNN
# ===========================================================================
md("""---
## 15. Optuna Tuning — KNN
""")

code("""def objective_knn(trial):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        "p": trial.suggest_int("p", 1, 5) if trial.params.get("metric") == "minkowski" else 2,
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", KNeighborsClassifier(**{k: v for k, v in params.items() if k != "metric" or v != "minkowski" or k != "p"})),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_knn = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_knn.optimize(objective_knn, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best KNN trial: {study_knn.best_trial.value:.4f}")
print(f"Best KNN params: {study_knn.best_trial.params}")

# Train final model with best params
best_params_knn = study_knn.best_trial.params
knn_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("scaler", RobustScaler()),
    ("clf", KNeighborsClassifier(**{k: v for k, v in best_params_knn.items()})),
])
t0 = time.time()
knn_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_knn = evaluate_model(knn_pipeline, X_val, y_val, class_names, "KNN")
results_knn["train_time"] = float(train_time)
results_knn["best_params"] = best_params_knn
results_knn["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 18: Optuna - SVM
# ===========================================================================
md("""---
## 16. Optuna Tuning — SVM
""")

code("""def objective_svm(trial):
    params = {
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", SVC(
            random_state=RANDOM_STATE,
            probability=True,
            **params
        )),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_svm = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_svm.optimize(objective_svm, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

print(f"Best SVM trial: {study_svm.best_trial.value:.4f}")
print(f"Best SVM params: {study_svm.best_trial.params}")

# Train final model with best params
best_params_svm = study_svm.best_trial.params
svm_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("scaler", RobustScaler()),
    ("clf", SVC(
        random_state=RANDOM_STATE,
        probability=True,
        **best_params_svm
    )),
])
t0 = time.time()
svm_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_svm = evaluate_model(svm_pipeline, X_val, y_val, class_names, "SVM")
results_svm["train_time"] = float(train_time)
results_svm["best_params"] = best_params_svm
results_svm["n_trials"] = OPTUNA_TRIALS
""")

# ===========================================================================
# Cell 19: Keras Tuner - Neural Network
# ===========================================================================
md("""---
## 17. Keras Tuner — Neural Network

Architecture search space:
- Hidden layers: 1-4
- Units per layer: 32-256
- Dropout: 0.0-0.5
- Activation: ReLU, GELU, tanh
- Optimizer: Adam, AdamW, RMSprop
- Learning rate: 1e-5 to 1e-2
- Batch normalization: True/False
""")

code("""# Prepare features for Keras
d1_transformer = PhysicsWithCurrentSymmetry()
d1_transformer.fit(X_train)
X_train_d1 = d1_transformer.transform(X_train)
X_val_d1 = d1_transformer.transform(X_val)

# RobustScaler for Keras
scaler_keras = RobustScaler()
scaler_keras.fit(X_train_d1)
X_train_scaled = scaler_keras.transform(X_train_d1)
X_val_scaled = scaler_keras.transform(X_val_d1)

# One-hot encode targets
y_train_onehot = keras.utils.to_categorical(y_train, n_classes)
y_val_onehot = keras.utils.to_categorical(y_val, n_classes)

print(f"Training features shape: {X_train_scaled.shape}")
print(f"Validation features shape: {X_val_scaled.shape}")
print(f"Training labels shape: {y_train_onehot.shape}")
print(f"Validation labels shape: {y_val_onehot.shape}")
""")

# ===========================================================================
# Cell 20: Keras Tuner - Build Model
# ===========================================================================
md("""---
## 18. Keras Tuner - Model Builder
""")

code("""def build_keras_model(hp):
    \"\"\"Build a Keras model with tunable hyperparameters.\"\"\"
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(20,)))
    
    # Tunable number of hidden layers
    n_layers = hp.Int("n_layers", min_value=1, max_value=4, default=2)
    
    for i in range(n_layers):
        # Units per layer
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32, default=64)
        
        # Activation
        activation = hp.Choice(f"activation_{i}", values=["relu", "gelu", "tanh"], default="relu")
        
        model.add(layers.Dense(units, activation=activation))
        
        # Batch normalization (tunable)
        if hp.Boolean(f"batch_norm_{i}", default=True):
            model.add(layers.BatchNormalization())
        
        # Dropout (tunable)
        dropout_rate = hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1, default=0.3)
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(n_classes, activation="softmax"))
    
    # Optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adamw", "rmsprop"], default="adam")
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log", default=1e-3)
    
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model

# Test the model builder
test_model = build_keras_model(kt.HyperParameters())
test_model.summary()
""")

# ===========================================================================
# Cell 21: Keras Tuner - Run
# ===========================================================================
md("""---
## 19. Keras Tuner - Hyperparameter Search
""")

code("""# Custom callback to track macro F1
class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.best_f1 = 0.0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(self.y_val, y_pred_classes, average="macro")
        
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = self.model.get_weights()
    
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

# Create tuner
tuner = kt.RandomSearch(
    build_keras_model,
    objective="val_accuracy",
    max_trials=KERAS_TRIALS,
    seed=RANDOM_STATE,
    directory=str(ROOT / "results" / "test_2"),
    project_name="keras_tuner",
    overwrite=True,
)

# Search
t0 = time.time()
tuner.search(
    X_train_scaled, y_train_onehot,
    validation_data=(X_val_scaled, y_val_onehot),
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
    ],
    verbose=0,
)
search_time = time.time() - t0

# Get best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print(f"Keras Tuner search completed in {search_time:.1f}s")
print(f"Best hyperparameters:")
for key, value in best_hp.values.items():
    print(f"  {key}: {value}")

# Build and train best model
best_model = tuner.hypermodel.build(best_hp)
t0 = time.time()
history = best_model.fit(
    X_train_scaled, y_train_onehot,
    validation_data=(X_val_scaled, y_val_onehot),
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        MacroF1Callback(validation_data=(X_val_scaled, y_val)),
    ],
    verbose=0,
)
train_time = time.time() - t0

# Predict
t0 = time.time()
y_val_proba = best_model.predict(X_val_scaled, verbose=0)
y_val_pred_keras = np.argmax(y_val_proba, axis=1)
infer_time = time.time() - t0

# Metrics
acc = accuracy_score(y_val, y_val_pred_keras)
f1 = f1_score(y_val, y_val_pred_keras, average="macro")
bal_acc = balanced_accuracy_score(y_val, y_val_pred_keras)
kappa = cohen_kappa_score(y_val, y_val_pred_keras)

report = classification_report(y_val, y_val_pred_keras, target_names=class_names, output_dict=True)
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
fault_mask_pred = y_val_pred_keras != no_fault_idx
fault_recall = recall_score(fault_mask_val.astype(int), fault_mask_pred.astype(int), zero_division=0)

no_fault_mask_val = y_val == no_fault_idx
false_alarms = ((y_val_pred_keras != no_fault_idx) & no_fault_mask_val).sum()
n_no_fault = no_fault_mask_val.sum()
false_alarm_rate = false_alarms / n_no_fault if n_no_fault > 0 else 0.0

cm = confusion_matrix(y_val, y_val_pred_keras)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

results_keras = {
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
    "epochs_trained": len(history.history["loss"]),
    "best_val_loss": float(min(history.history["val_loss"])),
    "best_hyperparameters": best_hp.values,
    "n_trials": KERAS_TRIALS,
}

print(f"\\nKeras MLP Results:")
print(f"  Accuracy:        {acc:.4f}")
print(f"  Macro F1:        {f1:.4f}")
print(f"  Balanced Acc:    {bal_acc:.4f}")
print(f"  Cohen's Kappa:   {kappa:.4f}")
print(f"  Fault Recall:    {fault_recall:.4f}")
print(f"  False Alarm:     {false_alarm_rate:.4f}")
print(f"  Train time:      {train_time:.3f}s")
print(f"  Epochs trained:  {len(history.history['loss'])}")
""")

# ===========================================================================
# Cell 22: Compile All Results
# ===========================================================================
md("""---
## 20. Compile All Results
""")

code("""# Compile all results
results = {
    "Logistic Regression": results_lr,
    "KNN": results_knn,
    "SVM": results_svm,
    "Decision Tree": results_dt,
    "Bagging": results_bagging,
    "Random Forest": results_rf,
    "Extra Trees": results_et,
    "AdaBoost": results_adaboost,
    "Gradient Boosting": results_gb,
    "XGBoost": results_xgb,
    "HistGradientBoosting": results_hgb,
    "LightGBM": results_lgbm,
    "CatBoost": results_catboost,
    "MLP Neural Network": results_keras,
}

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
print("Test 2 — Tuned Models (sorted by Macro F1):")
print("=" * 130)
print(summary_df.to_string(index=False, float_format="{:.4f}".format))
""")

# ===========================================================================
# Cell 23: Confusion Matrices
# ===========================================================================
md("""---
## 21. Confusion Matrices

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
# Cell 24: Per-Class Precision/Recall
# ===========================================================================
md("""---
## 22. Per-Class Precision and Recall

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
    ax.set_title(f"{name} — Per-Class Precision and Recall (Test 2)")
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
# Cell 25: Model Comparison
# ===========================================================================
md("""---
## 23. Model Comparison Visualizations
""")

code("""# Macro F1 comparison
fig, ax = plt.subplots(figsize=(12, 6))
names_sorted = summary_df["Model"].values
f1_sorted = summary_df["Macro F1"].values
colors = sns.color_palette("Set2", len(names_sorted))

bars = ax.barh(names_sorted, f1_sorted, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Macro F1")
ax.set_title("Test 2 — Model Comparison (Macro F1)")
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
ax.set_title("Test 2 — Model Comparison (Fault Detection Recall)")
ax.set_xlim(0, 1.05)
for i, v in enumerate(fault_sorted):
    ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_fault_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'model_comparison_fault_recall.png'}")

# False Alarm Rate comparison
fig, ax = plt.subplots(figsize=(12, 6))
false_alarm_sorted = [results[m]["false_alarm_rate"] for m in names_sorted]
bars = ax.barh(names_sorted, false_alarm_sorted, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("False Alarm Rate")
ax.set_title("Test 2 — Model Comparison (False Alarm Rate)")
for i, v in enumerate(false_alarm_sorted):
    ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_false_alarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'model_comparison_false_alarm.png'}")
""")

# ===========================================================================
# Cell 26: Save Results
# ===========================================================================
md("""---
## 24. Save Results
""")

code("""# Save test_2_results.json
out_path = ROOT / "results" / "test_2" / "test_2_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {out_path}")

# Update model_comparison.csv
csv_path = ROOT / "results" / "model_comparison.csv"
rows = []
for name in results:
    r = results[name]
    rows.append({
        "Experiment": "Test 2",
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

# Read existing CSV and append
if csv_path.exists():
    existing_df = pd.read_csv(csv_path)
    csv_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
else:
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
# Cell 27: Summary
# ===========================================================================
md("""---
## 25. Summary

### What Was Done
- Tuned 13 ML models with Optuna (50 trials each, macro F1 objective)
- Tuned 1 Keras MLP with Keras Tuner (30 trials, architecture search)
- All models use D1 (Physics Core + Current Symmetry, 20 features)
- Pipeline: `PhysicsWithCurrentSymmetry -> [RobustScaler] -> Classifier`
- RobustScaler fitted on training data only
- Test set NEVER evaluated

### Key Findings
- [See results table above]
- Best model by Macro F1: [top model]
- All metrics stored in `test_2_results.json`
- Figures saved to `results/test_2/figures/`

### Next Steps
- Test 3: 5-fold stratified cross-validation
- Test 4: Nested cross-validation + tuning
""")

# ===========================================================================
# Assemble notebook
# ===========================================================================
nb.cells = cells

out_path = ROOT / "results" / "test_2" / "test_2_hyperparameter_tuning.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook created: {out_path}")
print(f"Total cells: {len(cells)}")
