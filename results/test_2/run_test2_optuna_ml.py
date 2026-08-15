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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
)

import xgboost
import lightgbm
import catboost

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
OPTUNA_TRIALS = 10  # Reduced for initial testing

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("ROOT:", ROOT)
print("XGBoost:", xgboost.__version__)
print("LightGBM:", lightgbm.__version__)
print("CatBoost:", catboost.__version__)
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

results = {}
best_params_dict = {}

# Decision Tree
print("\n--- Decision Tree Optuna ---")
def objective_dt(trial):
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

best_params_dt = study_dt.best_trial.params
best_params_dict["Decision Tree"] = best_params_dt
print("Best DT Macro F1:", study_dt.best_trial.value)
print("Best DT params:", best_params_dt)

dt_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE, **best_params_dt)),
])
t0 = time.time()
dt_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_dt = evaluate_model(dt_pipeline, X_val, y_val, class_names, "Decision Tree")
results_dt["train_time"] = float(train_time)
results_dt["n_trials"] = OPTUNA_TRIALS
results_dt["best_params"] = best_params_dt
results["Decision Tree"] = results_dt

# Save checkpoint after each model
checkpoint_path = ROOT / "results" / "test_2" / "test_2_checkpoint.json"
with open(checkpoint_path, "w") as f:
    json.dump(results, f, indent=2)
print("Checkpoint saved after Decision Tree")

# Random Forest
print("\n--- Random Forest Optuna ---")
def objective_rf(trial):
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

best_params_rf = study_rf.best_trial.params
best_params_dict["Random Forest"] = best_params_rf
print("Best RF Macro F1:", study_rf.best_trial.value)
print("Best RF params:", best_params_rf)

rf_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **best_params_rf)),
])
t0 = time.time()
rf_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_rf = evaluate_model(rf_pipeline, X_val, y_val, class_names, "Random Forest")
results_rf["train_time"] = float(train_time)
results_rf["n_trials"] = OPTUNA_TRIALS
results_rf["best_params"] = best_params_rf
results["Random Forest"] = results_rf

# Extra Trees
print("\n--- Extra Trees Optuna ---")
def objective_et(trial):
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

best_params_et = study_et.best_trial.params
best_params_dict["Extra Trees"] = best_params_et
print("Best ET Macro F1:", study_et.best_trial.value)
print("Best ET params:", best_params_et)

et_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", ExtraTreesClassifier(random_state=RANDOM_STATE, **best_params_et)),
])
t0 = time.time()
et_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_et = evaluate_model(et_pipeline, X_val, y_val, class_names, "Extra Trees")
results_et["train_time"] = float(train_time)
results_et["n_trials"] = OPTUNA_TRIALS
results_et["best_params"] = best_params_et
results["Extra Trees"] = results_et

# Bagging
print("\n--- Bagging Optuna ---")
def objective_bagging(trial):
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

best_params_bagging = study_bagging.best_trial.params
best_params_dict["Bagging"] = best_params_bagging
print("Best Bagging Macro F1:", study_bagging.best_trial.value)
print("Best Bagging params:", best_params_bagging)

bagging_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", BaggingClassifier(random_state=RANDOM_STATE, **best_params_bagging)),
])
t0 = time.time()
bagging_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_bagging = evaluate_model(bagging_pipeline, X_val, y_val, class_names, "Bagging")
results_bagging["train_time"] = float(train_time)
results_bagging["n_trials"] = OPTUNA_TRIALS
results_bagging["best_params"] = best_params_bagging
results["Bagging"] = results_bagging

# AdaBoost
print("\n--- AdaBoost Optuna ---")
def objective_adaboost(trial):
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

best_params_adaboost = study_adaboost.best_trial.params
best_params_dict["AdaBoost"] = best_params_adaboost
print("Best AdaBoost Macro F1:", study_adaboost.best_trial.value)
print("Best AdaBoost params:", best_params_adaboost)

adaboost_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", AdaBoostClassifier(random_state=RANDOM_STATE, **best_params_adaboost)),
])
t0 = time.time()
adaboost_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_adaboost = evaluate_model(adaboost_pipeline, X_val, y_val, class_names, "AdaBoost")
results_adaboost["train_time"] = float(train_time)
results_adaboost["n_trials"] = OPTUNA_TRIALS
results_adaboost["best_params"] = best_params_adaboost
results["AdaBoost"] = results_adaboost

# Gradient Boosting
print("\n--- Gradient Boosting Optuna ---")
def objective_gb(trial):
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

best_params_gb = study_gb.best_trial.params
best_params_dict["Gradient Boosting"] = best_params_gb
print("Best GB Macro F1:", study_gb.best_trial.value)
print("Best GB params:", best_params_gb)

gb_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE, **best_params_gb)),
])
t0 = time.time()
gb_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_gb = evaluate_model(gb_pipeline, X_val, y_val, class_names, "Gradient Boosting")
results_gb["train_time"] = float(train_time)
results_gb["n_trials"] = OPTUNA_TRIALS
results_gb["best_params"] = best_params_gb
results["Gradient Boosting"] = results_gb

# XGBoost
print("\n--- XGBoost Optuna ---")
def objective_xgb(trial):
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

best_params_xgb = study_xgb.best_trial.params
best_params_dict["XGBoost"] = best_params_xgb
print("Best XGB Macro F1:", study_xgb.best_trial.value)
print("Best XGB params:", best_params_xgb)

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
results_xgb["n_trials"] = OPTUNA_TRIALS
results_xgb["best_params"] = best_params_xgb
results["XGBoost"] = results_xgb

# HistGradientBoosting
print("\n--- HistGradientBoosting Optuna ---")
def objective_hgb(trial):
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

best_params_hgb = study_hgb.best_trial.params
best_params_dict["HistGradientBoosting"] = best_params_hgb
print("Best HGB Macro F1:", study_hgb.best_trial.value)
print("Best HGB params:", best_params_hgb)

hgb_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_params_hgb)),
])
t0 = time.time()
hgb_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_hgb = evaluate_model(hgb_pipeline, X_val, y_val, class_names, "HistGradientBoosting")
results_hgb["train_time"] = float(train_time)
results_hgb["n_trials"] = OPTUNA_TRIALS
results_hgb["best_params"] = best_params_hgb
results["HistGradientBoosting"] = results_hgb

# LightGBM
print("\n--- LightGBM Optuna ---")
def objective_lgbm(trial):
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

best_params_lgbm = study_lgbm.best_trial.params
best_params_dict["LightGBM"] = best_params_lgbm
print("Best LGBM Macro F1:", study_lgbm.best_trial.value)
print("Best LGBM params:", best_params_lgbm)

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
results_lgbm["n_trials"] = OPTUNA_TRIALS
results_lgbm["best_params"] = best_params_lgbm
results["LightGBM"] = results_lgbm

# CatBoost
print("\n--- CatBoost Optuna ---")
def objective_catboost(trial):
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

best_params_catboost = study_catboost.best_trial.params
best_params_dict["CatBoost"] = best_params_catboost
print("Best CatBoost Macro F1:", study_catboost.best_trial.value)
print("Best CatBoost params:", best_params_catboost)

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
results_catboost["n_trials"] = OPTUNA_TRIALS
results_catboost["best_params"] = best_params_catboost
results["CatBoost"] = results_catboost

# Logistic Regression
print("\n--- Logistic Regression Optuna ---")
def objective_lr(trial):
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

best_params_lr = study_lr.best_trial.params
best_params_dict["Logistic Regression"] = best_params_lr
print("Best LR Macro F1:", study_lr.best_trial.value)
print("Best LR params:", best_params_lr)

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
results_lr["n_trials"] = OPTUNA_TRIALS
results_lr["best_params"] = best_params_lr
results["Logistic Regression"] = results_lr

# KNN
print("\n--- KNN Optuna ---")
def objective_knn(trial):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 5),
    }
    
    pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", KNeighborsClassifier(**params)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    if y_val_pred.ndim > 1:
        y_val_pred = y_val_pred.flatten()
    
    return f1_score(y_val, y_val_pred, average="macro")

study_knn = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_knn.optimize(objective_knn, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

best_params_knn = study_knn.best_trial.params
best_params_dict["KNN"] = best_params_knn
print("Best KNN Macro F1:", study_knn.best_trial.value)
print("Best KNN params:", best_params_knn)

knn_pipeline = Pipeline([
    ("features", PhysicsWithCurrentSymmetry()),
    ("scaler", RobustScaler()),
    ("clf", KNeighborsClassifier(**best_params_knn)),
])
t0 = time.time()
knn_pipeline.fit(X_train, y_train)
train_time = time.time() - t0

results_knn = evaluate_model(knn_pipeline, X_val, y_val, class_names, "KNN")
results_knn["train_time"] = float(train_time)
results_knn["n_trials"] = OPTUNA_TRIALS
results_knn["best_params"] = best_params_knn
results["KNN"] = results_knn

# SVM
print("\n--- SVM Optuna ---")
def objective_svm(trial):
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

best_params_svm = study_svm.best_trial.params
best_params_dict["SVM"] = best_params_svm
print("Best SVM Macro F1:", study_svm.best_trial.value)
print("Best SVM params:", best_params_svm)

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
results_svm["n_trials"] = OPTUNA_TRIALS
results_svm["best_params"] = best_params_svm
results["SVM"] = results_svm

# Save intermediate results
out_path = ROOT / "results" / "test_2" / "test_2_results_intermediate.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nIntermediate ML results saved to:", out_path)

# Save best params
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
print("Total Optuna trials per model:", OPTUNA_TRIALS)
