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
    classification_report, confusion_matrix, recall_score, precision_score,
    make_scorer
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
)

import xgboost
import lightgbm
import catboost

import optuna
from optuna.samplers import TPESampler

# Try to import Keras, fallback to sklearn MLPClassifier
KERAS_AVAILABLE = False
try:
    import keras
    from keras import layers
    import keras_tuner as kt
    KERAS_AVAILABLE = True
except Exception as e:
    print("Warning: Keras/Tuner not available, using sklearn MLPClassifier:", str(e)[:100])
    from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")
sys.path.insert(0, str(ROOT))
from src.features import PhysicsWithCurrentSymmetry, RAW_FEATURES

FIG_DIR = ROOT / "results" / "test_2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OPTUNA_TRIALS = 15
KERAS_TRIALS = 10

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("ROOT:", ROOT)
print("XGBoost:", xgboost.__version__)
print("LightGBM:", lightgbm.__version__)
print("CatBoost:", catboost.__version__)
if KERAS_AVAILABLE:
    print("Keras:", keras.__version__)
print("Optuna:", optuna.__version__)
if KERAS_AVAILABLE:
    print("Keras Tuner:", kt.__version__)
print("Optuna trials:", OPTUNA_TRIALS)
print("Keras Tuner trials:", KERAS_TRIALS)

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
for cls, count in zip(*np.unique(y, return_counts=True)):
    print("  " + class_names[cls] + ": " + str(count) + " (" + str(100*count/len(y)) + "%)")

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
results["Decision Tree"] = results_dt
print("DT Macro F1:", results_dt["macro_f1"])

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
results["Random Forest"] = results_rf
print("RF Macro F1:", results_rf["macro_f1"])

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
results["Extra Trees"] = results_et
print("ET Macro F1:", results_et["macro_f1"])

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
results["Bagging"] = results_bagging
print("Bagging Macro F1:", results_bagging["macro_f1"])

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
results["AdaBoost"] = results_adaboost
print("AdaBoost Macro F1:", results_adaboost["macro_f1"])

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
results["Gradient Boosting"] = results_gb
print("GB Macro F1:", results_gb["macro_f1"])

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
results["XGBoost"] = results_xgb
print("XGB Macro F1:", results_xgb["macro_f1"])

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
results["HistGradientBoosting"] = results_hgb
print("HGB Macro F1:", results_hgb["macro_f1"])

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
results["LightGBM"] = results_lgbm
print("LGBM Macro F1:", results_lgbm["macro_f1"])

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
results["CatBoost"] = results_catboost
print("CatBoost Macro F1:", results_catboost["macro_f1"])

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
results["Logistic Regression"] = results_lr
print("LR Macro F1:", results_lr["macro_f1"])

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
results_knn["best_params"] = best_params_knn
results_knn["n_trials"] = OPTUNA_TRIALS
results["KNN"] = results_knn
print("KNN Macro F1:", results_knn["macro_f1"])

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
results["SVM"] = results_svm
print("SVM Macro F1:", results_svm["macro_f1"])

# Neural Network
print("\n--- Neural Network ---")
if KERAS_AVAILABLE:
    print("Using Keras Tuner...")
    d1_transformer = PhysicsWithCurrentSymmetry()
    d1_transformer.fit(X_train)
    X_train_d1 = d1_transformer.transform(X_train)
    X_val_d1 = d1_transformer.transform(X_val)

    scaler_keras = RobustScaler()
    scaler_keras.fit(X_train_d1)
    X_train_scaled = scaler_keras.transform(X_train_d1)
    X_val_scaled = scaler_keras.transform(X_val_d1)

    y_train_onehot = keras.utils.to_categorical(y_train, n_classes)
    y_val_onehot = keras.utils.to_categorical(y_val, n_classes)

    def build_keras_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(20,)))
        
        n_layers = hp.Int("n_layers", min_value=1, max_value=4, default=2)
        
        for i in range(n_layers):
            units = hp.Int("units_" + str(i), min_value=32, max_value=256, step=32, default=64)
            activation = hp.Choice("activation_" + str(i), values=["relu", "gelu", "tanh"], default="relu")
            
            model.add(layers.Dense(units, activation=activation))
            
            if hp.Boolean("batch_norm_" + str(i), default=True):
                model.add(layers.BatchNormalization())
            
            dropout_rate = hp.Float("dropout_" + str(i), min_value=0.0, max_value=0.5, step=0.1, default=0.3)
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(n_classes, activation="softmax"))
        
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

    tuner = kt.RandomSearch(
        build_keras_model,
        objective="val_accuracy",
        max_trials=KERAS_TRIALS,
        seed=RANDOM_STATE,
        directory=str(ROOT / "results" / "test_2"),
        project_name="keras_tuner",
        overwrite=True,
    )

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

    best_hp = tuner.get_best_hyperparameters()[0]
    print("Keras Tuner search completed in", search_time, "s")

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
        ],
        verbose=0,
    )
    train_time = time.time() - t0

    t0 = time.time()
    y_val_proba = best_model.predict(X_val_scaled, verbose=0)
    y_val_pred_nn = np.argmax(y_val_proba, axis=1)
    infer_time = time.time() - t0

    extra_info = {
        "epochs_trained": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_hyperparameters": best_hp.values,
        "n_trials": KERAS_TRIALS,
    }
else:
    print("Using sklearn MLPClassifier with Optuna...")
    def objective_mlp(trial):
        hidden_layers = []
        n_layers = trial.suggest_int("n_layers", 1, 3)
        for i in range(n_layers):
            hidden_layers.append(trial.suggest_int("units_" + str(i), 32, 256, step=32))
        
        params = {
            "hidden_layer_sizes": tuple(hidden_layers),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
        }
        
        pipeline = Pipeline([
            ("features", PhysicsWithCurrentSymmetry()),
            ("scaler", RobustScaler()),
            ("clf", MLPClassifier(random_state=RANDOM_STATE, **params)),
        ])
        
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        if y_val_pred.ndim > 1:
            y_val_pred = y_val_pred.flatten()
        
        return f1_score(y_val, y_val_pred, average="macro")

    study_mlp = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study_mlp.optimize(objective_mlp, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params_mlp = study_mlp.best_trial.params
    hidden_layers = []
    n_layers = best_params_mlp.pop("n_layers")
    for i in range(n_layers):
        hidden_layers.append(best_params_mlp.pop("units_" + str(i)))
    best_params_mlp["hidden_layer_sizes"] = tuple(hidden_layers)
    
    mlp_pipeline = Pipeline([
        ("features", PhysicsWithCurrentSymmetry()),
        ("scaler", RobustScaler()),
        ("clf", MLPClassifier(random_state=RANDOM_STATE, **best_params_mlp)),
    ])
    t0 = time.time()
    mlp_pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_val_pred_nn = mlp_pipeline.predict(X_val)
    infer_time = time.time() - t0
    if y_val_pred_nn.ndim > 1:
        y_val_pred_nn = y_val_pred_nn.flatten()

    extra_info = {
        "n_trials": OPTUNA_TRIALS,
        "best_params": best_params_mlp,
    }

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

results_nn = {
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
results_nn.update(extra_info)

print("Neural Network Macro F1:", f1)
results["MLP Neural Network"] = results_nn

# Save results
out_path = ROOT / "results" / "test_2" / "test_2_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to:", out_path)

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
print("\nTest 2 — Tuned Models (sorted by Macro F1):")
print("=" * 130)
print(summary_df.to_string(index=False, float_format="{:.4f}".format))

# Confusion matrices
for name in results:
    r = results[name]
    cm = np.array(r["confusion_matrix"])
    cm_norm = np.array(r["confusion_matrix_normalized"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[0])
    axes[0].set_title(name + " — Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(name + " — Confusion Matrix (Row-Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(FIG_DIR / ("confusion_matrix_" + safe_name + ".png"), dpi=150, bbox_inches="tight")
    plt.close()

print("\nSaved", len(results), "confusion matrix pairs to:", FIG_DIR)

# Per-class PR
for name in results:
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
    ax.set_title(name + " — Per-Class Precision and Recall (Test 2)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(FIG_DIR / ("per_class_pr_" + safe_name + ".png"), dpi=150, bbox_inches="tight")
    plt.close()

print("Saved", len(results), "per-class PR charts to:", FIG_DIR)

# Model comparison
fig, ax = plt.subplots(figsize=(12, 6))
names_sorted = summary_df["Model"].values
f1_sorted = summary_df["Macro F1"].values
colors = sns.color_palette("Set2", len(names_sorted))

bars = ax.barh(names_sorted, f1_sorted, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Macro F1")
ax.set_title("Test 2 — Model Comparison (Macro F1)")
ax.set_xlim(0, 1.05)
for i, v in enumerate(f1_sorted):
    ax.text(v + 0.005, i, "{:.4f}".format(v), va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", FIG_DIR / "model_comparison_f1.png")

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

if csv_path.exists():
    existing_df = pd.read_csv(csv_path)
    csv_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
else:
    csv_df = pd.DataFrame(rows)

csv_df.to_csv(csv_path, index=False)
print("Model comparison saved to:", csv_path)

print("\nTest 2 completed!")
