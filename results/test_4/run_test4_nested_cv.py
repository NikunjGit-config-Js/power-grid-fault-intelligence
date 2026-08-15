"""Test 4: Nested Cross-Validation + Hyperparameter Optimization
Direct Python script with checkpointing after each outer fold.
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
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
)

import xgboost
import lightgbm
import catboost

import optuna
from optuna.samplers import TPESampler

import keras
from keras import layers
import keras_tuner as kt

import warnings
warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")
sys.path.insert(0, str(ROOT))
from src.features import PhysicsWithCurrentSymmetry, RAW_FEATURES

FIG_DIR = ROOT / "results" / "test_4" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "test_4" / "keras_tuner").mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
OPTUNA_TRIALS = 10
KERAS_TRIALS = 10

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"ROOT: {ROOT}")
print(f"XGBoost: {xgboost.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
print(f"CatBoost: {catboost.__version__}")
print(f"Keras: {keras.__version__}")
print(f"Keras Tuner: {kt.__version__}")
print(f"Optuna: {optuna.__version__}")

# ======================================================================
# Data Loading
# ======================================================================
df = pd.read_csv(ROOT / "classData.csv")
print(f"Dataset shape: {df.shape}")

FAULT_CODE_TO_LABEL = {
    "0000": "No Fault", "1001": "Line-to-Line AB", "0110": "LL-G BC",
    "0111": "Three-Phase", "1011": "LG A+B-G", "1111": "3P-G",
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

# ======================================================================
# Load Split Indices
# ======================================================================
split_data = np.load(ROOT / "results" / "test_1" / "split_indices.npz")
train_idx = split_data["train_idx"]
val_idx = split_data["val_idx"]
test_idx = split_data["test_idx"]

dev_idx = np.concatenate([train_idx, val_idx])
X_dev = X_raw[dev_idx]
y_dev = y[dev_idx]

print(f"Dev: {len(X_dev)}, Test: {len(test_idx)} [LOCKED]")

# ======================================================================
# CV Definitions
# ======================================================================
outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
outer_folds = list(outer_cv.split(X_dev, y_dev))

# ======================================================================
# Evaluation Helpers
# ======================================================================
def evaluate_fold(y_true, y_pred, class_names):
    if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    per_class = {}
    for cls_name in class_names:
        per_class[cls_name] = {
            "precision": report[cls_name]["precision"],
            "recall": report[cls_name]["recall"],
            "f1-score": report[cls_name]["f1-score"],
            "support": int(report[cls_name]["support"]),
        }
    no_fault_idx = list(class_names).index("No Fault")
    fault_mask_true = y_true != no_fault_idx
    fault_mask_pred = y_pred != no_fault_idx
    fault_recall = recall_score(fault_mask_true.astype(int), fault_mask_pred.astype(int), zero_division=0)
    no_fault_mask_true = y_true == no_fault_idx
    false_alarms = ((y_pred != no_fault_idx) & no_fault_mask_true).sum()
    n_no_fault = no_fault_mask_true.sum()
    false_alarm_rate = false_alarms / n_no_fault if n_no_fault > 0 else 0.0
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(acc), "macro_f1": float(f1), "balanced_accuracy": float(bal_acc),
        "cohen_kappa": float(kappa), "per_class": per_class,
        "fault_detection_recall": float(fault_recall), "false_alarm_rate": float(false_alarm_rate),
        "confusion_matrix": cm.tolist(), "confusion_matrix_normalized": cm_norm.tolist(),
    }

# ======================================================================
# ML Model Factories
# ======================================================================
def make_pipeline_lr(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()), ("scaler", RobustScaler()),
                     ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, **params))])

def make_pipeline_knn(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()), ("scaler", RobustScaler()),
                     ("clf", KNeighborsClassifier(**params))])

def make_pipeline_svm(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()), ("scaler", RobustScaler()),
                     ("clf", SVC(random_state=RANDOM_STATE, **params))])

def make_pipeline_dt(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE, **params))])

def make_pipeline_bagging(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", BaggingClassifier(random_state=RANDOM_STATE, **params))])

def make_pipeline_rf(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **params))])

def make_pipeline_et(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", ExtraTreesClassifier(random_state=RANDOM_STATE, **params))])

def make_pipeline_ab(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", AdaBoostClassifier(random_state=RANDOM_STATE, **params))])

def make_pipeline_gb(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE, **params))])

def make_pipeline_xgb(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", xgboost.XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss",
                                                   use_label_encoder=False, verbosity=0, **params))])

def make_pipeline_hgb(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE, **params))])

def make_pipeline_lgb(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", lightgbm.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **params))])

def make_pipeline_cb(params):
    return Pipeline([("features", PhysicsWithCurrentSymmetry()),
                     ("clf", catboost.CatBoostClassifier(random_seed=RANDOM_STATE, verbose=0, **params))])

# Search spaces
def suggest_lr(trial):
    return {"C": trial.suggest_float("C", 1e-4, 100, log=True),
            "penalty": "l2", "solver": trial.suggest_categorical("solver", ["lbfgs", "newton-cg", "sag", "saga"])}

def suggest_knn(trial):
    return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"])}

def suggest_svm(trial):
    return {"C": trial.suggest_float("C", 1e-2, 100, log=True),
            "kernel": "rbf",
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])}

def suggest_dt(trial):
    return {"max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])}

def suggest_bagging(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])}

def suggest_rf(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])}

def suggest_et(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])}

def suggest_ab(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True)}

def suggest_gb(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0)}

def suggest_xgb(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)}

def suggest_hgb(trial):
    return {"max_iter": trial.suggest_int("max_iter", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0)}

def suggest_lgb(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50)}

def suggest_cb(trial):
    return {"iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)}

ml_model_registry = {
    "Logistic Regression": (make_pipeline_lr, suggest_lr),
    "KNN": (make_pipeline_knn, suggest_knn),
    "SVM": (make_pipeline_svm, suggest_svm),
    "Decision Tree": (make_pipeline_dt, suggest_dt),
    "Bagging": (make_pipeline_bagging, suggest_bagging),
    "Random Forest": (make_pipeline_rf, suggest_rf),
    "Extra Trees": (make_pipeline_et, suggest_et),
    "AdaBoost": (make_pipeline_ab, suggest_ab),
    "Gradient Boosting": (make_pipeline_gb, suggest_gb),
    "XGBoost": (make_pipeline_xgb, suggest_xgb),
    "HistGradientBoosting": (make_pipeline_hgb, suggest_hgb),
    "LightGBM": (make_pipeline_lgb, suggest_lgb),
    "CatBoost": (make_pipeline_cb, suggest_cb),
}

# ======================================================================
# Keras Model Building + InnerCVTuner
# ======================================================================
class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, X_val=None, y_val=None, validation_data=None):
        super().__init__()
        if validation_data is not None:
            self.X_val, self.y_val = validation_data
        else:
            self.X_val, self.y_val = X_val, y_val
        self.best_f1 = 0.0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(self.y_val, y_pred_classes, average="macro")
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

def build_keras_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(20,)))
    n_layers = hp.Int("n_layers", min_value=1, max_value=4, default=2)
    for i in range(n_layers):
        units = hp.Int("units_" + str(i), min_value=32, max_value=256, step=32, default=64)
        activation = hp.Choice("activation_" + str(i), values=["relu", "gelu", "tanh"], default="relu")
        model.add(layers.Dense(units, activation=activation))
        if hp.Boolean("batch_norm_" + str(i), default=False):
            model.add(layers.BatchNormalization())
        dropout_rate = hp.Float("dropout_" + str(i), min_value=0.0, max_value=0.5, step=0.1, default=0.0)
        if dropout_rate > 0:
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
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

class FaultHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_keras_model(hp)

class InnerCVTuner(kt.RandomSearch):
    def __init__(self, hypermodel, X_outer_train, y_outer_train, inner_cv, **kwargs):
        super().__init__(hypermodel, **kwargs)
        self.X_outer_train = X_outer_train
        self.y_outer_train = y_outer_train
        self.inner_cv = inner_cv
        self._trial_best_epochs = {}  # trial_id -> median best epoch

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        inner_scores = []
        inner_best_epochs = []
        for inner_tr_idx, inner_val_idx in self.inner_cv.split(self.X_outer_train, self.y_outer_train):
            X_it = self.X_outer_train[inner_tr_idx]
            y_it = self.y_outer_train[inner_tr_idx]
            X_iv = self.X_outer_train[inner_val_idx]
            y_iv = self.y_outer_train[inner_val_idx]
            d1 = PhysicsWithCurrentSymmetry()
            scaler = RobustScaler()
            X_it_d1 = d1.transform(X_it)
            X_iv_d1 = d1.transform(X_iv)
            scaler.fit(X_it_d1)
            X_it_s = scaler.transform(X_it_d1)
            X_iv_s = scaler.transform(X_iv_d1)
            model = self.hypermodel.build(hp)
            y_it_oh = keras.utils.to_categorical(y_it, n_classes)
            y_iv_oh = keras.utils.to_categorical(y_iv, n_classes)
            cb = MacroF1Callback(validation_data=(X_iv_s, y_iv))
            model.fit(X_it_s, y_it_oh, validation_data=(X_iv_s, y_iv_oh),
                      epochs=100, batch_size=32, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                               restore_best_weights=True), cb])
            inner_scores.append(cb.best_f1)
            inner_best_epochs.append(cb.best_epoch)
            del model
            gc.collect()
        median_epoch = int(np.median(inner_best_epochs))
        self._trial_best_epochs[trial.trial_id] = median_epoch
        return {"val_macro_f1": float(np.mean(inner_scores))}

# ======================================================================
# Main Nested CV Loop with Checkpointing
# ======================================================================
checkpoint_path = ROOT / "results" / "test_4" / "test_4_checkpoint.json"

# Load checkpoint if exists
if checkpoint_path.exists():
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
    all_outer_results = checkpoint.get("all_outer_results", {})
    best_params_all = checkpoint.get("best_params_all", {})
    completed_folds = checkpoint.get("completed_folds", 0)
    print(f"Loaded checkpoint: {completed_folds} folds completed")
else:
    all_outer_results = {name: [] for name in ml_model_registry}
    all_outer_results["MLP Neural Network"] = []
    best_params_all = {name: [] for name in ml_model_registry}
    best_params_all["MLP Neural Network"] = []
    completed_folds = 0

model_order = list(ml_model_registry.keys()) + ["MLP Neural Network"]
ml_model_names = list(ml_model_registry.keys())

for outer_i in range(completed_folds, N_OUTER_FOLDS):
    outer_tr_idx, outer_val_idx = outer_folds[outer_i]
    X_ot = X_dev[outer_tr_idx]
    y_ot = y_dev[outer_tr_idx]
    X_ov = X_dev[outer_val_idx]
    y_ov = y_dev[outer_val_idx]

    print(f"\n{'='*60}")
    print(f"OUTER FOLD {outer_i+1}/{N_OUTER_FOLDS}: train={len(X_ot)}, val={len(X_ov)}")
    print(f"{'='*60}")
    fold_start = time.time()

    # Determine which ML models are already completed for this fold
    already_done_ml = set()
    mlp_done = False
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        # ML models: only skip if we're resuming the same partial fold
        if outer_i == completed_folds:
            already_done_ml = set(ckpt.get("completed_ml_models", []))
        # MLP: skip if results already have an entry for this fold index
        mlp_results = ckpt.get("all_outer_results", {}).get("MLP Neural Network", [])
        mlp_done = len(mlp_results) > outer_i
        if already_done_ml or mlp_done:
            print(f"  [Resuming] Skipping already completed: {already_done_ml} {'MLP' if mlp_done else ''}")

    # --- ML Models (Optuna) ---
    for name, (make_pipe_fn, suggest_fn) in ml_model_registry.items():
        if name in already_done_ml:
            print(f"  {name}... [CACHED]")
            continue
        print(f"  {name}...", end=" ", flush=True)

        def make_obj(mt, st, Xot, yot):
            def obj(trial):
                params = st(trial)
                scores = []
                for itr, ivl in inner_cv.split(Xot, yot):
                    p = mt(params)
                    p.fit(Xot[itr], yot[itr])
                    yp = p.predict(Xot[ivl])
                    scores.append(f1_score(yot[ivl], yp, average="macro"))
                return np.mean(scores)
            return obj

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
        study.optimize(make_obj(make_pipe_fn, suggest_fn, X_ot, y_ot),
                       n_trials=OPTUNA_TRIALS, show_progress_bar=False)

        best_p = study.best_trial.params
        best_params_all[name].append(best_p)

        pipe = make_pipe_fn(best_p)
        t0 = time.time()
        pipe.fit(X_ot, y_ot)
        train_t = time.time() - t0
        t0 = time.time()
        yp = pipe.predict(X_ov)
        infer_t = time.time() - t0

        m = evaluate_fold(y_ov, yp, class_names)
        m["train_time"] = float(train_t)
        m["inference_time"] = float(infer_t)
        all_outer_results[name].append(m)
        elapsed = time.time() - fold_start
        print(f"F1={m['macro_f1']:.4f} ({elapsed:.1f}s elapsed)")

        # Save checkpoint after each model in outer fold
        checkpoint = {
            "all_outer_results": all_outer_results,
            "best_params_all": best_params_all,
            "completed_folds": outer_i,
            "completed_ml_models": list(ml_model_registry.keys())[:list(ml_model_registry.keys()).index(name)+1],
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        gc.collect()

    # --- Keras MLP (Keras Tuner) ---
    if mlp_done:
        print(f"  MLP Neural Network... [CACHED]")
    else:
        print(f"  MLP Neural Network...", end=" ", flush=True)

        hypermodel = FaultHyperModel()
        tuner = InnerCVTuner(
            hypermodel=hypermodel,
            X_outer_train=X_ot,
            y_outer_train=y_ot,
            inner_cv=inner_cv,
            objective=kt.Objective("val_macro_f1", direction="max"),
            max_trials=KERAS_TRIALS,
            seed=RANDOM_STATE,
            overwrite=True,
            directory=str(ROOT / "results" / "test_4" / "keras_tuner"),
            project_name=f"outer_fold_{outer_i}",
        )
        tuner.search(verbose=0)

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_params_all["MLP Neural Network"].append(best_hp.values)

        # Derive fixed epoch count from inner-fold best epochs (median)
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        fixed_epochs = tuner._trial_best_epochs.get(best_trial.trial_id, 50)
        print(f"[fixed_epochs={fixed_epochs}]", end=" ", flush=True)

        # Fresh scaler fitted ONLY on outer training fold
        d1 = PhysicsWithCurrentSymmetry()
        scaler = RobustScaler()
        X_ot_d1 = d1.transform(X_ot)
        X_ov_d1 = d1.transform(X_ov)
        scaler.fit(X_ot_d1)
        X_ot_s = scaler.transform(X_ot_d1)
        X_ov_s = scaler.transform(X_ov_d1)

        # Fresh model, train on outer training ONLY for fixed epochs
        # NO validation_data, NO EarlyStopping, NO MacroF1Callback
        model = build_keras_model(best_hp)
        y_ot_oh = keras.utils.to_categorical(y_ot, n_classes)

        t0 = time.time()
        model.fit(X_ot_s, y_ot_oh, epochs=fixed_epochs, batch_size=32, verbose=0)
        train_t = time.time() - t0

        # Predict on outer validation ONCE after training is complete
        t0 = time.time()
        yp = np.argmax(model.predict(X_ov_s, verbose=0), axis=1)
        infer_t = time.time() - t0

        m = evaluate_fold(y_ov, yp, class_names)
        m["train_time"] = float(train_t)
        m["inference_time"] = float(infer_t)
        m["fixed_epochs"] = fixed_epochs
        all_outer_results["MLP Neural Network"].append(m)
        elapsed = time.time() - fold_start
        print(f"F1={m['macro_f1']:.4f} ({elapsed:.1f}s elapsed)")
        del model
        gc.collect()

    # Save checkpoint after each outer fold
    completed_folds = outer_i + 1
    checkpoint = {
        "all_outer_results": all_outer_results,
        "best_params_all": best_params_all,
        "completed_folds": completed_folds,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)
    print(f"  Checkpoint saved: {completed_folds}/{N_OUTER_FOLDS} folds")
    gc.collect()

print("\n\nNested CV complete!")
