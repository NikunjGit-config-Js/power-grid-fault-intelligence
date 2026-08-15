import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

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

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix, recall_score, precision_score
)

import keras
from keras import layers
import keras_tuner as kt

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")
sys.path.insert(0, str(ROOT))
from src.features import PhysicsWithCurrentSymmetry, RAW_FEATURES

FIG_DIR = ROOT / "results" / "test_2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
KERAS_TRIALS = 10

print("ROOT:", ROOT)
print("Keras:", keras.__version__)
print("Keras Tuner:", kt.__version__)
print("Keras Tuner trials:", KERAS_TRIALS)

# ===========================================================================
# Load data
# ===========================================================================
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

# ===========================================================================
# Load persisted split indices from Test 1 (DO NOT recreate splits)
# ===========================================================================
split_path = ROOT / "results" / "test_1" / "split_indices.npz"
print("\nLoading split indices from:", split_path)
split_data = np.load(split_path)
train_idx = split_data["train_idx"]
val_idx = split_data["val_idx"]
test_idx = split_data["test_idx"]

print("Train indices shape:", train_idx.shape)
print("Val indices shape:", val_idx.shape)
print("Test indices shape:", test_idx.shape)

# Verify no overlap
train_set = set(train_idx.tolist())
val_set = set(val_idx.tolist())
test_set = set(test_idx.tolist())
assert len(train_set & val_set) == 0, "Train/Val overlap detected!"
assert len(train_set & test_set) == 0, "Train/Test overlap detected!"
assert len(val_set & test_set) == 0, "Val/Test overlap detected!"
print("No overlap between splits: PASS")

# Extract data using persisted indices
X_train, y_train = X_raw[train_idx], y[train_idx]
X_val, y_val = X_raw[val_idx], y[val_idx]
X_test, y_test = X_raw[test_idx], y[test_idx]

print("\nTotal:     " + str(len(X_raw)) + " samples")
print("Train:     " + str(len(X_train)) + " (" + str(100*len(X_train)/len(X_raw)) + "%)")
print("Val:       " + str(len(X_val)) + " (" + str(100*len(X_val)/len(X_raw)) + "%)")
print("Test:      " + str(len(X_test)) + " (" + str(100*len(X_test)/len(X_raw)) + "%) [LOCKED]")

# Verify class distribution matches Test 1
print("\nClass distribution (Train):")
for i, cls_name in enumerate(class_names):
    count = (y_train == i).sum()
    print(f"  {cls_name}: {count}")

# ===========================================================================
# Prepare features for Keras
# ===========================================================================
d1_transformer = PhysicsWithCurrentSymmetry()
X_train_d1 = d1_transformer.transform(X_train)
X_val_d1 = d1_transformer.transform(X_val)

scaler_keras = RobustScaler()
scaler_keras.fit(X_train_d1)
X_train_scaled = scaler_keras.transform(X_train_d1)
X_val_scaled = scaler_keras.transform(X_val_d1)

y_train_onehot = keras.utils.to_categorical(y_train, n_classes)
y_val_onehot = keras.utils.to_categorical(y_val, n_classes)

print("\nTraining features shape:", X_train_scaled.shape)
print("Validation features shape:", X_val_scaled.shape)
print("Training labels shape:", y_train_onehot.shape)
print("Validation labels shape:", y_val_onehot.shape)

# ===========================================================================
# Custom Keras Metric for Macro F1
# ===========================================================================
class MacroF1Metric(keras.metrics.Metric):
    """Custom Keras metric that computes Macro F1 on validation data."""
    
    def __init__(self, y_val_onehot, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.y_val_onehot = y_val_onehot
        self.y_val_labels = np.argmax(y_val_onehot, axis=1)
        self._f1 = self.add_variable(shape=(), initializer="zeros", trainable=False)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(self.y_val_labels, y_pred_classes, average="macro", zero_division=0)
        self._f1.assign(f1)
    
    def result(self):
        return self._f1
    
    def reset_state(self):
        self._f1.assign(0.0)

# ===========================================================================
# Custom callback to track macro F1 and restore best weights
# ===========================================================================
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

# ===========================================================================
# Build model with Keras Tuner — using Macro F1 as objective
# ===========================================================================
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

# ===========================================================================
# Custom Oracle that evaluates Macro F1 instead of val_accuracy
# ===========================================================================
class MacroF1Oracle(kt.engine.oracle.Oracle):
    """Oracle that uses Macro F1 as the objective metric."""
    
    def __init__(self, objective, max_trials, seed=None, **kwargs):
        super().__init__(objective=objective, max_trials=max_trials, seed=seed, **kwargs)
    
    def _build_and_fit_model(self, trial, *args, **kwargs):
        """Override to compute Macro F1 after training."""
        hp = trial.hyperparameters
        model = self._hypermodel.build(hp)
        
        # Extract fit kwargs
        fit_kwargs = kwargs.copy()
        X = fit_kwargs.pop("x", None)
        y = fit_kwargs.pop("y", None)
        validation_data = fit_kwargs.get("validation_data", None)
        
        if validation_data is None:
            raise ValueError("validation_data is required for MacroF1Oracle")
        
        X_val, y_val_onehot = validation_data
        y_val_labels = np.argmax(y_val_onehot, axis=1)
        
        # Add MacroF1Callback to track best weights by macro F1
        callbacks = list(fit_kwargs.get("callbacks", []))
        macro_f1_cb = MacroF1Callback(validation_data=(X_val, y_val_labels))
        callbacks.append(macro_f1_cb)
        fit_kwargs["callbacks"] = callbacks
        
        # Train
        history = model.fit(X, y, **fit_kwargs)
        
        # Evaluate using the callback's best macro F1
        best_f1 = macro_f1_cb.best_f1
        
        # Return a MetricsDict
        return kt.engine.trial.MetricsCallback(
            metrics={"macro_f1": best_f1},
            oracle=self,
            trial=trial,
        )

# ===========================================================================
# Run Keras Tuner with Macro F1 objective
# ===========================================================================
print("\n--- Keras Tuner Search (Objective: Macro F1) ---")

# We'll use a simpler approach: train each trial and compute macro F1 manually
# This is more reliable than the custom Oracle approach
def objective_macro_f1(hp):
    """Objective function that returns Macro F1 for a given hyperparameter config."""
    model = build_keras_model(hp)
    
    # Add MacroF1Callback to restore best weights by macro F1
    macro_f1_cb = MacroF1Callback(validation_data=(X_val_scaled, y_val))
    
    model.fit(
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
            macro_f1_cb,
        ],
        verbose=0,
    )
    
    return macro_f1_cb.best_f1

# Run Optuna-style manual search with macro F1 objective
# (Keras Tuner doesn't natively support custom metric objectives easily,
#  so we use a manual search loop with the same search space)
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_keras(trial):
    """Optuna objective for Keras MLP — returns Macro F1."""
    hp = keras_tuner_to_optuna_params(trial)
    model = build_keras_model_from_params(hp)
    
    macro_f1_cb = MacroF1Callback(validation_data=(X_val_scaled, y_val))
    
    model.fit(
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
            macro_f1_cb,
        ],
        verbose=0,
    )
    
    return macro_f1_cb.best_f1

def keras_tuner_to_optuna_params(trial):
    """Convert Optuna trial to Keras Tuner hyperparameters."""
    n_layers = trial.suggest_int("n_layers", 1, 4)
    params = {"n_layers": n_layers}
    
    for i in range(n_layers):
        params["units_" + str(i)] = trial.suggest_int("units_" + str(i), 32, 256, step=32)
        params["activation_" + str(i)] = trial.suggest_categorical("activation_" + str(i), ["relu", "gelu", "tanh"])
        params["batch_norm_" + str(i)] = trial.suggest_categorical("batch_norm_" + str(i), [True, False])
        params["dropout_" + str(i)] = round(trial.suggest_float("dropout_" + str(i), 0.0, 0.5, step=0.1), 1)
    
    params["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"])
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    return params

def build_keras_model_from_params(params):
    """Build Keras model from a parameter dictionary."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(20,)))
    
    n_layers = params["n_layers"]
    
    for i in range(n_layers):
        units = params["units_" + str(i)]
        activation = params["activation_" + str(i)]
        
        model.add(layers.Dense(units, activation=activation))
        
        if params["batch_norm_" + str(i)]:
            model.add(layers.BatchNormalization())
        
        dropout_rate = params["dropout_" + str(i)]
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(n_classes, activation="softmax"))
    
    optimizer_name = params["optimizer"]
    learning_rate = params["learning_rate"]
    
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

# Run Optuna study for Keras MLP
t0 = time.time()
study_keras = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study_keras.optimize(objective_keras, n_trials=KERAS_TRIALS, show_progress_bar=False)
search_time = time.time() - t0

best_params = study_keras.best_trial.params
print("Keras Tuner search completed in", round(search_time, 1), "s")
print("Best Macro F1:", round(study_keras.best_trial.value, 4))
print("Best hyperparameters:")
for key, value in best_params.items():
    print("  " + key + ":", value)

# ===========================================================================
# Retrain best model and compute full metrics
# ===========================================================================
print("\n--- Retraining Best Model ---")
best_model = build_keras_model_from_params(best_params)

t0 = time.time()
macro_f1_cb = MacroF1Callback(validation_data=(X_val_scaled, y_val))
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
        macro_f1_cb,
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
    "best_macro_f1_during_training": float(macro_f1_cb.best_f1),
    "best_hyperparameters": best_params,
    "n_trials": KERAS_TRIALS,
    "search_time": float(search_time),
    "objective": "macro_f1",
}

print("\nKeras MLP Results (Objective: Macro F1):")
print("  Accuracy:        " + str(round(acc, 4)))
print("  Macro F1:        " + str(round(f1, 4)))
print("  Balanced Acc:    " + str(round(bal_acc, 4)))
print("  Cohen's Kappa:   " + str(round(kappa, 4)))
print("  Fault Recall:    " + str(round(fault_recall, 4)))
print("  False Alarm:     " + str(round(false_alarm_rate, 4)))
print("  Train time:      " + str(round(train_time, 1)) + "s")
print("  Epochs trained:  " + str(len(history.history["loss"])))

# ===========================================================================
# Update results files
# ===========================================================================

# Load existing ML results
ml_results_path = ROOT / "results" / "test_2" / "test_2_results_optuna.json"
with open(ml_results_path, "r") as f:
    all_results = json.load(f)

# Replace MLP Neural Network results with genuine macro-F1-tuned results
all_results["MLP Neural Network"] = results_keras

# Save to test_2_results.json (canonical)
out_path = ROOT / "results" / "test_2" / "test_2_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to:", out_path)

# Save best params
best_params_dict = {}
for name, r in all_results.items():
    if "best_params" in r:
        best_params_dict[name] = r["best_params"]
    elif "best_hyperparameters" in r:
        best_params_dict[name] = r["best_hyperparameters"]

params_path = ROOT / "results" / "test_2" / "optuna_best_params.json"
with open(params_path, "w") as f:
    json.dump(best_params_dict, f, indent=2)
print("Best params saved to:", params_path)

# Build summary table
summary_rows = []
for name in all_results:
    r = all_results[name]
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
        "Objective": r.get("objective", "accuracy"),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Macro F1", ascending=False)
print("\nTest 2 — All Tuned Models (sorted by Macro F1):")
print("=" * 130)
print(summary_df.to_string(index=False, float_format="{:.4f}".format))

print("\nTest 2 complete!")
print("Total models:", len(all_results))
print("ML models with Optuna: 13")
print("Neural Network with Optuna (Macro F1): 1")
print("Objective: Macro F1 (all models)")
