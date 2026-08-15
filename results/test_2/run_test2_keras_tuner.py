import os
# Use TensorFlow backend for Keras (NumPy backend doesn't support fit())
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
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

# Prepare features for Keras
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

print("\nTraining features shape:", X_train_scaled.shape)
print("Validation features shape:", X_val_scaled.shape)
print("Training labels shape:", y_train_onehot.shape)
print("Validation labels shape:", y_val_onehot.shape)

# Custom callback to track macro F1
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

# Build model with Keras Tuner
def build_keras_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(20,)))
    
    # Tunable number of hidden layers
    n_layers = hp.Int("n_layers", min_value=1, max_value=4, default=2)
    
    for i in range(n_layers):
        # Units per layer
        units = hp.Int("units_" + str(i), min_value=32, max_value=256, step=32, default=64)
        
        # Activation
        activation = hp.Choice("activation_" + str(i), values=["relu", "gelu", "tanh"], default="relu")
        
        model.add(layers.Dense(units, activation=activation))
        
        # Batch normalization (tunable)
        if hp.Boolean("batch_norm_" + str(i), default=True):
            model.add(layers.BatchNormalization())
        
        # Dropout (tunable)
        dropout_rate = hp.Float("dropout_" + str(i), min_value=0.0, max_value=0.5, step=0.1, default=0.3)
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

print("\n--- Keras Tuner Search ---")
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
print("Keras Tuner search completed in", search_time, "s")
print("Best hyperparameters:")
for key, value in best_hp.values.items():
    print("  " + key + ":", value)

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
    "search_time": float(search_time),
}

print("\nKeras MLP Results:")
print("  Accuracy:        " + str(acc))
print("  Macro F1:        " + str(f1))
print("  Balanced Acc:    " + str(bal_acc))
print("  Cohen's Kappa:   " + str(kappa))
print("  Fault Recall:    " + str(fault_recall))
print("  False Alarm:     " + str(false_alarm_rate))
print("  Train time:      " + str(train_time) + "s")
print("  Epochs trained:  " + str(len(history.history["loss"])))

# Load existing ML results
ml_results_path = ROOT / "results" / "test_2" / "test_2_results_optuna.json"
with open(ml_results_path, "r") as f:
    all_results = json.load(f)

# Add Keras results
all_results["MLP Neural Network"] = results_keras

# Save final results
out_path = ROOT / "results" / "test_2" / "test_2_results_optuna.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to:", out_path)

# Save best params for all models
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
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Macro F1", ascending=False)
print("\nTest 2 — All Tuned Models (sorted by Macro F1):")
print("=" * 130)
print(summary_df.to_string(index=False, float_format="{:.4f}".format))

print("\nTest 2 complete!")
print("Total models:", len(all_results))
print("ML models with Optuna: 13")
print("Neural Network with Keras Tuner: 1")
