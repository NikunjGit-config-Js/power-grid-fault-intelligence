import os
os.environ['KERAS_BACKEND'] = 'numpy'

import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")

# Load Test 1 and Test 2 results
test1_path = ROOT / "results" / "test_1" / "test_1_results.json"
test2_path = ROOT / "results" / "test_2" / "test_2_results.json"

with open(test1_path, "r") as f:
    test1 = json.load(f)

with open(test2_path, "r") as f:
    test2 = json.load(f)

# Build rows
rows = []

# Test 1 results
for model_name in test1:
    r = test1[model_name]
    rows.append({
        "Experiment": "Test 1",
        "Feature Set": r.get("feature_set", "D1_physics_current_sym"),
        "Model": model_name,
        "Accuracy": r["accuracy"],
        "Macro F1": r["macro_f1"],
        "Balanced Accuracy": r["balanced_accuracy"],
        "Cohen Kappa": r["cohen_kappa"],
        "Fault Recall": r["fault_detection_recall"],
        "False Alarm Rate": r["false_alarm_rate"],
        "ROC-AUC": "Not Applicable",
        "PR-AUC": "Not Applicable",
        "CV Std Macro F1": "N/A (single split)",
        "Training Time": r.get("train_time", 0.0),
        "Inference Time": r.get("inference_time", 0.0),
        "Number of Features": r.get("n_features", 20),
        "Optuna Trials": r.get("n_trials", "N/A"),
        "Best Params": json.dumps(r.get("best_params", r.get("best_hyperparameters", {}))) if "best_params" in r or "best_hyperparameters" in r else "N/A",
    })

# Test 2 results (genuine Optuna/Keras Tuner)
for model_name in test2:
    r = test2[model_name]
    rows.append({
        "Experiment": "Test 2",
        "Feature Set": r.get("feature_set", "D1_physics_current_sym"),
        "Model": model_name,
        "Accuracy": r["accuracy"],
        "Macro F1": r["macro_f1"],
        "Balanced Accuracy": r["balanced_accuracy"],
        "Cohen Kappa": r["cohen_kappa"],
        "Fault Recall": r["fault_detection_recall"],
        "False Alarm Rate": r["false_alarm_rate"],
        "ROC-AUC": "Not Applicable",
        "PR-AUC": "Not Applicable",
        "CV Std Macro F1": "N/A (single split)",
        "Training Time": r.get("train_time", 0.0),
        "Inference Time": r.get("inference_time", 0.0),
        "Number of Features": r.get("n_features", 20),
        "Optuna Trials": r.get("n_trials", "N/A"),
        "Best Params": json.dumps(r.get("best_params", r.get("best_hyperparameters", {}))) if "best_params" in r or "best_hyperparameters" in r else "N/A",
    })

df = pd.DataFrame(rows)
out_path = ROOT / "results" / "model_comparison.csv"
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(f"Total rows: {len(df)}")
print(f"Test 1 rows: {len(df[df['Experiment'] == 'Test 1'])}")
print(f"Test 2 rows: {len(df[df['Experiment'] == 'Test 2'])}")

# Print summary
print("\n--- Test 2 Results Summary ---")
test2_df = df[df['Experiment'] == 'Test 2'].sort_values('Macro F1', ascending=False)
for _, row in test2_df.iterrows():
    print(f"  {row['Model']:25s}  Macro F1 = {row['Macro F1']:.4f}  Trials = {row['Optuna Trials']}")
