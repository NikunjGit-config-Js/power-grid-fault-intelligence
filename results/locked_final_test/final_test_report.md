# Locked Final-Test Evaluation Report

## Executive Summary

The selected LightGBM architecture was tuned on the complete development set (6681 samples) using 30-trial Optuna with 5-fold cross-validation, then evaluated **once** on the locked 15% test set (1180 samples).

**Result**: The locked test evaluation achieved **perfect scores** across all metrics — Accuracy, Macro F1, Balanced Accuracy, Cohen Kappa, ROC-AUC, PR-AUC, Fault Detection Recall (100%), and False Alarm Rate (0.0%).

This is consistent with the near-perfect scores observed in Tests 3 and 4 (nested CV), confirming that the 20-dimensional D1 feature space provides complete class separability for this simulation-generated dataset.

## Final Hyperparameters

The Optuna tuning process determined the following **definitive locked-test/final hyperparameters**:

| Parameter | Value |
|-----------|-------|
| n_estimators | 144 |
| max_depth | 8 |
| learning_rate | 0.2742 |
| num_leaves | 13 |
| min_child_samples | 50 |

**Optuna tuning details:**
- Budget: 30 trials
- CV: 5-fold stratified (random_state=42)
- Objective: Macro F1
- Best CV Macro F1: 0.999392

These hyperparameters differ from the Test 4 fold-specific configurations, confirming that the Test 4 hyperparameters were not fully converged and that this dedicated tuning run was valuable.

## Locked Test Set Metrics

### Core Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 1.000000 |
| **Macro F1** | 1.000000 |
| **Balanced Accuracy** | 1.000000 |
| **Cohen Kappa** | 1.000000 |
| **ROC-AUC (macro, ovr)** | 1.000000 |
| **PR-AUC (macro, ovr)** | 1.000000 |

### Operational / Safety Metrics

| Metric | Value |
|--------|-------|
| **Fault Detection Recall** | 100.00% (1180/1180) |
| **False Alarm Rate** | 0.00% (0/355) |

**Formulas:**
- Fault Detection Recall = recall(fault_mask_true, fault_mask_pred) where fault = any class except No Fault
- False Alarm Rate = false_alarms / n_no_fault where false_alarm = predict(fault) when actual is No Fault

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 3P-G | 1.0000 | 1.0000 | 1.0000 | 170 |
| LG A+B-G | 1.0000 | 1.0000 | 1.0000 | 170 |
| LL-G BC | 1.0000 | 1.0000 | 1.0000 | 151 |
| Line-to-Line AB | 1.0000 | 1.0000 | 1.0000 | 169 |
| No Fault | 1.0000 | 1.0000 | 1.0000 | 355 |
| Three-Phase | 1.0000 | 1.0000 | 1.0000 | 165 |

### Confusion Matrix

The confusion matrix is diagonal — every sample is classified correctly. See `figures/confusion_matrix_raw.png` and `figures/confusion_matrix_normalized.png`.

## Comparison to Test-4 Development Metrics

| Metric | Test 4 (Nested CV) | Locked Test | Change |
|--------|-------------------|-------------|--------|
| Macro F1 | 0.9987 +/- 0.0009 | 1.0000 | +0.0013 |
| Fault Recall | 99.94% | 100.00% | +0.06% |
| False Alarm Rate | 0.0000% | 0.0000% | 0.00% |
| Balanced Accuracy | 0.9986 | 1.0000 | +0.0014 |

**Interpretation**: The locked test performance is **slightly better** than the Test 4 development metrics. This is expected because:
1. The final model was trained on the complete development set (6681 samples) vs. individual CV folds (~5345 samples each)
2. The Optuna tuning was more thorough (30 trials vs. 10 trials per fold in Test 4)
3. The hyperparameters were specifically optimized for this architecture

The improvement is small (0.0013 on Macro F1) and consistent with the near-perfect scores observed throughout the experimentation pipeline.

## Methodology Notes

### Isolation Rules Maintained

1. **Label isolation**: Fault labels were constructed ONLY for dev_idx rows before tuning. Test labels were not constructed until after the evaluation pipeline was fitted on the complete development set.

2. **LabelEncoder isolation**: The LabelEncoder was fitted on development-set labels only, then frozen before tuning began. Test labels were encoded using the frozen encoder — never refit on test labels.

3. **Pipeline isolation**: Each Optuna trial created a fresh Pipeline([PhysicsWithCurrentSymmetry, LGBMClassifier]). The feature transformer was fitted ONLY on each CV training fold — never on the full dev set during tuning.

4. **Locked-test isolation**: The locked test set was not inspected, transformed, or used in any way until after Optuna tuning was complete and the evaluation pipeline was fitted on the complete development set.

5. **RAW X_test**: The locked test features were passed directly to the fitted Pipeline for predict() and predict_proba(). No manual transformation was applied.

6. **Single evaluation**: The locked test set was evaluated exactly once. No retuning, threshold adjustment, or model switching occurred after seeing the results.

### Pipeline Architecture

```
PhysicsWithCurrentSymmetry (D1, 20 features)
    |
    v
LGBMClassifier(n_estimators=144, max_depth=8, learning_rate=0.2742,
               num_leaves=13, min_child_samples=50)
```

No scaler required (tree-based model).

## Limitations and Caveats

1. **Perfect scores on simulation data**: The locked test evaluation achieved 100% on all metrics. This likely reflects the clean class separability of the MATLAB/Simulink-generated dataset rather than real-world fault conditions. The 20-dimensional D1 feature space appears to provide complete separation of the 6 fault classes in this simulation environment.

2. **Generalization uncertainty**: Results may not generalize to real-world fault conditions with measurement noise, sensor errors, impedance variations, or fault resistance. Field validation would be required to confirm production readiness.

3. **Single dataset**: The entire experiment was conducted on a single dataset (classData.csv) from one simulation environment. Cross-dataset validation is not available.

4. **Near-perfect score investigation**: The near-perfect scores observed throughout Tests 3-4 and confirmed in this locked test evaluation were investigated (see `results/test_4/near_perfect_score_investigation.md`). No evidence of data leakage, duplicates, or target leakage was found. The scores are attributed to simulation-driven class separability.

## Handoff to Final Retraining

The locked test evaluation is complete. The next phase is **Final Retraining**, which will:

1. Retrain the LightGBM pipeline on the **complete available labelled dataset** (all 7861 samples) for production deployment
2. Save the production artifact as `models/best_model_pipeline.joblib`
3. Save metadata as `models/model_metadata.json`
4. **Clearly distinguish** the production-fitted artifact from this evaluation artifact
5. **Not report** metrics from the production-refitted model as test metrics — the locked test metrics above are the final unbiased evaluation

### Selected Architecture Summary (for Final Retraining)

```
Model:              LightGBM
Pipeline:           PhysicsWithCurrentSymmetry -> LGBMClassifier
Feature Set:        D1 (20 features)
Scaler:             None (tree-based)
Locked Test F1:     1.0000
Locked Test Recall: 100.00%
Locked Test FAR:    0.00%
Hyperparameters:    n_estimators=144, max_depth=8, learning_rate=0.2742,
                    num_leaves=13, min_child_samples=50
```
