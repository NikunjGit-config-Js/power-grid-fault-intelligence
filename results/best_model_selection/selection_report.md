# Best Model Selection Report

## Selected Architecture

**LightGBM** (LightGBM Gradient Boosting Framework)

Pipeline:
```
PhysicsWithCurrentSymmetry (D1, 20 features) → LightGBM classifier
```

No scaler required (tree-based model). sklearn-compatible. Serializable with `joblib`.

## Selection Rationale

### Selection Hierarchy (PROJECT_SPEC lines 1847-1855)

1. **Fault Detection Recall >= 99%**: 12 of 14 models qualify. Two models eliminated:
   - Logistic Regression: 88.80% (below 99% threshold)
   - SVM: 97.28% (below 99% threshold)

2. **Highest Macro F1**: LightGBM ranks first at 0.9987 +/- 0.0009

3. **Lowest False-Alarm Rate**: LightGBM achieves 0.0000 (zero false alarms)

4. **Lower CV Variance**: LightGBM CV std = 0.0009 (tied with CatBoost for lowest)

5. **Lower Complexity / Faster Inference**: LightGBM inference = 7.3ms (acceptable)

### Full Ranking (Qualifying Models)

| Rank | Model | Macro F1 | False Alarm | CV Std | Inference |
|------|-------|----------|-------------|--------|-----------|
| 1 | **LightGBM** | **0.9987** | **0.0000** | **0.0009** | **7.3ms** |
| 2 | CatBoost | 0.9983 | 0.0005 | 0.0009 | 4.0ms |
| 3 | Extra Trees | 0.9983 | 0.0005 | 0.0017 | 19.4ms |
| 4 | XGBoost | 0.9978 | 0.0000 | 0.0011 | 3.3ms |
| 5 | HistGradientBoosting | 0.9976 | 0.0005 | 0.0015 | 14.5ms |
| 6 | Bagging | 0.9973 | 0.0005 | 0.0021 | 6.7ms |
| 7 | Gradient Boosting | 0.9960 | 0.0000 | 0.0015 | 22.0ms |
| 8 | Random Forest | 0.9950 | 0.0000 | 0.0009 | 4.7ms |
| 9 | Decision Tree | 0.9929 | 0.0005 | 0.0020 | 0.8ms |
| 10 | MLP Neural Network | 0.9807 | 0.0025 | 0.0094 | 104.2ms |
| 11 | KNN | 0.9613 | 0.0080 | 0.0053 | 6.2ms |
| 12 | AdaBoost | 0.8174 | 0.0055 | 0.0200 | 8.4ms |

### Eliminated Models

| Model | Reason | Fault Recall |
|-------|--------|-------------|
| Logistic Regression | Recall below 99% threshold | 88.80% |
| SVM | Recall below 99% threshold | 97.28% |

## Test-4 Hyperparameter Stability

LightGBM hyperparameters diverged across the 5 outer folds (3 unique configurations observed):

| Fold | learning_rate | max_depth | min_child_samples | n_estimators | num_leaves |
|------|--------------|-----------|-------------------|--------------|------------|
| 1 | 0.1903 | 2 | 37 | 89 | 64 |
| 2 | 0.0586 | 4 | 13 | 216 | 59 |
| 3 | 0.1206 | 10 | 12 | 144 | 64 |
| 4 | 0.1206 | 10 | 12 | 144 | 64 |
| 5 | 0.1903 | 2 | 37 | 89 | 64 |

**Observation**: Folds 1 and 5 converged to identical configs; folds 3 and 4 converged to identical configs; fold 2 found a distinct configuration. All three configurations achieved high performance, suggesting the model is robust to hyperparameter choices within the explored space.

## Final Hyperparameters -- Deferred to Locked Final-Test Evaluation

The 5 outer-fold configurations are **not identical**. Therefore:

- No fold-specific configuration is adopted as the definitive production hyperparameter set.
- The mode is not used because continuous parameters (`learning_rate`) make that choice ambiguous.
- The fold with the highest outer-fold Macro F1 is not selected (that would be data-driven selection on test data).

**The definitive hyperparameters will be determined in the Locked Final-Test Evaluation phase by tuning the selected LightGBM architecture using ONLY the complete development set and cross-validation, before evaluating the locked test set exactly once.**

## Fold-Level Stability

### Macro F1 Across 5 Outer Folds

| Fold | Macro F1 |
|------|----------|
| 1 | 0.999134 |
| 2 | 0.998478 |
| 3 | 0.997407 |
| 4 | 0.998704 |
| 5 | 1.000000 |

- **Minimum**: 0.997407
- **Maximum**: 1.000000
- **Mean**: 0.998745
- **Std Dev**: 0.000947
- **Range**: 0.002593

### Outlier Detection

No fold is more than 3 standard deviations from the mean. No outlier folds detected.

### Leave-One-Out Sensitivity

Removing any single fold and recomputing the mean to check if the aggregate ranking changes:

| Fold Removed | LightGBM LOO Mean | Next-Best Model | Next-Best LOO Mean | Ranking Changes? |
|-------------|-------------------|-----------------|--------------------|--------------------|
| 1 | 0.998647 | CatBoost | ~0.9983 | No |
| 2 | 0.998811 | CatBoost | ~0.9983 | No |
| 3 | 0.999079 | CatBoost | ~0.9983 | No |
| 4 | 0.998755 | Extra Trees | 0.998950 | **Yes** |
| 5 | 0.998431 | CatBoost | ~0.9983 | No |

**Interpretation**: LightGBM remains the top-ranked model in 4 of 5 leave-one-out scenarios. When fold 4 is removed, Extra Trees would marginally overtake LightGBM (0.998950 vs 0.998755). This indicates the aggregate ranking is **not disproportionately driven by any single fold**, though there is mild sensitivity to fold composition. The margin between LightGBM and Extra Trees is small (0.0004 on Macro F1), and LightGBM's zero false-alarm rate provides an additional operational advantage.

## Deployment Characteristics

| Property | Value |
|----------|-------|
| **Serialization** | Compatible with `joblib` (sklearn-compatible API) |
| **Scaler Required** | No (tree-based model) |
| **Inference Time** | 7.3ms per sample (Test 4 measured) |
| **Feature Transformer** | `PhysicsWithCurrentSymmetry` (D1, 20 features) |
| **Raw Inputs** | 6 (Ia, Ib, Ic, Va, Vb, Vc) |
| **Output Classes** | 6 (3P-G, LG A+B-G, LL-G BC, Line-to-Line AB, No Fault, Three-Phase) |
| **Artifact Size** | Deferred to Final Retraining |

## Limitations

1. **Near-perfect scores**: The top models achieve Macro F1 > 0.997, likely reflecting simulation-driven class separability in the MATLAB-generated dataset. Real-world performance may differ.
2. **Hyperparameter instability**: LightGBM's 5 outer-fold configurations diverged (3 unique configs). This suggests the Optuna search space or budget (10 trials) may be insufficient for fully converged tuning. The definitive hyperparameters will be determined during the Locked Final-Test Evaluation phase.
3. **Leave-one-out sensitivity**: Removing fold 4 causes Extra Trees to overtake LightGBM, indicating the ranking margin is narrow. The zero false-alarm rate of LightGBM provides a secondary discriminator.
4. **10-trial Optuna budget**: Per-fold tuning used only 10 Optuna trials (vs PROJECT_SPEC recommendation of 50). This may explain the hyperparameter divergence and could leave performance on the table.
5. **Single dataset**: Results are validated on one dataset (classData.csv) from one simulation environment. Generalization to other fault scenarios, noise conditions, or real-world data is not verified.

## Handoff to Locked Final-Test Evaluation

The next phase must:

1. **Tune LightGBM** using the complete development set (85% of data = 6681 samples) with cross-validation
2. **Determine definitive hyperparameters** — the 5-fold Test-4 configs are diverged; the Locked Final-Test phase will find the production configuration
3. **Evaluate ONCE** on the locked 15% test set (1180 samples)
4. **Report all required metrics**: Accuracy, Macro F1, Balanced Accuracy, Cohen Kappa, per-class Precision/Recall, Fault Detection Recall, False Alarm Rate, confusion matrices
5. **Do not** use the Test-4 metrics as the final test metrics — the locked test set provides the unbiased evaluation
6. **Document** the final hyperparameters used and the evaluation results

### Selected Architecture Summary

```
Model:              LightGBM
Pipeline:           PhysicsWithCurrentSymmetry → LightGBM
Feature Set:        D1 (20 features)
Scaler:             None (tree-based)
Test-4 Macro F1:    0.9987 +/- 0.0009
Test-4 Fault Recall: 99.94%
Test-4 False Alarm:  0.0000
Inference Time:     7.3ms
Final Hyperparams:  Deferred to Locked Final-Test Evaluation
```

### Phase Constraints

- **Locked test set**: Untouched during this phase. No data from the locked 15% test set was loaded, referenced, or used in any analysis or decision.
- **No model training**: No models were instantiated, fitted, or evaluated. All metrics are sourced from Test 4 results.
- **No artifact modification**: No previous experiment files were modified.
