# Comprehensive Evaluation — Findings

## Executive Summary

This document consolidates validated results from Tests 1–4 of the power-grid fault classification project. The analysis evaluates 14 machine learning models across four experimental configurations, ranging from single-split baseline (Test 1) to nested cross-validation with hyperparameter tuning (Test 4).

**Key findings:**
- 12 of 14 models achieve ≥99% fault detection recall in Test 4 (nested CV)
- Top tree-based models achieve Macro F1 > 0.99 with very low variance
- 4 models (Random Forest, Gradient Boosting, XGBoost, LightGBM) have zero false alarms across all 5 outer folds
- Near-perfect scores likely reflect simulation-driven class separability in the MATLAB-generated dataset
- **No final model is selected** — this phase produces a candidate shortlist for the Best Model Selection phase

## Test 1 → Test 4 Progression

| Phase | Description | Split Method | Tuning |
|-------|-------------|--------------|--------|
| Test 1 | Default models, single split | 70/15/15 stratified | None |
| Test 2 | Optuna tuning on holdout | 70/15/15 stratified | Optuna (5 trials) |
| Test 3 | 5-fold stratified CV | 5-fold CV on dev set | None |
| Test 4 | Nested CV + tuning | 5-fold outer × 3-fold inner | Optuna (10 trials) + Keras Tuner |

The progression from Test 1 to Test 4 adds methodological rigor:
- Test 1 provides a quick baseline with potential split-optimism
- Test 2 introduces hyperparameter tuning but retains single-split evaluation
- Test 3 removes split-optimism via cross-validation but uses default hyperparameters
- Test 4 combines both: nested CV for unbiased evaluation with per-fold tuning

## Cross-Test Comparison

### Macro F1

Top models maintain high Macro F1 across all tests:

| Model | Test 1 | Test 2 | Test 3 | Test 4 (nested CV) |
|-------|--------|--------|--------|---------------------|
| LightGBM | 0.9979 | 0.9862 | 0.9984 | 0.9987 ± 0.0009 |
| CatBoost | 0.9960 | 0.9970 | 0.9984 | 0.9983 ± 0.0009 |
| Extra Trees | 0.9910 | 0.9959 | 0.9941 | 0.9983 ± 0.0017 |
| XGBoost | 0.9979 | 0.9759 | 0.9981 | 0.9978 ± 0.0011 |

The consistency between Test 3 (default params) and Test 4 (tuned) for tree models suggests high scores are not artifacts of tuning.

### Fault Detection Recall

In Test 4, 12 of 14 models meet the ≥99% target:

| Model | Fault Recall | Meets 99%? |
|-------|-------------|------------|
| Extra Trees | 0.9998 | Yes |
| CatBoost | 0.9998 | Yes |
| HistGradientBoosting | 0.9998 | Yes |
| Bagging | 0.9996 | Yes |
| XGBoost | 0.9996 | Yes |
| LightGBM | 0.9994 | Yes |
| Random Forest | 0.9991 | Yes |
| Gradient Boosting | 0.9991 | Yes |
| Decision Tree | 0.9983 | Yes |
| MLP Neural Network | 0.9979 | Yes |
| AdaBoost | 0.9968 | Yes |
| KNN | 0.9910 | Yes |
| SVM | 0.9728 | No |
| Logistic Regression | 0.8880 | No |

### False Alarm Rate

Four models achieve zero false alarms across all 5 outer folds in Test 4:
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

## Test 4 Detailed Evaluation

All metrics reported as mean ± std across 5 outer folds (nested CV):

| Model | Accuracy | Macro F1 | Balanced Acc | Cohen Kappa |
|-------|----------|----------|--------------|-------------|
| LightGBM | 0.9988 ± 0.0009 | 0.9987 ± 0.0009 | 0.9986 ± 0.0010 | 0.9985 ± 0.0011 |
| CatBoost | 0.9985 ± 0.0009 | 0.9983 ± 0.0009 | 0.9984 ± 0.0009 | 0.9982 ± 0.0011 |
| Extra Trees | 0.9985 ± 0.0015 | 0.9983 ± 0.0017 | 0.9983 ± 0.0017 | 0.9982 ± 0.0018 |
| XGBoost | 0.9981 ± 0.0010 | 0.9978 ± 0.0011 | 0.9977 ± 0.0012 | 0.9976 ± 0.0012 |
| HistGradientBoosting | 0.9979 ± 0.0013 | 0.9976 ± 0.0015 | 0.9977 ± 0.0015 | 0.9974 ± 0.0017 |
| Bagging | 0.9976 ± 0.0019 | 0.9973 ± 0.0021 | 0.9973 ± 0.0021 | 0.9970 ± 0.0024 |
| Gradient Boosting | 0.9964 ± 0.0014 | 0.9960 ± 0.0015 | 0.9958 ± 0.0017 | 0.9956 ± 0.0018 |
| Random Forest | 0.9955 ± 0.0005 | 0.9950 ± 0.0009 | 0.9948 ± 0.0006 | 0.9945 ± 0.0006 |
| Decision Tree | 0.9936 ± 0.0019 | 0.9929 ± 0.0020 | 0.9926 ± 0.0020 | 0.9921 ± 0.0023 |
| MLP Neural Network | 0.9829 ± 0.0083 | 0.9807 ± 0.0094 | 0.9805 ± 0.0095 | 0.9790 ± 0.0102 |
| KNN | 0.9650 ± 0.0047 | 0.9613 ± 0.0053 | 0.9600 ± 0.0050 | 0.9568 ± 0.0057 |
| SVM | 0.9710 ± 0.0037 | 0.9714 ± 0.0034 | 0.9658 ± 0.0042 | 0.9641 ± 0.0046 |
| AdaBoost | 0.8484 ± 0.0159 | 0.8174 ± 0.0200 | 0.8244 ± 0.0193 | 0.8131 ± 0.0196 |
| Logistic Regression | 0.6652 ± 0.0166 | 0.6001 ± 0.0231 | 0.6039 ± 0.0198 | 0.5811 ± 0.0216 |

## Per-Class Performance

The per-class F1 heatmap (see `figures/test4_per_class_heatmap.png`) shows:
- Tree-based models achieve near-perfect F1 across all 6 fault classes
- "No Fault" class is consistently well-classified (high precision and recall)
- "LG A+B-G" and "LL-G BC" are the most challenging classes for weaker models (LR, SVM)
- The top 5 models show no meaningful per-class weakness

## Fault Detection Assessment

**12 of 14 models meet the ≥99% fault recall target in Test 4.**

The two models that do not meet the target:
- SVM: 97.28% (insufficient)
- Logistic Regression: 88.80% (insufficient)

Models meeting the target span multiple families: tree-based (Decision Tree, Random Forest, Extra Trees, Bagging, Gradient Boosting, XGBoost, HistGradientBoosting, LightGBM, CatBoost), boosting (AdaBoost), instance-based (KNN), and neural network (MLP).

## False Alarm Assessment

**Four models achieve zero false alarms** (false_alarm_rate = 0.0 across all 5 outer folds):
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Other models with very low false alarm rates:
- Decision Tree: 0.0005 ± 0.0011
- Extra Trees: 0.0005 ± 0.0011
- Bagging: 0.0005 ± 0.0011
- HistGradientBoosting: 0.0005 ± 0.0011
- CatBoost: 0.0005 ± 0.0011

## CV Stability

Standard deviations across 5 outer folds for Macro F1:

| Model | CV Std (Macro F1) | Assessment |
|-------|-------------------|------------|
| LightGBM | 0.0009 | Very stable |
| CatBoost | 0.0009 | Very stable |
| Random Forest | 0.0009 | Very stable |
| XGBoost | 0.0011 | Stable |
| HistGradientBoosting | 0.0015 | Stable |
| Extra Trees | 0.0017 | Stable |
| Gradient Boosting | 0.0015 | Stable |
| Decision Tree | 0.0020 | Acceptable |
| Bagging | 0.0021 | Acceptable |
| MLP Neural Network | 0.0094 | Higher variance |
| SVM | 0.0034 | Acceptable |
| KNN | 0.0053 | Acceptable |
| AdaBoost | 0.0200 | Higher variance |
| Logistic Regression | 0.0231 | Higher variance |

Tree-based models show the most stable performance (std < 0.002). Neural network and linear models show higher variance, consistent with their lower overall performance.

## Near-Perfect Score Investigation Summary

Source: `results/test_4/near_perfect_score_investigation.md`

### Evidence Collected

1. **No duplicates**: 0 exact or near-exact duplicate rows in the 7,861-row dataset.

2. **Target/input separation**: Fault code columns (G, C, B, A) are used only for target construction, not as model inputs. Feature engineering uses only the 6 raw measurements. Target leakage is not possible.

3. **Test 3 vs Test 4 consistency**: Most tree models show <1% difference between default (Test 3) and tuned (Test 4) configurations. This consistency suggests the high scores are not artifacts of overfitting or lucky splits.

4. **CV variance**: Standard deviations are all <0.002 for near-perfect models, indicating stable performance across folds. Low variance is inconsistent with overfitting to particular fold compositions.

5. **Zero false-alarm results**: Random Forest, Gradient Boosting, XGBoost, and LightGBM have `false_alarm_rate = 0.0` across ALL 5 outer folds in Test 4. Combined with fault detection recall >99.9%, this indicates near-complete class separability in the 20-dimensional feature space.

6. **Likely simulation-driven separability**: The dataset originates from MATLAB/Simulink power-system simulations. Simulation data often exhibits deterministic signal structure and clean class boundaries that real-world data would not have.

### What the Evidence Supports

- No data leakage, no duplicates
- Genuine class separability likely driven by simulation-clean signals
- Consistent across methods, folds, and tuning budgets

### What the Evidence Does NOT Prove

- Real-world applicability (simulation data may not reflect field conditions)
- Generalization guarantee (locked test set evaluation is pending)
- Model complexity justification (simpler models may suffice)

**We do not claim the scores are proven genuine — only what the evidence supports.**

## Computational / Inference Trade-offs

| Model | Train Time (s) | Inference Time (s) | Artifact Size |
|-------|---------------|-------------------|---------------|
| LightGBM | 0.2034 | 0.0073 | Small |
| CatBoost | 14.6946 | 0.0040 | Medium |
| Extra Trees | 0.8761 | 0.0194 | Large (ensemble) |
| XGBoost | 0.4018 | 0.0033 | Small |
| HistGradientBoosting | 0.4743 | 0.0145 | Medium |
| Random Forest | 0.5332 | 0.0047 | Large (ensemble) |
| Decision Tree | 0.0463 | 0.0008 | Very small |
| Gradient Boosting | 20.9169 | 0.0220 | Medium |
| MLP Neural Network | 5.5772 | 0.1042 | Small |

For deployment:
- **Fastest inference**: Decision Tree (0.8ms), XGBoost (3.3ms), Random Forest (4.7ms)
- **Fastest training**: Decision Tree (46ms), KNN (6ms — lazy learner)
- **Smallest artifacts**: Decision Tree, XGBoost, LightGBM

## Feature Importance — Deferred

Feature importance extraction is **not performed** in this phase because:

1. No validated feature-importance artifacts exist from Tests 1–4.
2. Test 4 produced fold-specific tuned hyperparameters (5 different configurations per model). There is no methodologically justified single parameter set to use.
3. Arbitrary selection of one fold's hyperparameters would introduce an unjustified choice.

Feature importance extraction will be performed **after Best Model Selection**, when one definitive model configuration exists.

## Candidate Shortlist

**This is NOT final model selection.** The actual decision belongs to the Best Model Selection phase.

### Top 5 Candidates (ranked by Macro F1 → False Alarm → CV Std)

| Rank | Model | Macro F1 | False Alarm | CV Std | Inference Time |
|------|-------|----------|-------------|--------|----------------|
| 1 | LightGBM | 0.9987 | 0.0000 | 0.0008 | 7.3ms |
| 2 | CatBoost | 0.9983 | 0.0005 | 0.0008 | 4.0ms |
| 3 | Extra Trees | 0.9983 | 0.0005 | 0.0015 | 19.4ms |
| 4 | XGBoost | 0.9978 | 0.0000 | 0.0010 | 3.3ms |
| 5 | HistGradientBoosting | 0.9976 | 0.0005 | 0.0013 | 14.5ms |

### Trade-off Discussion

- **LightGBM**: Highest Macro F1, zero false alarms, very low CV variance, fast training. Slightly slower inference than XGBoost.
- **CatBoost**: Near-identical performance to LightGBM, slightly faster inference but much slower training (14.7s vs 0.2s).
- **Extra Trees**: Matches CatBoost on F1, but larger artifact size and slower inference.
- **XGBoost**: Zero false alarms, fastest inference among top candidates, but slightly lower F1.
- **HistGradientBoosting**: Strong all-around, but slightly lower F1 and higher variance than top 3.

## Limitations

1. **ROC-AUC/PR-AUC not computed**: Probability outputs were not persisted during Tests 1–4. These metrics are reported as N/A.
2. **Feature importance deferred**: No single model configuration exists yet for meaningful importance extraction.
3. **Simulation data**: Results may not generalize to real-world fault conditions with measurement noise and sensor errors.
4. **10-trial Optuna budget**: The tuning budget (10 trials per outer fold) may be insufficient for some models. Some models converged to identical hyperparameters across all 5 folds, suggesting the budget was too low for meaningful per-fold adaptation.
5. **Locked test set not yet evaluated**: The 15% held-out test set has not been touched. Final unbiased performance is pending.

## Handoff to Best Model Selection

The Best Model Selection phase should:

1. Review this shortlist and the trade-off discussion
2. Consider deployment requirements (inference speed, artifact size, interpretability)
3. Select one model architecture/pipeline
4. Document the selection rationale
5. Proceed to Locked Final-Test Evaluation (one-time evaluation on the 15% held-out set)
6. Then Final Retraining on the complete development set for production

**The comprehensive evaluation does not declare a winner. The final choice belongs exclusively to the Best Model Selection phase.**
