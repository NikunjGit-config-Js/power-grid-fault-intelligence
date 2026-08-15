# ROLE

Act as a **Senior Machine Learning Engineer, Applied Data Scientist, Power-Systems ML Researcher, MLOps Engineer, and Streamlit Deployment Engineer with 10+ years of experience**.

You are redesigning and improving this existing project:

**GitHub Repository:**
https://github.com/Marvellous-Udoye/power-grid-fault-detection-and-classification

This is **not** a request to create an unrelated project from scratch.

Your job is to:

1. inspect and understand the existing repository,
2. reproduce the existing baseline,
3. identify methodological problems,
4. redesign the complete data-science pipeline,
5. conduct a rigorous feature-engineering experiment,
6. evaluate multiple ML and neural-network classifiers through four experiment protocols,
7. select the best model scientifically,
8. save a production-ready inference pipeline,
9. build a minimal Streamlit inference demo,
10. prepare the project for deployment.

The final repository must be **reproducible, scientifically defensible, interview-ready, and deployable**.

---

# CORE PRINCIPLE

The redesigned project should follow this pipeline:

```text
Raw Power-Grid Dataset
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
EDA
        ↓
Univariate Analysis
        ↓
Outlier Analysis
        ↓
Bivariate / Multivariate Analysis
        ↓
Correlation Analysis
        ↓
Feature Engineering Experiment
        ↓
Physics-Informed Feature Selection
        ↓
ML / Neural Network Experiments
        ↓
Test 1
        ↓
Test 2
        ↓
Test 3
        ↓
Test 4
        ↓
Comprehensive Evaluation
        ↓
Best Model Selection
        ↓
Final Retraining
        ↓
Save Complete Inference Pipeline
        ↓
Minimal Streamlit Demo
        ↓
Heroku Deployment Preparation
```

Do not skip stages.

---

# PHASE 0 — AUDIT THE ORIGINAL PROJECT FIRST

Before modifying anything, inspect the complete repository.

Read:

* README
* notebook
* datasets
* trained `.pkl` artifacts
* current requirements
* feature-engineering code
* train/test split
* preprocessing
* target construction
* evaluation logic
* class mapping

Determine:

* exact dataset dimensions,
* raw feature names,
* target columns,
* observed fault classes,
* class frequencies,
* mapping from `G, C, B, A` to fault labels,
* current preprocessing,
* current polynomial expansion,
* current model configurations,
* any leakage,
* any mathematically meaningless features,
* any inconsistent feature definitions,
* any preprocessing performed before splitting,
* any mismatch between notebook and saved artifacts.

Do not trust comments or README claims without verifying them against code.

---

# PHASE 0.1 — REPRODUCE THE EXISTING BASELINE

Before redesigning anything, reproduce the shipped baseline.

The existing project uses the six raw measurements:

```text
Ia
Ib
Ic
Va
Vb
Vc
```

and expands them using degree-2 polynomial features.

The expected expansion is:

```text
6 original features
+ 6 squared features
+ 15 pairwise interaction features
= 27 total features
```

Verify this directly.

The legacy pipeline should be reproduced using the same original split:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The previously observed reference results are approximately:

```text
Decision Tree Accuracy ≈ 89.89%
Random Forest Accuracy ≈ 88.30%
```

Treat these only as **baseline reproduction checks**.

Do not use them as the final redesigned-project metrics.

If your reproduced values differ materially, investigate why before proceeding.

---

# PHASE 0.2 — AUDIT EXISTING FEATURE ENGINEERING

Explicitly verify suspicious existing features.

For example, investigate expressions similar to:

```python
THD_Ia = sqrt(Ia^2 - Ia**2) / Ia
```

If the polynomial column `Ia^2` is mathematically identical to `Ia**2`, then this numerator is zero and the resulting feature carries no useful information.

Perform the same investigation for:

```text
THD_Ia
THD_Ib
THD_Ic
```

and any so-called phase-angle features.

For example, determine whether something named:

```text
PhaseAngleDiffI
```

is actually calculating an angle or merely multiplying:

```python
Ia * Ib * Ic
```

Do not retain a feature merely because its variable name sounds physically meaningful.

Every domain-derived feature must have a valid mathematical interpretation.

---

# PHASE 1 — COMPLETE DATA ANALYSIS

Create a proper EDA section before modelling.

The analysis must be systematic.

---

## 1.1 Dataset Overview

Report:

* shape,
* columns,
* dtypes,
* unique values,
* missing values,
* duplicate rows,
* target distribution,
* observed fault codes,
* number of classes,
* class imbalance.

Explain the physical meaning of:

```text
Ia, Ib, Ic
Va, Vb, Vc
G, C, B, A
```

wherever it can be verified.

---

# 1.2 Data Quality Analysis

Check:

* missing values,
* duplicate samples,
* infinite values,
* impossible numeric values,
* near-zero variance columns,
* constant columns,
* suspiciously duplicated signals,
* invalid labels.

Do not silently delete rows.

Document every cleaning decision.

---

# 1.3 Class Distribution

Plot:

```text
Fault Class vs Sample Count
```

and percentages.

Also compare:

```text
Fault
vs
No Fault
```

as a binary summary.

Clearly identify which class corresponds to normal operation.

---

# 1.4 UNIVARIATE ANALYSIS

For every raw electrical measurement:

```text
Ia
Ib
Ic
Va
Vb
Vc
```

generate useful univariate plots.

Include where appropriate:

* histogram,
* KDE/density plot,
* box plot,
* descriptive statistics.

Report:

```text
mean
median
standard deviation
minimum
maximum
Q1
Q3
IQR
skewness
kurtosis
```

Avoid creating plots simply for decoration.

Explain what each important distribution suggests.

---

# 1.5 OUTLIER ANALYSIS

Perform outlier analysis using:

* IQR,
* box plots,
* optionally robust z-scores.

However:

## DO NOT blindly remove statistical outliers.

In an electrical-fault dataset, extreme current or voltage measurements may represent the actual fault signal.

Therefore classify suspicious points conceptually as:

```text
Potential measurement corruption
vs
Physically meaningful fault extremes
```

Remove only values that can reasonably be justified as erroneous.

Never remove rows merely to improve model scores.

Any learned outlier treatment must be fitted on training data only.

---

# 1.6 BIVARIATE ANALYSIS

Study relationships between:

```text
Ia ↔ Va
Ib ↔ Vb
Ic ↔ Vc
```

and cross-phase relationships.

Use suitable plots such as:

* scatter plots,
* class-colored scatter plots,
* grouped box plots,
* violin plots where useful.

Compare distributions across fault classes.

Examples:

```text
Ia by Fault Type
Ib by Fault Type
Ic by Fault Type

Va by Fault Type
Vb by Fault Type
Vc by Fault Type
```

Investigate whether fault classes create recognizable phase patterns.

---

# 1.7 MULTIVARIATE / CORRELATION ANALYSIS

Generate:

## Raw-feature correlation heatmap

using:

```text
Ia, Ib, Ic, Va, Vb, Vc
```

Also generate correlation analysis for engineered numerical features later.

Do not interpret class labels as continuous quantities in a Pearson correlation heatmap.

Discuss:

* strongly correlated measurements,
* redundant features,
* inverse relationships,
* possible multicollinearity.

---

# PHASE 2 — FEATURE ENGINEERING EXPERIMENT

This is one of the most important additions to the project.

The experiment is:

# Physics-Informed Feature Engineering vs Brute-Force Polynomial Expansion

Do not assume beforehand that physics-informed features will win.

The experiment must test the hypothesis objectively.

---

# FEATURE SET A — ORIGINAL BRUTE-FORCE POLYNOMIAL BASELINE

Starting with:

```text
Ia
Ib
Ic
Va
Vb
Vc
```

apply:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

Expected:

```text
27 features
```

consisting of:

* six raw features,
* six squares,
* fifteen pairwise interactions.

Verify generated names programmatically using:

```python
get_feature_names_out()
```

Do not hardcode incorrect feature names.

---

# FEATURE SET B — PHYSICS-INFORMED FEATURES

Create a deterministic custom transformer.

Start with the six raw measurements:

```text
Ia
Ib
Ic
Va
Vb
Vc
```

---

## Clarke Transform — Current

Calculate:

```text
I0 = (Ia + Ib + Ic) / 3

Iα = (2Ia - Ib - Ic) / 3

Iβ = (Ib - Ic) / √3

Imag = √(Iα² + Iβ²)
```

Interpretation:

```text
I0    → zero-sequence / ground-related behaviour
Iαβ   → transformed orthogonal current representation
Imag  → magnitude of αβ current component
```

---

# Clarke Transform — Voltage

Likewise:

```text
V0 = (Va + Vb + Vc) / 3

Vα = (2Va - Vb - Vc) / 3

Vβ = (Vb - Vc) / √3

Vmag = √(Vα² + Vβ²)
```

---

# Guarded Per-Phase Impedance

Create:

```text
Za = Va / Ia
Zb = Vb / Ib
Zc = Vc / Ic
```

but NEVER perform an unsafe raw division.

Use a numerically guarded implementation.

For example:

```python
safe_divide(V, I, epsilon)
```

that preserves sign and prevents:

```text
division by zero
inf
-inf
NaN
```

Do not choose epsilon based on test-set information.

If a bounded impedance representation such as:

```python
arctan(Z)
```

is evaluated, treat it as an explicit experiment rather than silently changing the representation.

---

# IMPORTANT FEATURE COUNT NOTE

Do not artificially force the feature count to equal exactly 15.

The listed core set actually produces:

```text
6 raw
+ 4 current Clarke features
+ 4 voltage Clarke features
+ 3 impedance features
= 17 core features
```

Therefore use the **defined physics features as authoritative**, not the approximate feature-count description.

---

# OPTIONAL PHASE-INVARIANT FEATURES

Evaluate separately:

```text
sorted(|Ia|, |Ib|, |Ic|)
```

giving:

```text
I_max
I_mid
I_min
```

Potentially also evaluate:

```text
V_max
V_mid
V_min
```

if scientifically useful.

The purpose is to represent:

```text
one phase abnormal
two phases abnormal
overall asymmetry
```

without making the representation excessively dependent on a specific phase name.

Do not automatically retain these.

Evaluate whether they improve generalisation.

---

# FEATURE ENGINEERING COMPARISON

Compare at least:

```text
A. Raw six features
B. Degree-2 polynomial features
C. Physics-informed core features
D. Physics-informed + optional symmetry-invariant features
```

Use identical splits/folds across feature sets.

Do not compare models using different data splits.

---

# FEATURE ENGINEERING EXPERIMENT CRITERIA

Compare feature sets on:

1. macro F1,
2. balanced accuracy,
3. fault-detection recall,
4. false-alarm rate,
5. feature count,
6. cross-validation variance,
7. interpretability,
8. computational cost.

The preferred feature set should not be chosen merely because it has the highest score by 0.001.

Consider:

```text
performance
+
stability
+
compactness
+
physical interpretability
```

Report honestly if polynomial expansion still performs better.

Do not cherry-pick the result to prove the hypothesis.

---

# PHASE 3 — DATA SPLIT STRATEGY

Create ONE locked test set at the beginning of the redesigned experiment.

Use:

```text
70% Training
15% Validation
15% Final Test
```

with stratification and:

```python
random_state=42
```

The final 15% test set must remain untouched during:

* feature selection,
* Optuna,
* Keras Tuner,
* architecture decisions,
* model selection.

Conceptually:

```text
FULL DATA
   ↓
────────────────────────────
85% DEVELOPMENT   15% TEST
────────────────────────────
   ↓
70% TRAIN
15% VALIDATION
```

Tests 1 and 2 use the development split as train/validation.

Tests 3 and 4 perform cross-validation within the development data.

The locked test set is used only after final model selection.

This prevents the final test set from becoming another validation set.

---

# PHASE 4 — PREPROCESSING RULES

All preprocessing must avoid leakage.

Anything that learns from data must be fitted ONLY on training data.

Examples:

```text
scaler (RobustScaler for scale-sensitive models in redesigned experiments; MinMaxScaler only for legacy reproduction)
imputer
feature selector
resampler
polynomial transformer when relevant
learned clipping thresholds
```

must live inside a proper:

```python
Pipeline
```

or:

```python
imblearn.pipeline.Pipeline
```

when resampling is used.

---

# SCALING POLICY

## Legacy Baseline Reproduction

For the legacy baseline reproduction ONLY, preserve the historical MinMaxScaler behavior exactly:

```python
# LEGACY ONLY — reproduce the original pipeline for comparison
scaler = MinMaxScaler()  # fitted on full dataset (historical leakage, preserved deliberately)
```

This is necessary because the shipped `.pkl` artifacts were trained under this transform. Changing the scaler would corrupt predictions on existing artifacts.

## Redesigned Experiments — RobustScaler for Scale-Sensitive Models

For the redesigned experiments (Tests 1–4), **RobustScaler** is the required scaler for scale-sensitive models. Tree-based models generally do not require scaling and may omit RobustScaler.

### Tree-Based Models — Scaling Optional

The following models generally do **not** require scaling:

```text
Decision Tree
Random Forest
Extra Trees
Bagging Classifier (tree base estimator)
Gradient Boosting
HistGradient Boosting
XGBoost
LightGBM
CatBoost
```

These models may omit RobustScaler unless there is a specific experimentally justified reason.

### Scale-Sensitive Models — RobustScaler Mandatory

The following models **require** scaling and must use RobustScaler:

```text
Logistic Regression
K-Nearest Neighbours
Support Vector Machine
Neural Network (MLP)
```

Any other distance/gradient-sensitive model added to the suite must also use RobustScaler.

### RobustScaler Rules

1. When used, RobustScaler must be fitted **ONLY** on the training data / training fold. Never on validation, test, or the full dataset.
2. **Test 1 and Test 2**: Fit RobustScaler on the training split only. Transform validation data using that fitted scaler. Include it in the pipeline for scale-sensitive models only.
3. **Test 3**: RobustScaler must be inside the sklearn `Pipeline` and fitted separately inside each of the 5 CV training folds. Do not reuse one scaler across folds. Apply to scale-sensitive models only.
4. **Test 4**: RobustScaler must remain inside the pipeline during both inner tuning folds and outer evaluation folds. The locked final test set must never influence RobustScaler fitting. Apply to scale-sensitive models only.
5. The locked final test set must never influence RobustScaler fitting.
6. Do not fit RobustScaler on the complete dataset before splitting.
7. Do not reuse one scaler across CV folds.

### PolynomialFeatures Clarification

`PolynomialFeatures(degree=2, include_bias=False)` is a **deterministic** transform with no learned parameters. Fitting it before splitting does not create information leakage. However, it should still live inside the redesigned `Pipeline` for consistency and modularity.

Do not conflate PolynomialFeatures with MinMaxScaler leakage. The genuine leakage in the legacy pipeline comes from fitting MinMaxScaler on the full dataset, not from PolynomialFeatures.

### Most Importantly

# FIT THE SCALER INSIDE THE TRAINING FOLD.

Never:

```text
fit scaler → split data
```

Instead:

```text
split data
→ fit pipeline on training fold
→ transform validation/test fold
```

---

# SUSPICIOUS NEAR-PERFECT SCORES

The legacy notebook reports near-perfect or perfect test accuracy for several models (e.g., Decision Tree 99.49%, Random Forest 99.83%, LightGBM 100%, CatBoost 100% on various feature sets).

Do NOT attribute these scores to scaler leakage alone. While MinMaxScaler fitted on the full dataset is genuine leakage, the magnitude of these scores (especially 100%) suggests additional factors may be at play.

Before drawing conclusions, rigorously investigate:

1. **Duplicate or near-duplicate rows** — Check for identical or near-identical samples across train/test splits.
2. **Simulation structure** — The dataset is generated by MATLAB simulation. Deterministic simulation parameters may create artificially separable classes.
3. **Random-split optimism** — A single random split (random_state=42) may by chance produce an easy test set. Verify with multiple splits or cross-validation.
4. **Repeated scenarios** — Check if the same fault conditions appear with slight numerical variations, creating clusters that tree models can memorize.
5. **Target leakage** — Verify that no feature directly encodes the target label (e.g., through G/C/B/A indicator columns leaking into measurements).
6. **Deterministic class structure** — The 6-class structure with 7,861 samples from MATLAB simulation may have inherently separable classes, especially for fault types with distinct current/voltage signatures.

Document findings. Do not silently dismiss suspicious results.

---

# CLASS IMBALANCE

Do not automatically use SMOTE.

First quantify class imbalance.

Because macro F1 is the main metric, class imbalance is already explicitly reflected in evaluation.

If resampling or class weighting is explored:

* perform it only on training folds,
* put it inside the pipeline,
* never resample validation/test data,
* compare it objectively.

---

# PHASE 5 — MACHINE LEARNING ENGINE

Run four major experiment protocols.

Every experiment must be reproducible.

Use fixed random seeds wherever applicable.

---

# MODEL SUITE

At minimum include:

## Classical Baselines

```text
Logistic Regression
K-Nearest Neighbours
Support Vector Machine
Decision Tree
```

## Bagging / Randomised Trees

```text
Bagging Classifier
Random Forest
Extra Trees
```

## Boosting

Mandatory:

```text
AdaBoost
Gradient Boosting
XGBoost
```

Also include when technically reasonable:

```text
HistGradientBoosting
LightGBM
CatBoost
```

If a dependency cannot be supported, state that clearly.

Do not silently skip mandatory models.

---

# NEURAL NETWORK

Use a feed-forward neural network suitable for tabular classification.

The output layer must correspond to the verified number of classes.

For example:

```python
Dense(n_classes, activation="softmax")
```

Use an appropriate multiclass cross-entropy loss.

---

# TEST 1 — DEFAULT / BASELINE MODELS

## Goal

Determine how every algorithm performs without serious hyperparameter optimisation.

Use the same:

```text
70% train
15% validation
15% locked test
```

structure.

The test portion remains locked.

Model development uses training + validation only.

---

## ML Algorithms

Use sensible library defaults or minimally modified defaults required for:

```text
random_state
multiclass compatibility
reproducibility
```

Do not tune hyperparameters.

---

## Neural Network Baseline

Create a simple fixed MLP.

For example:

```text
Input
↓
Dense
↓
ReLU
↓
Dense
↓
ReLU
↓
Output Softmax
```

The exact width may be chosen based on input dimensionality, but keep it simple.

For Test 1:

* no Keras Tuner,
* no optimizer search,
* no elaborate BatchNorm,
* no custom weight-init experiment,
* no architecture search,
* no aggressive regularisation search.

Use:

```text
Adam
default/sensible learning rate
default Glorot initialization
fixed architecture
fixed batch size
```

A predefined EarlyStopping callback may be used only to prevent obvious overfitting.

It is not considered hyperparameter optimisation.

---

# TEST 2 — HOLDOUT HYPERPARAMETER OPTIMISATION

Use:

```text
70% Train
15% Validation
15% Final Test
```

The final test remains locked.

---

# ML MODELS — OPTUNA

Use Optuna for every appropriate ML algorithm.

Primary objective:

# MACRO F1

NOT accuracy.

Example conceptual objective:

```python
trial
→ suggest hyperparameters
→ fit train
→ predict validation
→ macro_f1_score
→ maximise
```

Do not tune against the final test set.

---

# OPTUNA SEARCH SPACES

Construct appropriate algorithm-specific spaces.

Examples include:

## Decision Tree

```text
max_depth
min_samples_split
min_samples_leaf
max_features
criterion
class_weight
```

## Random Forest

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
bootstrap
class_weight
```

## Extra Trees

Tune comparable tree-ensemble parameters.

## Bagging

```text
n_estimators
max_samples
max_features
bootstrap
bootstrap_features
```

## AdaBoost

```text
n_estimators
learning_rate
base estimator depth where supported
```

## Gradient Boosting

```text
n_estimators
learning_rate
max_depth
subsample
min_samples_split
min_samples_leaf
```

## XGBoost

Include sensible ranges for:

```text
n_estimators
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
gamma
reg_alpha
reg_lambda
```

## LightGBM / CatBoost

Use appropriate equivalent spaces.

Do not create meaningless or incompatible parameter combinations.

---

# OPTUNA TRIAL BUDGET

Use a configurable value.

For example:

```python
N_TRIALS = 50
```

Use fewer only if compute constraints make the experiment impractical.

Document the value.

Set deterministic seeds where supported.

---

# NEURAL NETWORK — KERAS TUNER

For Test 2, use Keras Tuner.

Optimise macro F1 on the validation set.

Search important architecture/training choices such as:

```text
number of hidden layers
units per layer
activation
dropout
batch normalization
kernel initializer
L2 regularisation
optimizer
learning rate
batch size
```

Reasonable search ranges might include:

```text
Hidden layers: 1–4
Units: 32–256
Dropout: 0.0–0.5
Learning rate: 1e-5–1e-2
Batch size: 32 / 64 / 128
```

Possible activations:

```text
ReLU
GELU
tanh
```

Possible optimizers:

```text
Adam
AdamW
RMSprop
```

where supported.

Possible initializers:

```text
Glorot Uniform
He Normal
He Uniform
```

Use BatchNorm as a tunable boolean option rather than blindly inserting it everywhere.

---

# EARLY STOPPING

Use something similar to:

```python
EarlyStopping(
    monitor=...,
    patience=10,
    restore_best_weights=True
)
```

The monitored quantity should align as closely as possible with model selection.

If macro F1 is implemented as a Keras metric, monitor validation macro F1.

Otherwise use a callback that computes validation macro F1 reliably.

Do not optimize only `val_accuracy`.

---

# TEST 3 — 5-FOLD STRATIFIED CROSS-VALIDATION

Use:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Apply to the development dataset.

Use the same folds for every model.

This enables paired comparisons.

---

# TEST 3 ML

Run all ML algorithms with their baseline/default configurations.

For every fold:

```text
fit preprocessing on training fold
fit model
evaluate validation fold
store metrics
```

Never fit preprocessing before CV.

---

# TEST 3 NEURAL NETWORK

Use the fixed baseline neural-network architecture from Test 1.

For each outer fold:

```text
reinitialize model
train only on fold training data
evaluate fold validation data
```

Never continue training the same NN across folds.

Reinitialize weights for every fold.

Store fold-level metrics.

---

# TEST 4 — NESTED CROSS-VALIDATION + HYPERPARAMETER OPTIMISATION

This is the most rigorous experiment.

Use:

```text
Outer CV → unbiased evaluation
Inner CV → hyperparameter search
```

---

# OUTER CV

Use:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

---

# INNER CV

Use another stratified CV exclusively inside the outer training fold.

For example:

```python
StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
```

---

# ML + OPTUNA NESTED CV

For every outer fold:

```text
Outer Train
      ↓
Inner Stratified CV
      ↓
Optuna
      ↓
Best Inner Hyperparameters
      ↓
Retrain on complete Outer Train
      ↓
Evaluate Outer Validation
```

The outer validation fold must never participate in Optuna.

Store all outer-fold metrics.

Final reported Test-4 values must be:

```text
mean ± standard deviation
```

across the five outer folds.

---

# NEURAL NETWORK + KERAS TUNER NESTED CV

Follow the same principle.

For every outer fold:

```text
Outer Training Data
        ↓
Inner folds
        ↓
Keras-Tuner hyperparameter evaluation
        ↓
Select architecture
        ↓
Reinitialize selected architecture
        ↓
Train on complete outer-training partition
        ↓
Evaluate outer-validation partition
```

Do not tune directly against the outer fold.

If standard Keras Tuner cannot perform inner-fold averaging directly, implement an appropriate custom tuning loop or custom `run_trial`.

Scientific correctness is more important than convenience.

---

# PHASE 6 — EVALUATION METRICS

Accuracy must NOT be the main metric.

The headline/model-selection metric is:

# Macro F1

because every class should receive equal importance.

---

# REQUIRED MULTICLASS METRICS

For every model report:

```text
Accuracy
Macro F1
Balanced Accuracy
Cohen's Kappa
```

Also report:

# PER-CLASS PRECISION

and

# PER-CLASS RECALL

for every observed class.

Do not collapse these into a single weighted metric.

---

# OPERATIONAL INTERPRETATION

Fault-class recall is particularly important because:

```text
False Negative
=
Fault exists but model says safe
```

No-Fault precision/false alarms are operationally important because:

```text
False Positive
=
Healthy condition predicted as fault
```

Therefore calculate deployment-facing binary KPIs as well.

Collapse predictions into:

```text
FAULT
vs
NO FAULT
```

and report:

# Fault Detection Recall

Target aspiration:

```text
≥ 99%
```

Also report:

# False Alarm Rate

defined using No-Fault examples.

Clearly state the formula in the notebook.

---

# CONFUSION MATRIX

For every model produce:

1. raw-count confusion matrix,
2. row-normalised confusion matrix.

Pay particular attention to confusion between:

```text
Three-Phase Fault
vs
Three-Phase-with-Ground Fault
```

if this is verified to be a weakness in the source data/model.

Do not assume it is still the dominant error after redesign.

Verify it.

---

# ROC-AUC

Only calculate meaningful multiclass ROC-AUC for models that produce graded probabilities.

Use:

```python
roc_auc_score(
    y_true,
    y_proba,
    multi_class="ovr",
    average="macro"
)
```

where appropriate.

---

# PR-AUC

Also calculate macro one-vs-rest PR-AUC for probabilistic models.

Use proper class binarization.

---

# DECISION TREE WARNING

An unpruned Decision Tree may produce essentially one-hot leaf probabilities.

Therefore its ROC-AUC curve may have little useful ranking interpretation.

If displayed, explicitly state this limitation.

Prefer not to present its ROC/PR plots as equivalent to genuinely probabilistic models such as:

```text
Random Forest
XGBoost
well-calibrated ensembles
Neural Network
```

---

# CROSS-VALIDATION REPORTING

For Tests 3 and 4, report every metric as:

```text
mean ± standard deviation
```

Example:

```text
Macro F1:
0.913 ± 0.018
```

Never report a single CV fold as the model's score.

---

# PHASE 7 — REQUIRED PLOTS

Generate and SAVE the plots.

Organize them systematically.

Recommended structure:

```text
results/
│
├── eda/
├── feature_engineering/
├── test_1/
├── test_2/
├── test_3/
├── test_4/
└── final_model/
```

---

# EDA PLOTS

Save:

```text
class distribution
histograms
boxplots
bivariate plots
raw correlation heatmap
engineered-feature heatmap
```

---

# MODEL EVALUATION PLOTS

For every appropriate model/experiment create:

## 1. Confusion Matrix — Counts

Example title:

```text
Random Forest — Confusion Matrix (Counts)
Test 3 — 5-Fold Stratified CV
```

---

## 2. Confusion Matrix — Row Normalised

Use class names, not encoded integers where possible.

---

## 3. Per-Class Precision / Recall

Grouped bar chart:

```text
Class
Precision
Recall
```

---

## 4. ROC Curve

For models producing useful graded probabilities only.

Use:

```text
macro one-vs-rest ROC
```

---

## 5. Precision-Recall Curve

For probabilistic models.

Use:

```text
macro one-vs-rest PR
```

---

## 6. Cross-Validation Macro-F1 Box Plot

Create a comparison plot across models:

```text
x-axis → Model
y-axis → Fold Macro F1
```

Use individual fold scores.

This plot should make model stability visible.

---

## 7. Feature Importance

For compatible tree models show:

```text
Top N Features
```

using verified feature names.

Never display:

```text
feature_0
feature_1
...
```

if actual feature names can be recovered.

---

# TUNING PLOTS

For Test 2 and Test 4 save Optuna:

```text
optimization history
parameter importance
```

and other useful Optuna visualisations where appropriate.

For Keras Tuner save:

```text
best-trial learning curves
training vs validation loss
training vs validation macro F1
```

---

# PLOT QUALITY RULES

Every plot must include:

* meaningful title,
* axis labels,
* model name,
* experiment number,
* fold count where relevant,
* readable class names.

Do not produce unlabeled plots.

---

# PHASE 8 — RESULTS TABLE

Build one master results table.

Suggested columns:

```text
Experiment
Feature Set
Model
Accuracy
Macro F1
Balanced Accuracy
Cohen Kappa
Fault Recall
False Alarm Rate
ROC-AUC
PR-AUC
CV Std Macro F1
Training Time
Inference Time
Number of Features
```

Use `NaN` / `Not Applicable` when a metric is inappropriate.

Never fake missing numbers.

Save as:

```text
results/model_comparison.csv
```

---

# PHASE 9 — BEST MODEL SELECTION

Do NOT select the deployment model based solely on accuracy.

Primary criterion:

# Macro F1

But also consider deployment requirements.

Prefer models satisfying:

```text
Fault Detection Recall ≥ 99%
```

if such a model exists.

Among qualifying models compare:

1. macro F1,
2. false-alarm rate,
3. CV variance,
4. balanced accuracy,
5. inference speed,
6. artifact size,
7. deployment complexity,
8. interpretability.

---

# SELECTION RULE

A reasonable hierarchy is:

```text
1. Must have acceptable fault-detection recall
2. Highest Macro F1
3. Lowest false-alarm rate
4. Lower CV variance
5. Lower complexity / faster inference
```

If no model reaches 99% fault recall:

DO NOT manipulate thresholds or results to pretend that it does.

Report:

```text
Target not achieved
```

and select the model with the best verified tradeoff.

---

# DO NOT SELECT USING FINAL TEST

Choose the model architecture/pipeline based only on development results.

After selection:

# evaluate it ONCE on the locked final 15% test set.

That becomes the final unbiased test performance.

---

# FINAL RETRAINING

After final evaluation is documented:

* retain the unbiased test metrics,
* optionally retrain the selected deployment pipeline on the complete available labelled dataset for production,
* clearly distinguish the evaluation artifact from the production-fitted artifact.

Do not report metrics from the production-refitted model as test metrics.

---

# PHASE 10 — SAVE PRODUCTION ARTIFACTS

The deployed system must not manually reproduce preprocessing in random Streamlit code.

For an sklearn-compatible winner, save the COMPLETE pipeline:

```text
feature transformer
+
scaler if required
+
classifier
```

using something like:

```text
models/best_model_pipeline.joblib
```

Also save:

```text
models/model_metadata.json
```

Metadata should contain:

```json
{
  "model_name": "...",
  "experiment": "...",
  "feature_set": "...",
  "raw_inputs": ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"],
  "classes": [],
  "class_mapping": {},
  "macro_f1": 0.0,
  "accuracy": 0.0,
  "fault_detection_recall": 0.0,
  "false_alarm_rate": 0.0,
  "expected_input_features": 6,
  "random_state": 42
}
```

Use actual values.

---

# IF NEURAL NETWORK WINS

Do not pickle the neural network unnecessarily.

Prefer:

```text
models/best_nn.keras
```

and separately save any required preprocessing transformer.

For example:

```text
models/nn_preprocessor.joblib
models/model_metadata.json
```

The deployed inference helper must apply exactly the same transformations used during training.

---

# PHASE 11 — REPOSITORY STRUCTURE

Refactor toward something similar to:

```text
power-grid-fault-detection/
│
├── notebooks/
│   └── power_grid_complete_analysis.ipynb
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── tuning.py
│   └── plotting.py
│
├── utils/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── inference.py
│   └── constants.py
│
├── models/
│   ├── best_model_pipeline.joblib
│   └── model_metadata.json
│
├── results/
│   ├── eda/
│   ├── feature_engineering/
│   ├── test_1/
│   ├── test_2/
│   ├── test_3/
│   ├── test_4/
│   └── model_comparison.csv
│
├── app.py
├── requirements.txt
├── Procfile
├── runtime configuration
├── README.md
└── .gitignore
```

The notebook should remain understandable, but reusable logic should live in `src/`.

Do not duplicate complex feature-engineering logic in:

```text
notebook
AND
Streamlit
```

Import the same tested functions.

---

# PHASE 12 — STREAMLIT FRONTEND

The Streamlit application should intentionally be VERY SIMPLE.

It is an inference demo.

It is not another analytics dashboard.

---

# SINGLE PAGE ONLY

The application should contain:

## Six inputs

```text
Ia
Ib
Ic
Va
Vb
Vc
```

---

# LOAD EXAMPLE BUTTON

Add:

```text
Load example from dataset
```

It should populate the six inputs using a **real labelled row from the repository dataset**.

Prefer allowing the user to load another random sample if clicked again.

Do not fabricate example measurements.

---

# PREDICT BUTTON

Use one button:

# Predict

On click:

```text
Raw six inputs
        ↓
Saved inference pipeline
        ↓
Prediction
        ↓
Class mapping
        ↓
Probability if supported
```

---

# RESULT

Display prominently:

```text
FAULT DETECTED
```

or:

```text
NORMAL OPERATION
```

Then show:

```text
Predicted Fault:
<verified class name>
```

and, when available:

```text
Confidence:
XX.XX%
```

---

# PROBABILITY VISUALISATION

If the deployed model supports meaningful graded probabilities, show a small probability bar chart.

Example:

```text
No Fault       0.03
AB Fault       0.83
BC Fault       0.05
...
```

Use actual model output.

Never fabricate confidence.

If probabilities do not exist:

```text
Probability scores are not available for this model.
```

---

# SIDEBAR

Sidebar should contain ONLY:

```text
Model selector
Verified final test accuracy
```

Optionally show:

```text
Macro F1
```

as a very small secondary line if useful.

Do not clutter the sidebar.

The deployable selector should contain only models for which:

* artifacts exist,
* inference works,
* preprocessing is known,
* metrics are verified.

The selected best model should be the default.

If only one deployment artifact is retained, the selector may contain only that model.

---

# DO NOT ADD TO STREAMLIT

Do NOT include:

```text
Batch CSV upload
EDA dashboard
Correlation plots
Model comparison tables
Feature importance
Optuna plots
Data explorer
Training controls
Retraining
Notebook visualisations
```

Those belong in the notebook and `results/`.

---

# APP SIZE

Keep:

```text
app.py ≈ 150 lines or less
```

as a target.

Place inference code in:

```text
utils/
```

For example:

```text
utils/model_loader.py
utils/inference.py
utils/constants.py
```

---

# STREAMLIT DESIGN

Use native Streamlit.

No elaborate custom CSS.

Use:

```python
st.set_page_config(
    page_title="Power Grid Fault Intelligence",
    page_icon="⚡",
    layout="centered"
)
```

or `wide` only if the probability chart needs it.

Use normal:

```text
st.number_input
st.button
st.metric
st.success
st.error
st.warning
st.bar_chart / Plotly
```

Keep it professional and clean.

---

# MODEL FAILURE RULE

If a model fails to load:

DO NOT silently switch to Random Forest or another model.

Display:

```text
Selected model could not be loaded.
```

and include a concise technical reason.

Never return a fake prediction.

---

# PHASE 13 — DEPLOYMENT PREPARATION

Prepare the final repository for Heroku.

Create a:

```text
Procfile
```

with:

```text
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

Do not hardcode port `8501`.

Heroku provides `$PORT`.

---

# PYTHON VERSION

Pin a tested Python version.

Before finalising deployment files, verify the Python versions currently supported by the active Heroku Python buildpack.

Do not blindly assume an old `runtime.txt` convention is still preferred.

If current Heroku requires:

```text
.python-version
```

instead of:

```text
runtime.txt
```

use the currently supported method and explain the change.

If `runtime.txt` remains supported, pin a verified compatible runtime.

---

# STREAMLIT CONFIG

Create:

```text
.streamlit/config.toml
```

when appropriate.

Configure:

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

Only disable security-related settings when required by the hosting configuration.

Do not change them without explanation.

---

# REQUIREMENTS

Keep production dependencies minimal.

For an sklearn winner, the deployment requirements should resemble:

```text
streamlit
plotly
pandas
numpy
scikit-learn
joblib
```

Pin versions that were actually tested.

Do not automatically use:

```text
scikit-learn==1.8.0
```

unless the final artifacts were created and verified under that version.

Serialized sklearn objects can be version-sensitive.

Pin the exact working environment.

---

# IF NEURAL NETWORK IS DEPLOYED

Avoid dragging a massive training environment into the production slug unnecessarily.

If using modern standalone Keras, determine which currently supported inference backend is compatible with:

* the saved model,
* chosen Keras version,
* Heroku,
* all layers used by the model.

Do not blindly set:

```text
KERAS_BACKEND=numpy
```

without verifying that it is an actual supported backend for the installed Keras version.

If JAX CPU is used, pin a verified compatible combination.

Example conceptual dependencies:

```text
keras
jax[cpu]
```

Only include them if the neural network is actually deployed.

Do not include TensorFlow merely because it was used during experimentation if the final production model does not require it.

---

# DEPLOYMENT SIZE

Measure the actual deployment artifact size.

Before deployment report:

```text
Repository size
Model artifact size
Installed dependency footprint if measurable
Estimated deployment slug/container size
```

Keep only models needed by the demo.

The original Random Forest model is relatively modest in size, but newly tuned artifacts may be much larger.

If deployment size becomes problematic:

1. deploy only the selected best model,
2. remove unused training artifacts from production,
3. keep plots/notebooks in GitHub but exclude unnecessary runtime artifacts,
4. use external model storage only if necessary.

Do not degrade inference correctness simply to reduce file size.

---

# GIT LFS WARNING

Make sure deployable `.pkl`, `.joblib`, or `.keras` artifacts are actually available to the deployed application.

If Git LFS is used, verify the deployment platform resolves LFS files correctly.

Do not deploy a tiny LFS pointer file and then attempt to unpickle it.

---

# HEROKU README

Document:

```text
local installation
local execution
Heroku setup
environment variables
deployment commands
```

Also clearly state current Heroku pricing/free-tier behaviour based on up-to-date official documentation rather than outdated assumptions.

If sleeping/idle behaviour applies to the selected plan, explain its effect on first-load latency.

Also note that Streamlit Community Cloud may be a simpler alternative for a portfolio Streamlit application.

---

# PHASE 14 — README REDESIGN

Rewrite the README so a recruiter can understand the project.

Recommended structure:

```text
Project Title

Problem Statement

Why Power-Grid Fault Classification Matters

Dataset

EDA

Existing Baseline

Problems Identified in Original Pipeline

Physics-Informed Feature Engineering

Polynomial vs Physics-Informed Experiment

Machine Learning Models

Neural Network

Experiment Protocols
    Test 1
    Test 2
    Test 3
    Test 4

Metrics

Results

Best Model

Confusion Matrix

Deployment

Streamlit Demo

Project Structure

How to Run

Limitations

Future Work
```

Do not exaggerate performance or deployment readiness.

---

# PHASE 15 — REPRODUCIBILITY

Use a central random seed:

```python
RANDOM_STATE = 42
```

Set seeds for:

```text
NumPy
scikit-learn models where available
XGBoost
LightGBM
CatBoost
Keras
```

as much as practically possible.

Document remaining nondeterminism.

---

# PHASE 16 — COMPUTE MANAGEMENT

The full Test-4 nested-tuning experiment can be expensive.

Therefore make trial budgets configurable.

For example:

```python
FAST_MODE = False

OPTUNA_TRIALS = 50
KERAS_TRIALS = 30
```

For nested CV, lower trial counts may be used only if clearly documented.

Do not silently replace nested CV with normal CV because runtime is inconvenient.

If a full experiment is computationally prohibitive in the current environment:

1. implement the correct pipeline,
2. run the largest scientifically valid subset possible,
3. clearly mark experiments not fully executed,
4. never fabricate missing metrics.

---

# PHASE 17 — ACCEPTANCE TESTS

Before saying the project is finished, verify everything.

## Data

* dataset loads,
* labels verified,
* no accidental target leakage,
* missing/infinite values handled.

## Features

* polynomial feature count verified,
* physics feature count verified,
* no NaN/inf impedance values,
* feature names preserved,
* feature transformer deterministic.

## Test 1

* all mandatory models run,
* baseline neural network runs.

## Test 2

* Optuna runs for ML,
* Keras Tuner runs for NN,
* macro F1 is optimization objective.

## Test 3

* five-fold StratifiedKFold works,
* preprocessing occurs inside fold,
* mean ± std reported.

## Test 4

* outer folds separated from inner tuning,
* Optuna is nested,
* NN tuning is nested,
* outer-fold metrics reported.

## Metrics

Verify:

* accuracy,
* macro F1,
* balanced accuracy,
* Cohen's κ,
* per-class precision,
* per-class recall,
* fault recall,
* false-alarm rate,
* confusion matrix.

Verify ROC/PR metrics only where mathematically appropriate.

## Final Model

* selected without looking at locked test,
* evaluated once on locked test,
* pipeline saved,
* metadata saved,
* reload gives identical predictions.

## Streamlit

* app launches,
* six inputs work,
* real example loads,
* selected model loads,
* prediction works,
* class decoding works,
* confidence corresponds to real probabilities,
* app handles invalid model cleanly.

## Deployment

* Procfile valid,
* runtime configuration valid,
* requirements install,
* model files accessible,
* Streamlit binds to `$PORT`,
* deployment smoke test succeeds.

---

# PHASE 18 — FINAL RESPONSE FORMAT

When the entire project has been modified, give me a concise but complete final report with these sections.

## 1. Original Pipeline Audit

Explain the major problems discovered.

## 2. EDA Findings

Summarise the most important observations.

## 3. Feature Engineering Experiment

Give a comparison table:

```text
Feature Set
Feature Count
Macro F1
Balanced Accuracy
Fault Recall
False Alarm Rate
CV Variance
```

Explain whether physics-informed engineering actually beat polynomial expansion.

## 4. Test 1 Results

Default models.

## 5. Test 2 Results

Optuna + Keras Tuner.

## 6. Test 3 Results

5-fold CV.

## 7. Test 4 Results

Nested CV + tuning.

## 8. Best Model

State:

```text
Model
Feature set
Hyperparameters
Macro F1
Accuracy
Fault Recall
False Alarm Rate
Balanced Accuracy
Cohen's κ
```

## 9. Final Locked-Test Performance

Clearly distinguish this from CV results.

## 10. Saved Artifacts

List every important saved file.

## 11. Streamlit

Give:

```bash
streamlit run app.py
```

## 12. Deployment

Explain exact Heroku deployment steps.

## 13. Limitations

Be explicit.

## 14. Improvements Over Original Repository

Explain what now makes this project stronger for ML/Data Science interviews.

---

# NON-NEGOTIABLE RULES

Do not:

* fabricate metrics,
* fabricate plots,
* fabricate confidence scores,
* intentionally force the physics-informed method to win,
* use the final test set for tuning,
* preprocess the full dataset before splitting,
* oversample before splitting,
* fit the scaler before CV,
* use accuracy as the Optuna objective,
* use accuracy as the Keras-Tuner objective,
* report a single CV fold,
* retrain the model during Streamlit inference,
* duplicate preprocessing manually in `app.py`,
* silently substitute another model if loading fails,
* claim a statistical outlier is bad data without justification,
* claim a feature is physics-informed without verifying its formula,
* retain mathematically broken features because the old notebook used them.

---

# PRIORITY ORDER

If tradeoffs arise, prioritize:

```text
1. No data leakage
2. Correct experimental methodology
3. Macro-F1 / safety-related performance
4. Reproducibility
5. Correct physics-informed features
6. Stable inference
7. Interpretability
8. Deployment simplicity
9. UI polish
```

The Streamlit frontend is intentionally simple.

The scientific redesign of the ML pipeline is the main project.

---

# START NOW

Begin by cloning/accessing the GitHub repository and auditing the existing notebook and artifacts.

Do not start with Streamlit.

Proceed in this order:

```text
Audit
→ Baseline Reproduction
→ EDA
→ Feature Engineering Experiment
→ Test 1
→ Test 2
→ Test 3
→ Test 4
→ Comprehensive Metrics
→ Best Model Selection
→ Locked Test Evaluation
→ Save Pipeline
→ Streamlit
→ Deployment
→ README
→ Final Verification
```

Work directly on the project and implement the changes rather than merely describing what code I should write.
