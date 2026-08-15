# AGENTS.md

## Project

Power-grid fault detection/classification system. Six raw three-phase current/voltage measurements → classify into 6 fault classes.

## Authoritative Specification

**`docs/PROJECT_SPEC.md` is the single source of truth for all design decisions.** If anything in the existing code, README, notebook, comments, or this file conflicts with PROJECT_SPEC.md, the spec wins.

## Mandatory Phase Order

Work on ONE phase at a time. Do not skip phases.

```text
Audit → Baseline Reproduction → EDA → Feature Engineering Experiment
→ Test 1 → Test 2 → Test 3 → Test 4 → Comprehensive Evaluation
→ Best Model Selection → Locked Final-Test Evaluation
→ Final Retraining → Save Production Pipeline → Streamlit → Deployment → README → Final Verification
```

## Non-Negotiable Rules

1. **Never fabricate** metrics, plots, confidence scores, experiments, or results.
2. **Prevent data leakage.** Anything that learns from data (scaler, imputer, resampler, feature selector) must be fitted ONLY on the training fold, inside a `Pipeline`.
3. **The locked 15% final test set is untouchable** during feature selection, tuning, architecture decisions, or model selection. It is used ONCE after the best model is chosen.
4. **Macro F1 is the main model-selection metric.** Also consider fault recall, false-alarm rate, balanced accuracy, and CV stability.
5. **Physics-informed features must be mathematically valid.** Verify every formula. Do not retain a feature because its name sounds physical. Compare objectively against the polynomial baseline.
6. **Fitting a scaler before splitting is forbidden** in the redesigned experiments. The legacy baseline reproduction may reproduce the original leakage for comparison, but all new experiments must fit inside training folds.
7. **RobustScaler is the required scaler for scale-sensitive models** in the redesigned experiments (Tests 1–4). This applies to Logistic Regression, KNN, SVM, Neural Network, and any other distance/gradient-sensitive model. Tree-based models (Decision Tree, Random Forest, Extra Trees, Bagging, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost) generally do not require scaling and may omit RobustScaler unless there is a specific experimentally justified reason. When used, RobustScaler must be fitted ONLY on the training data / training fold. Never on validation, test, or the full dataset.
8. **PolynomialFeatures(degree=2, include_bias=False) is deterministic** and does not itself create information leakage when fitted before splitting. However, it should still live inside the redesigned Pipeline for consistency. Do not conflate it with MinMaxScaler leakage.
9. **Investigate suspicious near-perfect results rigorously.** Do not attribute high tree-model scores to scaler leakage alone. Check for duplicates, near-duplicates, simulation structure, random-split optimism, repeated scenarios, target leakage, and deterministic class structure before drawing conclusions.
10. **Do not use accuracy as the Optuna or Keras Tuner objective.** Use macro F1.
11. **Do not silently substitute another model if loading fails.** Display an error.
12. **Do not claim a statistical outlier is bad data without physical justification.** In fault data, extremes may be the fault signal.
13. **Never report a single CV fold as the model's score.** Always mean ± std.

## Legacy Baseline

The existing repository is the LEGACY BASELINE. It must be audited and reproduced before redesign. The following facts are preserved from `/init` to support baseline reproduction.

### Existing Commands

```bash
# Run the legacy app
streamlit run app.py

# Run legacy tests (custom @check decorators, NOT pytest)
python tests/test_pipeline.py    # 23 checks: inference correctness + notebook accuracy reproduction
python tests/test_app_smoke.py   # 23 checks: headless Streamlit render over every page/model
```

No linting, formatting, or CI configuration exists.

### Legacy Architecture

- **`app.py`** — Entrypoint. Sets `KERAS_BACKEND=numpy` and `TF_CPP_MIN_LOG_LEVEL=3` before imports, renders sidebar (page selector + model selector), routes to view modules.
- **`utils/`** — Inference logic (no Streamlit UI except `@st.cache_*`)
  - `constants.py` — Paths, feature names, class map, model registry with verified accuracies
  - `preprocessing.py` — Refits `PolynomialFeatures(degree=2)` + `MinMaxScaler` on `classData.csv` at runtime (transformers not shipped)
  - `model_loader.py` — Loads `.pkl` artifacts via `joblib`, `@st.cache_resource` cached
  - `prediction.py` — `predict()` entry point returning `PredictionResult` dataclass
- **`views/`** — One Streamlit page per module. Deliberately **not** named `pages/` (Streamlit auto-builds multipage nav from `pages/`, breaking the custom sidebar router).

### Legacy Data Flow

```
Raw input (6 values) → validate_raw_frame() → PolynomialFeatures(degree=2) → MinMaxScaler.transform() → model.predict() → CLASS_INDEX_TO_LABEL → PredictionResult

Note: This legacy pipeline uses MinMaxScaler, which was fit on the full dataset. This is deliberate reproduction of the original leakage for baseline comparison. The redesigned experiments (Tests 1–4) use RobustScaler for scale-sensitive models, fitted inside training folds only.
```

### Dependency/Runtime Constraints

- **`scikit-learn==1.8.0` pinned exactly** — Tree artifacts record `_sklearn_version == 1.8.0`. Unpickling across versions is not guaranteed.
- **`KERAS_BACKEND=numpy` must be set before any Keras import** — Keras 3 defaults to TensorFlow backend and raises `ModuleNotFoundError` if TensorFlow is absent.
- **`DL_model.pkl` uses Keras 3.13+ with JAX/numpy** — TensorFlow was removed to save ~600 MB.

### Known Legacy Gotchas

1. **Preprocessing leakage reproduced deliberately** — The notebook fits `MinMaxScaler` on the entire dataset before splitting. The app reproduces this because shipped models were trained under this transform. The redesigned project must NOT repeat this.
2. **Scalers/encoders NOT shipped** — Reconstructed by refitting on `classData.csv` at runtime. Works because the notebook fitted on unshuffled full data.
3. **`DL_model.pkl` is a Dense MLP, not CNN-LSTM** — `Dense(128→64→32→6)` with BatchNorm + Dropout 0.5. Predicts class 0 for ~68% of test rows (dropout-induced collapse).
4. **Two class labels are questionable** — Code `1001` named "Line-to-Line AB" (actually single line-to-ground), code `0110` named "with Ground" (G=0). The spec says to flag these, not silently rename.
5. **`views/` folder naming** — Must NOT be renamed to `pages/`.

### Legacy Test Approach

Tests use custom `@check` decorators (not pytest). Each prints `PASS`/`FAIL` per check and exits 0 or 1. Run directly with `python`.

### Legacy Model Artifacts

- `decision_trees_model.pkl` — DecisionTreeClassifier
- `random_forest_model.pkl` — RandomForestClassifier (~16 MB)
- `DL_model.pkl` — Keras Dense MLP

Legacy reference accuracies (baseline reproduction only, not targets):
- Decision Tree ≈ 89.89%
- Random Forest ≈ 88.30%

## What NOT to Preserve

The following are legacy decisions that the spec explicitly replaces. Do NOT carry them forward into the redesign:

- Polynomial-only feature engineering (spec requires a physics-informed experiment)
- Fitting scalers before splitting
- Accuracy as the primary metric
- Multi-page Streamlit dashboard (spec: single-page inference demo only)
- Batch CSV upload, EDA dashboard, correlation plots, model comparison tables, feature importance, Optuna plots in Streamlit (spec: belongs in notebook/results only)
- CNN-LSTM claim for DL model (artifact is Dense MLP)

## Priority Order

When tradeoffs arise:

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
