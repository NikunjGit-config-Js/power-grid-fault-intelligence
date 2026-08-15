"""
Central, single-source-of-truth constants for the Power Grid Fault Intelligence app.

Every value in this file was VERIFIED against the repository artifacts, not guessed:

  * feature order        -> PolynomialFeatures(degree=2, include_bias=False).get_feature_names_out()
                            reproduced from the notebook and asserted at runtime
  * class index -> label -> LabelEncoder() fitted on data['Fault_Type'] (alphabetical order),
                            cross-checked against `class_labels` hardcoded in notebook cell 50
  * accuracies           -> recomputed against the shipped .pkl artifacts; they reproduce the
                            notebook's printed outputs to 4 decimal places
"""

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_DATA_PATH = PROJECT_ROOT / "classData.csv"
DETECT_DATA_PATH = PROJECT_ROOT / "detect_dataset.csv"

MODEL_PATHS = {
    "decision_tree": PROJECT_ROOT / "decision_trees_model.pkl",
    "random_forest": PROJECT_ROOT / "random_forest_model.pkl",
    "deep_learning": PROJECT_ROOT / "DL_model.pkl",
    "lightgbm": PROJECT_ROOT / "models" / "best_model_pipeline.joblib",
}

# --------------------------------------------------------------------------------------
# Production model (new redesigned pipeline)
# --------------------------------------------------------------------------------------
PRODUCTION_MODEL_KEY = "lightgbm"
PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "best_model_pipeline.joblib"
PRODUCTION_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

# --------------------------------------------------------------------------------------
# Raw sensor inputs -- exactly what the user supplies in the UI
# --------------------------------------------------------------------------------------
RAW_FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
CURRENT_FEATURES = ["Ia", "Ib", "Ic"]
VOLTAGE_FEATURES = ["Va", "Vb", "Vc"]

FAULT_INDICATOR_COLUMNS = ["G", "C", "B", "A"]

# --------------------------------------------------------------------------------------
# Engineered feature vector fed to every shipped model.
#
# Source: notebook cell 35
#     poly = PolynomialFeatures(degree=2, include_bias=False)
#     poly_features = poly.fit_transform(data[['Ia','Ib','Ic','Va','Vb','Vc']])
#
# 6 raw terms + 21 degree-2 terms = 27 features. All three shipped artifacts report
# n_features_in_ == 27 / input_shape == (None, 27), which confirms this stage.
#
# The order below is the literal output of get_feature_names_out() and is asserted
# against the live transformer in preprocessing.get_feature_pipeline().
# --------------------------------------------------------------------------------------
POLY_FEATURE_NAMES = [
    "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
    "Ia^2", "Ia Ib", "Ia Ic", "Ia Va", "Ia Vb", "Ia Vc",
    "Ib^2", "Ib Ic", "Ib Va", "Ib Vb", "Ib Vc",
    "Ic^2", "Ic Va", "Ic Vb", "Ic Vc",
    "Va^2", "Va Vb", "Va Vc",
    "Vb^2", "Vb Vc",
    "Vc^2",
]

N_MODEL_FEATURES = len(POLY_FEATURE_NAMES)  # 27

# --------------------------------------------------------------------------------------
# Fault-label construction (notebook cell 11)
#
#     data['Fault_Type'] = data[['G','C','B','A']].astype(str).agg(''.join, axis=1)
#
# Only six G/C/B/A codes actually occur in classData.csv. The notebook's lookup dict
# contains other codes too, but they never match a row, so they are irrelevant here.
# --------------------------------------------------------------------------------------
FAULT_CODE_TO_LABEL = {
    "0000": "No Fault",
    "1001": "Line-to-Line AB",
    "0110": "Line-to-Line with Ground BC",
    "0111": "Three-Phase",
    "1011": "Line A Line B to Ground Fault",
    "1111": "Three-Phase with Ground",
}

# --------------------------------------------------------------------------------------
# Encoded class index -> label.
#
# LabelEncoder sorts alphabetically, so the mapping below is deterministic. It matches
# the `class_labels` list the notebook hardcodes in cell 50 when printing its
# classification reports, so the two agree.
#
# `physical` describes what the G/C/B/A bit pattern means electrically. Where the
# notebook's own name disagrees with the bit pattern, `label_note` records it -- the app
# surfaces this rather than silently "fixing" a label the model was trained under.
# --------------------------------------------------------------------------------------
CLASS_INDEX_TO_LABEL = {
    0: "Line A Line B to Ground Fault",
    1: "Line-to-Line AB",
    2: "Line-to-Line with Ground BC",
    3: "No Fault",
    4: "Three-Phase",
    5: "Three-Phase with Ground",
}

CLASS_LABELS = [CLASS_INDEX_TO_LABEL[i] for i in range(len(CLASS_INDEX_TO_LABEL))]

N_CLASSES = len(CLASS_LABELS)

CLASS_METADATA = {
    0: {
        "label": "Line A Line B to Ground Fault",
        "code": "1011",
        "short": "LLG (A-B-G)",
        "is_fault": True,
        "physical": "Phases A and B shorted together and to ground (double line-to-ground).",
        "label_note": None,
    },
    1: {
        "label": "Line-to-Line AB",
        "code": "1001",
        "short": "LL (A-B)",
        "is_fault": True,
        "physical": "Ground flag set with phase A energised.",
        "label_note": (
            "The source dataset documents code 1001 as a single line-to-ground fault on "
            "phase A. The notebook names it 'Line-to-Line AB'. The name is kept as-is "
            "because the model was trained under it, but treat the wording with care."
        ),
    },
    2: {
        "label": "Line-to-Line with Ground BC",
        "code": "0110",
        "short": "LL (B-C)",
        "is_fault": True,
        "physical": "Phases B and C shorted together; the ground flag G is 0.",
        "label_note": (
            "Code 0110 has G=0, so ground is not involved despite 'with Ground' in the "
            "notebook's name. Kept as-is for consistency with the trained artifacts."
        ),
    },
    3: {
        "label": "No Fault",
        "code": "0000",
        "short": "Normal",
        "is_fault": False,
        "physical": "Balanced three-phase operation, no fault indicator set.",
        "label_note": None,
    },
    4: {
        "label": "Three-Phase",
        "code": "0111",
        "short": "LLL",
        "is_fault": True,
        "physical": "All three phases shorted together, ground not involved (symmetrical fault).",
        "label_note": None,
    },
    5: {
        "label": "Three-Phase with Ground",
        "code": "1111",
        "short": "LLLG",
        "is_fault": True,
        "physical": "All three phases shorted together and to ground; most severe case.",
        "label_note": None,
    },
}

# --------------------------------------------------------------------------------------
# Model registry.
#
# `test_accuracy` values were RECOMPUTED in this repo by rebuilding the notebook's
# train_test_split(test_size=0.2, random_state=42, stratify=y) and scoring the shipped
# artifact. Each reproduces the notebook's printed number exactly, so they are verified
# rather than copied on trust.
# --------------------------------------------------------------------------------------
MODEL_REGISTRY = {
    "decision_tree": {
        "display_name": "Decision Tree",
        "file": "decision_trees_model.pkl",
        "loader": "joblib",
        "family": "scikit-learn",
        "type_name": "DecisionTreeClassifier",
        "n_features": 27,
        "supports_proba": True,
        "has_feature_importances": True,
        "test_accuracy": 0.898919,
        "cv_accuracy": 0.892970,
        "trained_on": "27 polynomial features, min-max scaled, SMOTE-balanced training split",
        "strengths": [
            "Best verified accuracy of the three shipped artifacts.",
            "Fully interpretable - the decision path can be read end to end.",
            "Smallest artifact (186 KB) and fastest inference.",
        ],
        "limitations": [
            "A single unpruned tree (max_depth=None) memorises the training split.",
            "Confuses 'Three-Phase' with 'Three-Phase with Ground', as the notebook notes.",
            "predict_proba returns near-hard 0/1 values, so confidence is not well calibrated.",
        ],
    },
    "random_forest": {
        "display_name": "Random Forest",
        "file": "random_forest_model.pkl",
        "loader": "joblib",
        "family": "scikit-learn",
        "type_name": "RandomForestClassifier",
        "n_features": 27,
        "supports_proba": True,
        "has_feature_importances": True,
        "test_accuracy": 0.883026,
        "cv_accuracy": 0.893939,
        "trained_on": "27 polynomial features, min-max scaled, SMOTE-balanced training split",
        "strengths": [
            "100-tree ensemble, so it is more stable than the single tree.",
            "Best-calibrated probabilities of the three (votes averaged across trees).",
            "Gives usable feature-importance rankings.",
        ],
        "limitations": [
            "Scores slightly below the single Decision Tree on this test split.",
            "Largest artifact at 16 MB, which dominates repository size.",
            "Shares the three-phase confusion of the Decision Tree.",
        ],
    },
    "deep_learning": {
        "display_name": "Deep Learning (Dense MLP)",
        "file": "DL_model.pkl",
        "loader": "keras",
        "family": "Keras 3",
        "type_name": "keras.Sequential",
        "n_features": 27,
        "supports_proba": True,
        "has_feature_importances": False,
        "test_accuracy": 0.385251,
        "cv_accuracy": None,
        "trained_on": "27 polynomial features, min-max scaled (no SMOTE)",
        "strengths": [
            "Genuine softmax outputs, so probabilities are smooth rather than near-binary.",
            "Included for an honest architecture-versus-tree comparison.",
        ],
        "limitations": [
            "Verified test accuracy is only 38.53% - well below both tree models.",
            "Dropout of 0.5 after every layer starves a network this small (14,566 trainable params).",
            "Collapses toward one class: it predicted class 0 for 1,070 of 1,573 test rows.",
            "Not recommended for the demo. It is exposed for transparency, not performance.",
        ],
    },
}

MODEL_KEYS = list(MODEL_REGISTRY.keys())

# --------------------------------------------------------------------------------------
# Notebook results for models that are NOT shipped as artifacts.
#
# Source: notebook cell 60. These were trained on the LATER 37-feature pipeline
# (27 polynomial + 10 domain features) and saved under different filenames that are not
# in this repository. Shown for context only -- the app never runs inference with them.
# --------------------------------------------------------------------------------------
NOTEBOOK_ONLY_RESULTS = [
    ("Logistic Regression", 0.824585, 0.836618),
    ("Support Vector Machines", 0.821883, 0.831532),
    ("K-Nearest Neighbors", 0.839853, 0.811825),
    ("Decision Trees", 0.994911, 0.996186),
    ("Random Forest", 0.998251, 0.999364),
    ("Gradient Boosting", 0.997773, 0.998729),
    ("Neural Networks", 0.859574, 0.857597),
    ("Naive Bayes", 0.948314, 0.947235),
    ("AdaBoost", 0.503816, 0.490782),
    ("XGBoost", 0.998251, 0.999364),
    ("LightGBM", 0.999046, 1.000000),
    ("CatBoost", 0.999682, 1.000000),
]

# Test-split geometry reproduced from the notebook, used in the UI for context.
TEST_SPLIT = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
    "n_train": 6288,
    "n_test": 1573,
}

APP_TITLE = "Power Grid Fault Intelligence"
APP_ICON = "⚡"
