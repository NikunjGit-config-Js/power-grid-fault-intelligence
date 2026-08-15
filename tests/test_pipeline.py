"""
Verification suite for the inference layer.

Run from the repository root:

    python tests/test_pipeline.py

The important test is `test_reproduces_notebook_accuracy`. It replays the notebook's
train/test split, scores each shipped artifact, and asserts the result matches the
accuracy the notebook printed. If the reconstructed scaler or the feature order were
wrong in any way, those numbers would not line up.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.constants import (
    CLASS_LABELS,
    MODEL_REGISTRY,
    N_MODEL_FEATURES,
    POLY_FEATURE_NAMES,
    RAW_FEATURES,
)
from utils.model_loader import load_model, model_availability, model_runtime_facts
from utils.prediction import feature_importance_frame, predict, predict_frame
from utils.preprocessing import (
    PreprocessingError,
    get_feature_pipeline,
    load_class_data,
    load_detect_data,
    prepare_features,
    sample_row,
)

# Printed by the notebook; see cell 40 (trees) and cell 46 (neural network).
NOTEBOOK_ACCURACY = {
    "decision_tree": 0.898919,
    "random_forest": 0.883026,
    "deep_learning": 0.385251,
}

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    def wrap(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
        return fn

    return wrap


def _test_split():
    """Rebuild the notebook's exact split so metrics are comparable."""
    df = load_class_data()
    X = prepare_features(df[RAW_FEATURES])
    y = LabelEncoder().fit_transform(df["Fault_Type"])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main() -> int:
    print("\n=== 1. datasets ===")

    @check("classData.csv loads with 6 mapped classes")
    def _():
        df = load_class_data()
        assert len(df) == 7861, len(df)
        assert sorted(df["Fault_Type"].unique()) == sorted(CLASS_LABELS)
        assert df["Fault_Type"].isna().sum() == 0

    @check("detect_dataset.csv loads and drops empty columns")
    def _():
        df = load_detect_data()
        assert len(df) == 12001, len(df)
        assert not [c for c in df.columns if str(c).startswith("Unnamed")]

    print("\n=== 2. feature engineering ===")

    @check("feature order matches the constant exactly")
    def _():
        _, _, names = get_feature_pipeline()
        assert names == POLY_FEATURE_NAMES
        assert len(names) == N_MODEL_FEATURES == 27

    @check("single row transforms to shape (1, 27)")
    def _():
        row = sample_row(seed=3)
        row.pop("_actual_label")
        X = prepare_features(row)
        assert X.shape == (1, 27), X.shape
        assert np.isfinite(X).all()

    @check("scaled training data lies in [0, 1]")
    def _():
        X = prepare_features(load_class_data()[RAW_FEATURES])
        assert X.shape == (7861, 27), X.shape
        assert X.min() >= -1e-9 and X.max() <= 1 + 1e-9, (X.min(), X.max())

    print("\n=== 3. model loading ===")

    for key in MODEL_REGISTRY:
        @check(f"{key} loads and reports 27 inputs")
        def _(key=key):
            status = model_availability()[key]
            assert status["ok"], status["error"]
            facts = model_runtime_facts(key)
            assert facts["n_features_in"] == 27, facts["n_features_in"]

    @check("DL artifact is a Keras Sequential with output shape (None, 6)")
    def _():
        facts = model_runtime_facts("deep_learning")
        assert "keras" in facts["python_type"].lower(), facts["python_type"]
        assert tuple(facts["input_shape"]) == (None, 27), facts["input_shape"]
        assert tuple(facts["output_shape"]) == (None, 6), facts["output_shape"]

    print("\n=== 4. notebook accuracy reproduction ===")

    _, X_test, _, y_test = _test_split()

    for key, expected in NOTEBOOK_ACCURACY.items():
        @check(f"{key} reproduces notebook accuracy {expected:.6f}")
        def _(key=key, expected=expected):
            model = load_model(key)
            if MODEL_REGISTRY[key]["loader"] == "keras":
                pred = np.asarray(model.predict(X_test, verbose=0)).argmax(axis=1)
            else:
                pred = model.predict(X_test)
            acc = float((pred == y_test).mean())
            assert abs(acc - expected) < 1e-5, f"got {acc:.6f}, expected {expected:.6f}"
            assert abs(acc - MODEL_REGISTRY[key]["test_accuracy"]) < 1e-5

    print("\n=== 5. prediction and decoding ===")

    @check("every model predicts a valid decoded label")
    def _():
        row = sample_row("No Fault", seed=42)
        row.pop("_actual_label")
        for key in MODEL_REGISTRY:
            r = predict(key, row)
            assert r.labels[0] in CLASS_LABELS, r.labels
            assert 0 <= int(r.class_indices[0]) <= 5

    @check("tree models classify a known No Fault row correctly")
    def _():
        row = sample_row("No Fault", seed=42)
        row.pop("_actual_label")
        for key in ("decision_tree", "random_forest"):
            r = predict(key, row)
            assert r.labels[0] == "No Fault", f"{key} said {r.labels[0]}"
            assert not r.is_fault()

    @check("probabilities are valid and aligned to classes_")
    def _():
        row = sample_row(seed=5)
        row.pop("_actual_label")
        for key in MODEL_REGISTRY:
            r = predict(key, row)
            assert r.has_probabilities, key
            p = r.probabilities
            assert p.shape == (1, 6), p.shape
            assert abs(p.sum() - 1.0) < 1e-4, p.sum()
            assert int(p.argmax(axis=1)[0]) == int(r.class_indices[0]), key

    @check("feature importances are named, not indexed")
    def _():
        for key in ("decision_tree", "random_forest"):
            frame = feature_importance_frame(key, top_n=27)
            assert len(frame) == 27
            assert set(frame["Feature"]) == set(POLY_FEATURE_NAMES)
            assert abs(frame["Importance"].sum() - 1.0) < 1e-6
        assert feature_importance_frame("deep_learning") is None

    print("\n=== 6. batch prediction ===")

    @check("valid CSV scores end to end")
    def _():
        df = load_class_data().sample(50, random_state=9)[RAW_FEATURES]
        out = predict_frame("decision_tree", df)
        assert len(out) == 50
        for col in ("Predicted_Class", "Predicted_Fault", "Status", "Confidence"):
            assert col in out.columns, col
        assert set(out["Status"]) <= {"FAULT", "NORMAL"}

    @check("CSV with reordered and extra columns still works")
    def _():
        df = load_class_data().sample(10, random_state=4)[["Vc", "Ib", "Va", "Ia", "Vb", "Ic"]]
        df["note"] = "ignore me"
        out = predict_frame("random_forest", df)
        assert len(out) == 10

    @check("rows with non-numeric values are skipped, not guessed")
    def _():
        # A real CSV upload yields object-dtype columns when a cell is not numeric.
        df = load_class_data().sample(10, random_state=6)[RAW_FEATURES].copy()
        df["Ia"] = df["Ia"].astype(object)
        df.iloc[0, df.columns.get_loc("Ia")] = "broken"
        df.iloc[1, df.columns.get_loc("Vb")] = np.nan
        out = predict_frame("decision_tree", df)
        assert len(out) == 8, len(out)
        assert out.attrs["n_dropped"] == 2

    @check("missing column raises a readable error")
    def _():
        df = load_class_data().sample(5, random_state=2)[["Ia", "Ib", "Ic", "Va", "Vb"]]
        try:
            predict_frame("decision_tree", df)
        except PreprocessingError as exc:
            assert "Vc" in str(exc), str(exc)
        else:
            raise AssertionError("expected PreprocessingError")

    @check("malformed CSV text raises a readable error")
    def _():
        bad = io.StringIO("this is not,a valid\ncsv,file,at,all,really\n")
        try:
            df = pd.read_csv(bad)
            predict_frame("decision_tree", df)
        except PreprocessingError:
            pass
        except pd.errors.ParserError:
            pass
        else:
            raise AssertionError("expected an error")

    @check("empty frame raises a readable error")
    def _():
        try:
            predict_frame("decision_tree", pd.DataFrame(columns=RAW_FEATURES))
        except PreprocessingError:
            pass
        else:
            raise AssertionError("expected PreprocessingError")

    print("\n=== 7. shape guards ===")

    @check("wrong-width input is rejected")
    def _():
        try:
            prepare_features(pd.DataFrame([[1.0, 2.0, 3.0]], columns=["Ia", "Ib", "Ic"]))
        except PreprocessingError:
            pass
        else:
            raise AssertionError("expected PreprocessingError")

    print("\n" + "=" * 66)
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name, err in FAILED:
            print(f"    FAILED: {name}\n            {err}")
    print("=" * 66)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
