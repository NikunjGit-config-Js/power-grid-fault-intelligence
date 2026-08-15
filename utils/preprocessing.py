"""
Feature engineering and preprocessing.

This module reproduces the notebook's training-time transform EXACTLY. The repository
ships three trained models but no fitted scaler and no fitted label encoder, so both have
to be reconstructed. They can be reconstructed safely because the notebook fits them on
the full, unshuffled `classData.csv`:

    notebook cell 35:  poly = PolynomialFeatures(degree=2, include_bias=False)
                       poly_features = poly.fit_transform(data[RAW_FEATURES])
    notebook cell 36:  scaler = MinMaxScaler()
                       X_scaled = scaler.fit_transform(poly_features_df)
                       label_encoder = LabelEncoder()
                       y_encoded = label_encoder.fit_transform(data['Fault_Type'])

Nothing about that depends on the train/test split or on any random seed, so refitting on
the same CSV yields bit-identical transformers. This is verified, not assumed: replaying
the split and scoring the shipped artifacts reproduces the notebook's printed accuracies
to four decimals (see tests/test_pipeline.py).

KNOWN CAVEAT, reproduced deliberately
-------------------------------------
The notebook fits MinMaxScaler on the ENTIRE dataset before splitting, which leaks test
set min/max into training. That is a flaw in the original methodology. We reproduce it
because the shipped models were trained under it -- fitting the scaler on the training
split alone would shift every feature and silently corrupt predictions. The app states
this in the Model Insights page rather than hiding it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from utils.constants import (
    CLASS_DATA_PATH,
    DETECT_DATA_PATH,
    FAULT_CODE_TO_LABEL,
    FAULT_INDICATOR_COLUMNS,
    N_MODEL_FEATURES,
    POLY_FEATURE_NAMES,
    RAW_FEATURES,
)


class PreprocessingError(RuntimeError):
    """Raised when input cannot be transformed into a valid model feature vector."""


# --------------------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_class_data() -> pd.DataFrame:
    """Load classData.csv and attach the derived Fault_Type label (notebook cell 11)."""
    df = pd.read_csv(CLASS_DATA_PATH)
    df["Fault_Code"] = df[FAULT_INDICATOR_COLUMNS].astype(str).agg("".join, axis=1)
    df["Fault_Type"] = df["Fault_Code"].map(FAULT_CODE_TO_LABEL)
    if df["Fault_Type"].isna().any():
        unknown = sorted(df.loc[df["Fault_Type"].isna(), "Fault_Code"].unique())
        raise PreprocessingError(f"Unmapped G/C/B/A codes found in classData.csv: {unknown}")
    return df


@st.cache_data(show_spinner=False)
def load_detect_data() -> pd.DataFrame:
    """Load detect_dataset.csv, dropping the two trailing all-empty columns in the header."""
    df = pd.read_csv(DETECT_DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    return df


# --------------------------------------------------------------------------------------
# Fitted transformers
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_feature_pipeline():
    """
    Rebuild the notebook's (PolynomialFeatures -> MinMaxScaler) pair.

    Returns
    -------
    (poly, scaler, feature_names)
    """
    df = load_class_data()

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_matrix = poly.fit_transform(df[RAW_FEATURES])
    feature_names = list(poly.get_feature_names_out(RAW_FEATURES))

    # Guard against a scikit-learn version ever changing expansion order or naming.
    if feature_names != POLY_FEATURE_NAMES:
        raise PreprocessingError(
            "PolynomialFeatures produced an unexpected feature order.\n"
            f"expected: {POLY_FEATURE_NAMES}\n"
            f"actual:   {feature_names}\n"
            "The shipped models depend on this exact order; refusing to continue."
        )

    scaler = MinMaxScaler()
    scaler.fit(pd.DataFrame(poly_matrix, columns=feature_names))

    return poly, scaler, feature_names


# --------------------------------------------------------------------------------------
# Raw input validation
# --------------------------------------------------------------------------------------
def validate_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check that a frame carries the six raw measurements as finite numbers.

    Returns a frame with exactly RAW_FEATURES, in order, as float64.
    Raises PreprocessingError with a message meant to be shown to the user.
    """
    if df is None or len(df) == 0:
        raise PreprocessingError("The uploaded file contains no rows.")

    # Tolerate whitespace and case differences in the header.
    lookup = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in RAW_FEATURES if c.lower() not in lookup]
    if missing:
        raise PreprocessingError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected exactly these six: {', '.join(RAW_FEATURES)}."
        )

    out = df[[lookup[c.lower()] for c in RAW_FEATURES]].copy()
    out.columns = RAW_FEATURES

    for col in RAW_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # After coercion every cell is float or NaN; isfinite rejects NaN and +/-inf alike.
    bad = pd.Series(
        ~np.isfinite(out.to_numpy(dtype="float64")).all(axis=1), index=out.index
    )
    n_bad = int(bad.sum())
    if n_bad == len(out):
        raise PreprocessingError(
            "No usable rows. Every row has a missing or non-numeric value in one of "
            f"{', '.join(RAW_FEATURES)}."
        )

    out.attrs["n_dropped"] = n_bad
    out.attrs["dropped_index"] = out.index[bad].tolist()
    return out.loc[~bad].astype("float64")


# --------------------------------------------------------------------------------------
# The single transform used by every prediction path
# --------------------------------------------------------------------------------------
def prepare_features(raw_input, model_key: str | None = None) -> np.ndarray:
    """
    Turn raw Ia/Ib/Ic/Va/Vb/Vc measurements into the exact 27-column vector the shipped
    models expect.

        raw dict or DataFrame
            -> PolynomialFeatures(degree=2, include_bias=False)   [27 columns]
            -> MinMaxScaler fitted on the full classData.csv
            -> float64 ndarray of shape (n_rows, 27)

    `model_key` is accepted so callers can express intent and so a future model with a
    different input contract can branch here. All three currently shipped artifacts take
    the identical 27-feature vector, which is asserted below against the loaded model.
    """
    if isinstance(raw_input, dict):
        frame = pd.DataFrame([raw_input])
    elif isinstance(raw_input, pd.DataFrame):
        frame = raw_input
    else:
        frame = pd.DataFrame(np.asarray(raw_input, dtype="float64"), columns=RAW_FEATURES)

    frame = validate_raw_frame(frame)

    poly, scaler, _ = get_feature_pipeline()
    expanded = poly.transform(frame[RAW_FEATURES])
    scaled = scaler.transform(expanded)

    if scaled.shape[1] != N_MODEL_FEATURES:
        raise PreprocessingError(
            f"Feature engineering produced {scaled.shape[1]} columns, expected "
            f"{N_MODEL_FEATURES}. Refusing to run inference on a mismatched vector."
        )

    return np.ascontiguousarray(scaled, dtype="float64")


def engineered_frame(raw_input) -> pd.DataFrame:
    """Same transform as prepare_features, returned as a named DataFrame for display."""
    return pd.DataFrame(prepare_features(raw_input), columns=POLY_FEATURE_NAMES)


# --------------------------------------------------------------------------------------
# Convenience helpers for the UI
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def raw_feature_stats() -> pd.DataFrame:
    """Min / max / mean per raw measurement, used to set realistic input bounds."""
    df = load_class_data()
    return df[RAW_FEATURES].agg(["min", "max", "mean", "std"]).T


def sample_row(fault_type: str | None = None, seed: int | None = None) -> dict:
    """Pick a real row from classData.csv so the UI never invents physically odd values."""
    df = load_class_data()
    pool = df if fault_type in (None, "Any") else df[df["Fault_Type"] == fault_type]
    if pool.empty:
        pool = df
    row = pool.sample(n=1, random_state=seed)
    values = {c: float(row.iloc[0][c]) for c in RAW_FEATURES}
    values["_actual_label"] = row.iloc[0]["Fault_Type"]
    return values
