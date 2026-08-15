"""
Inference layer.

One entry point, `predict`, serves the Live page and the Batch page so the two can never
drift apart. Probabilities are only ever reported when the underlying model genuinely
produces them -- there is no fallback that manufactures a confidence number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.constants import (
    CLASS_INDEX_TO_LABEL,
    CLASS_LABELS,
    CLASS_METADATA,
    MODEL_REGISTRY,
    N_CLASSES,
    N_MODEL_FEATURES,
    RAW_FEATURES,
)
from utils.model_loader import load_model, load_production_pipeline, load_production_metadata
from utils.preprocessing import prepare_features, validate_raw_frame


class PredictionError(RuntimeError):
    """Raised when inference cannot be completed for the selected model."""


@dataclass
class PredictionResult:
    """Result of a batch of predictions (a single prediction is a batch of one)."""

    model_key: str
    class_indices: np.ndarray
    labels: list[str]
    probabilities: np.ndarray | None = None
    features: np.ndarray | None = field(default=None, repr=False)

    @property
    def model_name(self) -> str:
        return MODEL_REGISTRY[self.model_key]["display_name"]

    @property
    def has_probabilities(self) -> bool:
        return self.probabilities is not None

    @property
    def confidences(self) -> np.ndarray | None:
        if self.probabilities is None:
            return None
        return self.probabilities.max(axis=1)

    def meta(self, i: int = 0) -> dict:
        return CLASS_METADATA[int(self.class_indices[i])]

    def is_fault(self, i: int = 0) -> bool:
        return bool(self.meta(i)["is_fault"])

    def probability_frame(self, i: int = 0) -> pd.DataFrame | None:
        if self.probabilities is None:
            return None
        return pd.DataFrame(
            {
                "Fault class": CLASS_LABELS,
                "Probability": self.probabilities[i],
            }
        ).sort_values("Probability", ascending=False, ignore_index=True)


def _raw_scores(model, features: np.ndarray, model_key: str):
    """
    Return (class_indices, probabilities|None) for either model family.

    scikit-learn : predict() plus predict_proba() when the estimator exposes it.
    Keras        : the softmax output IS the probability vector, so argmax gives the class.
    """
    if MODEL_REGISTRY[model_key]["loader"] == "keras":
        proba = np.asarray(model.predict(features, verbose=0), dtype="float64")
        if proba.ndim != 2 or proba.shape[1] != N_CLASSES:
            raise PredictionError(
                f"The deep-learning model returned shape {proba.shape}, expected "
                f"(n_rows, {N_CLASSES})."
            )
        return proba.argmax(axis=1), proba

    indices = np.asarray(model.predict(features)).astype(int)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            candidate = np.asarray(model.predict_proba(features), dtype="float64")
            # Align columns to canonical class order via the estimator's own classes_.
            if candidate.shape[1] == N_CLASSES:
                classes = np.asarray(getattr(model, "classes_", np.arange(N_CLASSES))).astype(int)
                aligned = np.zeros_like(candidate)
                for col, cls in enumerate(classes):
                    aligned[:, cls] = candidate[:, col]
                proba = aligned
        except Exception:
            # A model that advertises predict_proba but fails is reported as
            # "no probabilities" rather than given invented ones.
            proba = None

    return indices, proba


def predict(model_key: str, raw_input) -> PredictionResult:
    """
    Run the full pipeline for one or many rows.

        raw Ia/Ib/Ic/Va/Vb/Vc
          -> prepare_features (polynomial expansion + min-max scaling)
          -> shape assertion against the loaded model
          -> model.predict
          -> class index decoded to a human-readable fault name
    """
    if model_key not in MODEL_REGISTRY:
        raise PredictionError(f"Unknown model: {model_key!r}")

    features = prepare_features(raw_input, model_key)
    model = load_model(model_key)

    # Contract check against the artifact itself, not against a constant.
    expected = getattr(model, "n_features_in_", None)
    if expected is None:
        shape = getattr(model, "input_shape", None)
        expected = shape[-1] if shape else None
    if expected is not None and features.shape[1] != expected:
        raise PredictionError(
            f"Feature vector has {features.shape[1]} columns but "
            f"{MODEL_REGISTRY[model_key]['display_name']} expects {expected}."
        )
    if features.shape[1] != N_MODEL_FEATURES:
        raise PredictionError(
            f"Feature vector has {features.shape[1]} columns, expected {N_MODEL_FEATURES}."
        )

    indices, proba = _raw_scores(model, features, model_key)

    unknown = sorted(set(indices) - set(CLASS_INDEX_TO_LABEL))
    if unknown:
        raise PredictionError(f"Model returned class index/indices outside 0-5: {unknown}")

    return PredictionResult(
        model_key=model_key,
        class_indices=indices,
        labels=[CLASS_INDEX_TO_LABEL[int(i)] for i in indices],
        probabilities=proba,
        features=features,
    )


def predict_frame(model_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Batch helper: returns the validated input with prediction columns appended."""
    from utils.preprocessing import validate_raw_frame

    clean = validate_raw_frame(df)
    result = predict(model_key, clean)

    out = clean.copy()
    out["Predicted_Class"] = result.class_indices
    out["Predicted_Fault"] = result.labels
    out["Fault_Code_GCBA"] = [CLASS_METADATA[int(i)]["code"] for i in result.class_indices]
    out["Status"] = ["FAULT" if CLASS_METADATA[int(i)]["is_fault"] else "NORMAL"
                     for i in result.class_indices]
    if result.has_probabilities:
        out["Confidence"] = result.confidences
    out.attrs["n_dropped"] = clean.attrs.get("n_dropped", 0)
    out.attrs["dropped_index"] = clean.attrs.get("dropped_index", [])
    out.attrs["has_probabilities"] = result.has_probabilities
    return out


def feature_importance_frame(model_key: str, top_n: int = 12) -> pd.DataFrame | None:
    """
    Top-N feature importances with VERIFIED names.

    Names come from POLY_FEATURE_NAMES, which is asserted against the live
    PolynomialFeatures transformer at pipeline construction, so index i genuinely is
    the i-th column the model was trained on.
    """
    from utils.constants import POLY_FEATURE_NAMES

    model = load_model(model_key)
    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.asarray(model.feature_importances_, dtype="float64")
    if len(importances) != len(POLY_FEATURE_NAMES):
        raise PredictionError(
            f"Model reports {len(importances)} importances but the pipeline defines "
            f"{len(POLY_FEATURE_NAMES)} feature names. Refusing to mislabel them."
        )

    frame = pd.DataFrame({"Feature": POLY_FEATURE_NAMES, "Importance": importances})
    return frame.sort_values("Importance", ascending=False, ignore_index=True).head(top_n)


# --------------------------------------------------------------------------------------
# Production pipeline inference (LightGBM with D1 physics-informed features)
# --------------------------------------------------------------------------------------

@dataclass
class ProductionPredictionResult:
    """Result of a production pipeline prediction."""

    model_name: str
    class_indices: np.ndarray
    labels: list[str]
    probabilities: np.ndarray | None = None

    @property
    def has_probabilities(self) -> bool:
        return self.probabilities is not None

    @property
    def confidences(self) -> np.ndarray | None:
        if self.probabilities is None:
            return None
        return self.probabilities.max(axis=1)


def predict_lightgbm(raw_input) -> ProductionPredictionResult:
    """
    Run inference using the production LightGBM pipeline.

    Accepts raw 6-feature input, validates order/count, passes RAW values
    directly to the saved Pipeline, calls predict()/predict_proba(),
    decodes using model_metadata.json.

    Never duplicates the D1 feature engineering outside the saved Pipeline.
    """
    pipeline = load_production_pipeline()
    metadata = load_production_metadata()

    # Convert raw input to DataFrame
    if isinstance(raw_input, dict):
        frame = pd.DataFrame([raw_input])
    elif isinstance(raw_input, pd.DataFrame):
        frame = raw_input.copy()
    else:
        frame = pd.DataFrame(np.asarray(raw_input, dtype="float64"), columns=RAW_FEATURES)

    # Validate raw inputs (order and count)
    frame = validate_raw_frame(frame)

    # Pipeline handles feature transformation internally
    features = frame[RAW_FEATURES].values

    # Predict
    indices = np.asarray(pipeline.predict(features)).astype(int)
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = np.asarray(pipeline.predict_proba(features), dtype="float64")
        except Exception:
            proba = None

    # Decode using metadata["idx_to_class"]
    idx_to_class = metadata["idx_to_class"]
    labels = [idx_to_class[str(int(i))] for i in indices]

    return ProductionPredictionResult(
        model_name=metadata["model_name"],
        class_indices=indices,
        labels=labels,
        probabilities=proba,
    )
