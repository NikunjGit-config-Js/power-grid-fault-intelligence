"""
Model loading.

The three artifacts are NOT loaded the same way, despite all three having a .pkl suffix:

  decision_trees_model.pkl   joblib -> sklearn.tree.DecisionTreeClassifier
  random_forest_model.pkl    joblib -> sklearn.ensemble.RandomForestClassifier
  DL_model.pkl               joblib -> a pickled Keras 3 Sequential model

DL_model.pkl was inspected before deciding how to deserialize it. Its pickle opcodes call
`keras.src.models.sequential.Sequential._unpickle_model` on an embedded BytesIO, and that
buffer is a full `.keras` zip archive whose metadata.json reports keras_version 3.13.2.
So it needs Keras 3.13+ importable at unpickle time -- joblib alone is not enough, and
`keras.models.load_model` on the .pkl path would not work either.

Backend choice
--------------
Keras 3 needs a backend. This app defaults to KERAS_BACKEND=numpy, which is
inference-only and pulls in jax instead of TensorFlow. That matters for deployment:
TensorFlow would add roughly 600 MB to the image and repeatedly trips the Streamlit
Community Cloud resource limit. Any backend the user has already set is respected.
"""

from __future__ import annotations

import os

# Must be set before keras is imported anywhere in the process.
os.environ.setdefault("KERAS_BACKEND", "numpy")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import streamlit as st

import json

from utils.constants import (
    MODEL_PATHS,
    MODEL_REGISTRY,
    N_MODEL_FEATURES,
    PRODUCTION_MODEL_PATH,
    PRODUCTION_METADATA_PATH,
)


class ModelLoadError(RuntimeError):
    """Raised when a model artifact cannot be loaded or fails its input-shape contract."""


def _describe_missing_backend(exc: Exception) -> str:
    return (
        "The deep-learning artifact is a pickled Keras 3 model, so Keras must be "
        "importable to deserialize it.\n\n"
        "Install it with:\n\n"
        "    pip install 'keras>=3.13' 'jax[cpu]'\n\n"
        f"Underlying error: {type(exc).__name__}: {exc}"
    )


def _load_sklearn(path):
    model = joblib.load(path)
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and n_in != N_MODEL_FEATURES:
        raise ModelLoadError(
            f"{path.name} expects {n_in} input features but the pipeline produces "
            f"{N_MODEL_FEATURES}. Refusing to load a mismatched model."
        )
    return model


def _load_keras(path):
    try:
        import keras  # noqa: F401  (import registers the unpickler Keras needs)
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ModelLoadError(_describe_missing_backend(exc)) from exc

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not deserialize {path.name}. It was saved with Keras 3.13.2; a much "
            "older Keras cannot read it.\n\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    shape = getattr(model, "input_shape", None)
    if shape is not None and len(shape) == 2 and shape[1] != N_MODEL_FEATURES:
        raise ModelLoadError(
            f"{path.name} declares input_shape={shape} but the pipeline produces "
            f"{N_MODEL_FEATURES} features. Refusing to load a mismatched model."
        )
    return model


_LOADERS = {"joblib": _load_sklearn, "keras": _load_keras}


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Load one artifact. Cached, so it is deserialized once per server process."""
    if model_key not in MODEL_REGISTRY:
        raise ModelLoadError(f"Unknown model key: {model_key!r}")

    path = MODEL_PATHS[model_key]
    if not path.exists():
        raise ModelLoadError(
            f"Model file not found: {path.name}\n\n"
            "It should sit in the repository root next to app.py. If the repository was "
            "cloned without Git LFS, re-download the artifact."
        )

    loader = _LOADERS[MODEL_REGISTRY[model_key]["loader"]]
    return loader(path)


@st.cache_data(show_spinner=False)
def model_availability() -> dict:
    """
    Probe every registered model once and report which are actually usable.

    Returns {model_key: {"ok": bool, "error": str | None}}. The UI uses this to grey out
    models it cannot run instead of failing mid-prediction. Nothing here ever substitutes
    one model for another.
    """
    status = {}
    for key in MODEL_REGISTRY:
        try:
            load_model(key)
            status[key] = {"ok": True, "error": None}
        except Exception as exc:
            status[key] = {"ok": False, "error": str(exc)}
    return status


def usable_model_keys() -> list[str]:
    return [k for k, v in model_availability().items() if v["ok"]]


def model_runtime_facts(model_key: str) -> dict:
    """Read facts off the loaded object itself rather than trusting the registry."""
    model = load_model(model_key)
    facts = {
        "python_type": f"{type(model).__module__}.{type(model).__name__}",
        "n_features_in": getattr(model, "n_features_in_", None),
        "feature_names_in": getattr(model, "feature_names_in_", None),
        "classes": getattr(model, "classes_", None),
        "supports_proba": hasattr(model, "predict_proba"),
        "input_shape": getattr(model, "input_shape", None),
        "output_shape": getattr(model, "output_shape", None),
        "has_feature_importances": hasattr(model, "feature_importances_"),
    }
    if facts["input_shape"] is not None and facts["n_features_in"] is None:
        facts["n_features_in"] = facts["input_shape"][-1]
    if facts["output_shape"] is not None:
        facts["supports_proba"] = True
    return facts


# --------------------------------------------------------------------------------------
# Production pipeline (redesigned LightGBM with D1 physics-informed features)
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_production_pipeline():
    """Load the production LightGBM pipeline from models/best_model_pipeline.joblib."""
    if not PRODUCTION_MODEL_PATH.exists():
        raise ModelLoadError(
            f"Production pipeline not found: {PRODUCTION_MODEL_PATH.name}\n\n"
            "Run the Final Retraining phase to generate this artifact."
        )
    return joblib.load(PRODUCTION_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_production_metadata() -> dict:
    """Load production metadata from models/model_metadata.json."""
    if not PRODUCTION_METADATA_PATH.exists():
        raise ModelLoadError(
            f"Production metadata not found: {PRODUCTION_METADATA_PATH.name}\n\n"
            "Run the Final Retraining phase to generate this artifact."
        )
    with open(PRODUCTION_METADATA_PATH, "r") as f:
        return json.load(f)
