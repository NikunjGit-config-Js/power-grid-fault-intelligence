"""
Feature engineering transformers for the power-grid fault classification experiment.

Implements four feature sets for objective comparison:
  A. Raw six features (6)
  B. Polynomial baseline (27)
  C. Physics-informed core (17)
  D1. Physics core + current symmetry (20)
  D2. Physics core + current + voltage symmetry (23)

Every transformer is scikit-learn compatible (fit/transform/fit_transform/get_feature_names_out)
so it can live inside a Pipeline.

Mathematical formulas are verified against power-systems literature.
Guarded impedance uses a fixed epsilon never derived from data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Fixed epsilon for guarded impedance division.
# This is an implementation constant, never derived from validation or test data.
EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Utility: guarded division
# ---------------------------------------------------------------------------

def safe_divide(V: np.ndarray, I: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Numerically guarded division V/I.

    Preserves sign and prevents:
      - division by zero
      - inf / -inf
      - NaN

    Parameters
    ----------
    V : voltage array
    I : current array
    epsilon : fixed small constant (1e-10), never derived from data

    Returns
    -------
    impedance array, clipped to [-1e10, 1e10]
    """
    abs_I = np.abs(I)
    safe_I = np.where(abs_I > epsilon, I, np.sign(I) * epsilon)
    result = V / safe_I
    return np.clip(result, -1e10, 1e10)


# ---------------------------------------------------------------------------
# Feature Set A: Raw Features (6)
# ---------------------------------------------------------------------------

class RawFeatures(BaseEstimator, TransformerMixin):
    """Passthrough transformer for the 6 raw measurements."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return X_arr[:, :6]

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES)


# ---------------------------------------------------------------------------
# Feature Set B: Polynomial Baseline (27)
# ---------------------------------------------------------------------------

class PolynomialBaseline(BaseEstimator, TransformerMixin):
    """PolynomialFeatures(degree=2, include_bias=False) wrapped as a transformer.

    Produces 27 features: 6 raw + 6 squares + 15 pairwise interactions.
    """

    def __init__(self):
        self._poly = PolynomialFeatures(degree=2, include_bias=False)
        self._feature_names = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        self._poly.fit(X_arr[:, :6])
        self._feature_names = list(self._poly.get_feature_names_out(RAW_FEATURES))
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self._poly.transform(X_arr[:, :6])

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names)


# ---------------------------------------------------------------------------
# Feature Set C: Physics-Informed Core (17)
# ---------------------------------------------------------------------------

class PhysicsCore(BaseEstimator, TransformerMixin):
    """Physics-informed features: Clarke transforms + guarded impedance.

    Produces 17 features:
      6 raw (Ia, Ib, Ic, Va, Vb, Vc)
      + 4 current Clarke (I0, Ialpha, Ibeta, Imag)
      + 4 voltage Clarke (V0, Valpha, Vbeta, Vmag)
      + 3 guarded impedance (Za, Zb, Zc)
    """

    FEATURE_NAMES = [
        "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
        "I0", "Ialpha", "Ibeta", "Imag",
        "V0", "Valpha", "Vbeta", "Vmag",
        "Za", "Zb", "Zc",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        Ia, Ib, Ic = X_arr[:, 0], X_arr[:, 1], X_arr[:, 2]
        Va, Vb, Vc = X_arr[:, 3], X_arr[:, 4], X_arr[:, 5]

        # Clarke Transform - Current
        I0 = (Ia + Ib + Ic) / 3.0
        Ialpha = (2.0 * Ia - Ib - Ic) / 3.0
        Ibeta = (Ib - Ic) / np.sqrt(3.0)
        Imag = np.sqrt(Ialpha**2 + Ibeta**2)

        # Clarke Transform - Voltage
        V0 = (Va + Vb + Vc) / 3.0
        Valpha = (2.0 * Va - Vb - Vc) / 3.0
        Vbeta = (Vb - Vc) / np.sqrt(3.0)
        Vmag = np.sqrt(Valpha**2 + Vbeta**2)

        # Guarded Per-Phase Impedance
        Za = safe_divide(Va, Ia)
        Zb = safe_divide(Vb, Ib)
        Zc = safe_divide(Vc, Ic)

        result = np.column_stack([
            Ia, Ib, Ic, Va, Vb, Vc,
            I0, Ialpha, Ibeta, Imag,
            V0, Valpha, Vbeta, Vmag,
            Za, Zb, Zc,
        ])
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Feature Set D1: Physics Core + Current Symmetry (20)
# ---------------------------------------------------------------------------

class PhysicsWithCurrentSymmetry(BaseEstimator, TransformerMixin):
    """Physics core + sorted absolute currents.

    Produces 20 features:
      Set C (17) + I_max, I_mid, I_min
    """

    FEATURE_NAMES = [
        "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
        "I0", "Ialpha", "Ibeta", "Imag",
        "V0", "Valpha", "Vbeta", "Vmag",
        "Za", "Zb", "Zc",
        "I_max", "I_mid", "I_min",
    ]

    def __init__(self):
        self._core = PhysicsCore()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        core = self._core.transform(X_arr)

        Ia, Ib, Ic = X_arr[:, 0], X_arr[:, 1], X_arr[:, 2]
        abs_currents = np.abs(np.column_stack([Ia, Ib, Ic]))
        sorted_currents = np.sort(abs_currents, axis=1)  # ascending
        I_min = sorted_currents[:, 0]
        I_mid = sorted_currents[:, 1]
        I_max = sorted_currents[:, 2]

        result = np.column_stack([core, I_max, I_mid, I_min])
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Feature Set D2: Physics Core + Current + Voltage Symmetry (23)
# ---------------------------------------------------------------------------

class PhysicsWithFullSymmetry(BaseEstimator, TransformerMixin):
    """Physics core + sorted absolute currents + sorted absolute voltages.

    Produces 23 features:
      Set C (17) + I_max, I_mid, I_min + V_max, V_mid, V_min
    """

    FEATURE_NAMES = [
        "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
        "I0", "Ialpha", "Ibeta", "Imag",
        "V0", "Valpha", "Vbeta", "Vmag",
        "Za", "Zb", "Zc",
        "I_max", "I_mid", "I_min",
        "V_max", "V_mid", "V_min",
    ]

    def __init__(self):
        self._core = PhysicsCore()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        core = self._core.transform(X_arr)

        Ia, Ib, Ic = X_arr[:, 0], X_arr[:, 1], X_arr[:, 2]
        Va, Vb, Vc = X_arr[:, 3], X_arr[:, 4], X_arr[:, 5]

        abs_currents = np.abs(np.column_stack([Ia, Ib, Ic]))
        sorted_currents = np.sort(abs_currents, axis=1)
        I_min = sorted_currents[:, 0]
        I_mid = sorted_currents[:, 1]
        I_max = sorted_currents[:, 2]

        abs_voltages = np.abs(np.column_stack([Va, Vb, Vc]))
        sorted_voltages = np.sort(abs_voltages, axis=1)
        V_min = sorted_voltages[:, 0]
        V_mid = sorted_voltages[:, 1]
        V_max = sorted_voltages[:, 2]

        result = np.column_stack([core, I_max, I_mid, I_min, V_max, V_mid, V_min])
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Registry for easy access
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "A_raw": RawFeatures,
    "B_polynomial": PolynomialBaseline,
    "C_physics_core": PhysicsCore,
    "D1_physics_current_sym": PhysicsWithCurrentSymmetry,
    "D2_physics_full_sym": PhysicsWithFullSymmetry,
}

FEATURE_SET_COUNTS = {
    "A_raw": 6,
    "B_polynomial": 27,
    "C_physics_core": 17,
    "D1_physics_current_sym": 20,
    "D2_physics_full_sym": 23,
}
