"""
Power Grid Fault Intelligence - Single-Page Inference Demo

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd

from utils.constants import RAW_FEATURES
from utils.model_loader import load_production_pipeline, load_production_metadata
from utils.prediction import predict_lightgbm

st.set_page_config(
    page_title="Power Grid Fault Intelligence",
    page_icon="⚡",
    layout="centered",
)

# --- Load metadata for sidebar info ---
try:
    metadata = load_production_metadata()
except Exception as exc:
    st.error(f"Could not load production metadata.\n\n{exc}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚡ Power Grid Fault Intelligence")
    st.caption("Transmission-line fault detection and classification")
    st.divider()
    st.markdown("##### Model")
    st.caption(f"**{metadata['model_name']}**")
    st.caption(f"Feature set: {metadata['feature_set']}")
    st.divider()
    st.caption("6 raw measurements -> 20 D1 features -> LightGBM")

# --- Main page ---
st.title("Power Grid Fault Intelligence")
st.caption("Enter three-phase measurements and classify the grid condition.")

# --- Demo presets (clearly labelled as demo inputs) ---
st.markdown("##### Demo presets")
preset_cols = st.columns(3)
presets = {
    "Normal": {"Ia": 100.0, "Ib": 100.0, "Ic": 100.0, "Va": 230.0, "Vb": 230.0, "Vc": 230.0},
    "Fault A": {"Ia": 150.0, "Ib": 80.0, "Ic": 70.0, "Va": 200.0, "Vb": 240.0, "Vc": 245.0},
    "Fault B": {"Ia": 90.0, "Ib": 160.0, "Ic": 60.0, "Va": 235.0, "Vb": 195.0, "Vc": 250.0},
}
with preset_cols[0]:
    if st.button("Normal operation"):
        st.session_state["pg_inputs"] = presets["Normal"].copy()
with preset_cols[1]:
    if st.button("Fault example A"):
        st.session_state["pg_inputs"] = presets["Fault A"].copy()
with preset_cols[2]:
    if st.button("Fault example B"):
        st.session_state["pg_inputs"] = presets["Fault B"].copy()

st.caption("*Demo presets are illustrative inputs, not evaluation examples.*")

# --- Seed defaults ---
if "pg_inputs" not in st.session_state:
    st.session_state["pg_inputs"] = presets["Normal"].copy()

values = st.session_state["pg_inputs"]

# --- Six raw inputs ---
st.markdown("##### Measurements")
cur_col, volt_col = st.columns(2, gap="large")
with cur_col:
    st.markdown("**Current** (amperes)")
    for f in ["Ia", "Ib", "Ic"]:
        values[f] = st.number_input(f, value=float(values[f]), step=1.0, format="%.4f", key=f"in_{f}")
with volt_col:
    st.markdown("**Voltage** (per unit)")
    for f in ["Va", "Vb", "Vc"]:
        values[f] = st.number_input(f, value=float(values[f]), step=0.01, format="%.5f", key=f"in_{f}")

st.session_state["pg_inputs"] = values

# --- Predict button ---
predict_clicked = st.button("Predict", type="primary", width="stretch")

if not predict_clicked:
    st.info("Set the six measurements above, then select **Predict**.")
    st.stop()

# --- Inference ---
try:
    with st.spinner("Running inference..."):
        result = predict_lightgbm(values)
except Exception as exc:
    st.error(f"Selected model could not be loaded.\n\n{type(exc).__name__}: {exc}")
    st.stop()

# --- Result ---
pred_idx = int(result.class_indices[0])
fault_name = result.labels[0]
is_fault = fault_name != "No Fault"
confidence = float(result.probabilities[0][pred_idx]) if result.has_probabilities else None

if is_fault:
    st.error("**FAULT DETECTED**")
else:
    st.success("**NORMAL OPERATION**")

r1, r2 = st.columns(2)
r1.metric("Predicted fault" if is_fault else "Predicted condition", fault_name)
if confidence is not None:
    r2.metric("Model probability", f"{confidence:.2%}")

# --- Probability distribution ---
if result.has_probabilities:
    st.subheader("Probability distribution")
    idx_to_class = metadata["idx_to_class"]
    prob_df = pd.DataFrame({
        "Class": [idx_to_class[str(i)] for i in range(6)],
        "Probability": result.probabilities[0],
    }).sort_values("Probability", ascending=False)
    st.bar_chart(prob_df.set_index("Class"))
    st.caption("*Note: Probabilities are model outputs and are not calibrated confidence estimates.*")
else:
    st.info("Probability scores are not available for this model.")
