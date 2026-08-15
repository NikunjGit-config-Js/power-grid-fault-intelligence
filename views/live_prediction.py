"""Page 2 - Live Fault Prediction. The main demonstration surface."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.constants import (
    CURRENT_FEATURES,
    MODEL_REGISTRY,
    POLY_FEATURE_NAMES,
    RAW_FEATURES,
    VOLTAGE_FEATURES,
)
from utils.preprocessing import engineered_frame, load_class_data, raw_feature_stats, sample_row
from utils.prediction import PredictionError, predict
from utils.preprocessing import PreprocessingError
from views.ui import AMBER, GREEN, MUTED, RED, note, phase_bar, status_banner, style_fig

_DEFAULT_KEY = "pg_inputs"


def _seed_defaults() -> None:
    """Seed the six inputs from a real dataset row rather than invented numbers."""
    if _DEFAULT_KEY in st.session_state:
        return
    row = sample_row("No Fault", seed=42)
    row.pop("_actual_label", None)
    st.session_state[_DEFAULT_KEY] = row
    st.session_state["pg_loaded_label"] = None


def _load_example(fault_type: str) -> None:
    row = sample_row(fault_type)
    st.session_state["pg_loaded_label"] = row.pop("_actual_label")
    st.session_state[_DEFAULT_KEY] = row


def _probability_chart(frame) -> go.Figure:
    colors = [AMBER if i == 0 else "rgba(139,148,158,.55)" for i in range(len(frame))]
    fig = go.Figure(
        go.Bar(
            x=frame["Probability"],
            y=frame["Fault class"],
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in frame["Probability"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(title="Probability by class", xaxis_title="Probability")
    fig.update_xaxes(range=[0, 1.14], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=310)


def render(model_key: str) -> None:
    st.title("Live Fault Prediction")
    st.caption(
        f"Enter three-phase measurements and classify the grid condition using "
        f"**{MODEL_REGISTRY[model_key]['display_name']}**."
    )

    _seed_defaults()
    stats = raw_feature_stats()
    values = st.session_state[_DEFAULT_KEY]

    # ---------------------------------------------------------------- input panel
    with st.container(border=True):
        top = st.columns([2, 1.1, 1])
        with top[0]:
            st.markdown("##### Load an example from the dataset")
        with top[1]:
            options = ["Any"] + sorted(load_class_data()["Fault_Type"].unique())
            choice = st.selectbox(
                "Condition", options, label_visibility="collapsed", key="pg_example_choice"
            )
        with top[2]:
            st.button(
                "Load sample",
                width="stretch",
                on_click=_load_example,
                args=(choice,),
            )

        loaded = st.session_state.get("pg_loaded_label")
        if loaded:
            st.caption(f"Loaded a real row labelled **{loaded}** — see if the model agrees.")

        st.markdown("")
        cur_col, volt_col = st.columns(2, gap="large")

        with cur_col:
            st.markdown("**Current measurements** (amperes)")
            for f in CURRENT_FEATURES:
                values[f] = st.number_input(
                    f,
                    value=float(values[f]),
                    step=1.0,
                    format="%.4f",
                    key=f"in_{f}",
                    help=f"Dataset range {stats.loc[f,'min']:,.1f} to {stats.loc[f,'max']:,.1f} A",
                )

        with volt_col:
            st.markdown("**Voltage measurements** (per unit)")
            for f in VOLTAGE_FEATURES:
                values[f] = st.number_input(
                    f,
                    value=float(values[f]),
                    step=0.01,
                    format="%.5f",
                    key=f"in_{f}",
                    help=f"Dataset range {stats.loc[f,'min']:,.3f} to {stats.loc[f,'max']:,.3f} pu",
                )

        analyse = st.button("Analyze Grid Condition", type="primary", width="stretch")

    st.session_state[_DEFAULT_KEY] = values

    # ---------------------------------------------------------------- measurements
    st.subheader("Measured waveform snapshot")
    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.plotly_chart(
            phase_bar(CURRENT_FEATURES, [values[f] for f in CURRENT_FEATURES],
                      "Three-phase current", "Amperes"),
            width="stretch",
        )
    with m2:
        st.plotly_chart(
            phase_bar(VOLTAGE_FEATURES, [values[f] for f in VOLTAGE_FEATURES],
                      "Three-phase voltage", "Per unit"),
            width="stretch",
        )

    residual = sum(values[f] for f in CURRENT_FEATURES)
    note(
        f"Residual (zero-sequence) current <b>Ia + Ib + Ic = {residual:,.3f} A</b>. "
        "A large residual normally indicates a ground path; a balanced healthy line sums "
        "close to zero."
    )

    if not analyse:
        st.info("Set the six measurements above, then select **Analyze Grid Condition**.")
        return

    # ---------------------------------------------------------------- inference
    st.divider()
    payload = {f: values[f] for f in RAW_FEATURES}

    try:
        with st.spinner("Running inference…"):
            result = predict(model_key, payload)
    except (PredictionError, PreprocessingError) as exc:
        st.error(f"Prediction failed.\n\n{exc}")
        return
    except Exception as exc:  # surfaced cleanly rather than as a raw traceback
        st.error(f"Unexpected error while running {MODEL_REGISTRY[model_key]['display_name']}.\n\n"
                 f"{type(exc).__name__}: {exc}")
        return

    meta = result.meta()
    fault = result.is_fault()

    status_banner(
        kicker="Grid status",
        headline="⚠️  FAULT DETECTED" if fault else "✅  NORMAL OPERATION",
        detail=(
            f"{meta['physical']}"
            if fault
            else "No transmission-line fault detected in these measurements."
        ),
        tone="fault" if fault else "normal",
    )

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted fault" if fault else "Predicted condition", meta["label"])
    r2.metric("Encoded class / G C B A", f"{int(result.class_indices[0])}  ·  {meta['code']}")
    r3.metric("Model", result.model_name)

    if meta["label_note"]:
        st.warning(f"**Label caveat.** {meta['label_note']}", icon="⚠️")

    if st.session_state.get("pg_loaded_label"):
        truth = st.session_state["pg_loaded_label"]
        if truth == meta["label"]:
            st.success(f"Matches the dataset label for the loaded sample: **{truth}**.")
        else:
            st.error(f"Dataset label for the loaded sample is **{truth}** — this is a miss.")

    # ---------------------------------------------------------------- confidence
    st.subheader("Prediction confidence")
    if result.has_probabilities:
        frame = result.probability_frame()
        conf = float(result.confidences[0])
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.metric("Top-class confidence", f"{conf:.2%}")
            st.progress(min(max(conf, 0.0), 1.0))
            runner = frame.iloc[1]
            st.caption(f"Next most likely: **{runner['Fault class']}** at {runner['Probability']:.2%}")
            if MODEL_REGISTRY[model_key]["loader"] == "joblib" and conf >= 0.999:
                st.caption(
                    "Tree models often return a hard 1.00 — treat this as a vote share, "
                    "not a calibrated probability."
                )
        with c2:
            st.plotly_chart(_probability_chart(frame), width="stretch")
    else:
        st.info("Probability scores are not available for this model.")

    # ---------------------------------------------------------------- transparency
    with st.expander("Engineered feature vector actually sent to the model"):
        st.caption(
            f"{len(POLY_FEATURE_NAMES)} min-max scaled features, in the exact order the "
            "model was trained on."
        )
        st.dataframe(
            engineered_frame(payload).T.rename(columns={0: "Scaled value"}),
            width="stretch",
            height=320,
        )
