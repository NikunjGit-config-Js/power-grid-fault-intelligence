"""Page 1 - Overview."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.constants import (
    CLASS_METADATA,
    MODEL_REGISTRY,
    N_CLASSES,
    N_MODEL_FEATURES,
    RAW_FEATURES,
)
from utils.model_loader import usable_model_keys
from utils.preprocessing import load_class_data, load_detect_data
from views.ui import AMBER, note, style_fig

PIPELINE = """Three-Phase Current &amp; Voltage  (Ia, Ib, Ic, Va, Vb, Vc)
                 &darr;
        Input Validation            finite, numeric, six channels
                 &darr;
        Feature Engineering         degree-2 polynomial expansion &rarr; 27 terms
                 &darr;
        Min-Max Scaling             scaler refit from classData.csv
                 &darr;
        ML / DL Model               Decision Tree | Random Forest | Dense MLP
                 &darr;
        Fault Classification        one of 6 encoded classes
                 &darr;
        Grid Monitoring Result      NORMAL or FAULT + fault type"""


def render() -> None:
    st.title("⚡ Power Grid Fault Intelligence")
    st.caption(
        "Fault detection and classification on transmission lines from three-phase "
        "current and voltage measurements."
    )

    data = load_class_data()
    detect = load_detect_data()
    usable = usable_model_keys()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labelled samples", f"{len(data):,}", help="Rows in classData.csv")
    c2.metric("Fault classes", N_CLASSES, help="Distinct G/C/B/A patterns present")
    c3.metric("Models available", f"{len(usable)} of {len(MODEL_REGISTRY)}")
    c4.metric("Sensor inputs", len(RAW_FEATURES), help=", ".join(RAW_FEATURES))

    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Inference pipeline")
        st.markdown(f'<div class="pg-pipe">{PIPELINE}</div>', unsafe_allow_html=True)
        note(
            f"Six raw measurements expand to <b>{N_MODEL_FEATURES}</b> engineered features. "
            "Every model in this app consumes that same 27-column vector."
        )

    with right:
        st.subheader("Class distribution")
        counts = data["Fault_Type"].value_counts()
        fig = go.Figure(
            go.Bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
                marker_color=AMBER,
                text=counts.values,
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(xaxis_title="Samples")
        st.plotly_chart(style_fig(fig, height=330), width="stretch")

    st.divider()
    st.subheader("What the six classes mean")

    rows = []
    for idx in sorted(CLASS_METADATA):
        m = CLASS_METADATA[idx]
        rows.append(
            {
                "Class": idx,
                "Label": m["label"],
                "G C B A": " ".join(m["code"]),
                "Type": m["short"],
                "Condition": "Normal" if not m["is_fault"] else "Fault",
                "Description": m["physical"],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    flagged = [CLASS_METADATA[i] for i in CLASS_METADATA if CLASS_METADATA[i]["label_note"]]
    if flagged:
        with st.expander(f"Naming caveat on {len(flagged)} classes", expanded=False):
            for m in flagged:
                st.markdown(f"**{m['label']}** (code {m['code']}) — {m['label_note']}")

    st.divider()
    cols = st.columns(3)
    cols[0].markdown(
        f"**Datasets**  \n`classData.csv` — {len(data):,} rows, 6 labelled classes  \n"
        f"`detect_dataset.csv` — {len(detect):,} rows, binary fault flag"
    )
    cols[1].markdown(
        "**Models shipped**  \n"
        + "  \n".join(
            f"{'🟢' if k in usable else '🔴'} {v['display_name']}"
            for k, v in MODEL_REGISTRY.items()
        )
    )
    cols[2].markdown(
        "**Verified accuracy**  \n"
        + "  \n".join(
            f"{v['display_name']}: `{v['test_accuracy']:.2%}`" for v in MODEL_REGISTRY.values()
        )
    )
