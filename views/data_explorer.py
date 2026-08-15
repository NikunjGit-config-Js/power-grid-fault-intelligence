"""Page 4 - Data Explorer."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.constants import CURRENT_FEATURES, RAW_FEATURES, VOLTAGE_FEATURES
from utils.preprocessing import load_class_data, load_detect_data
from views.ui import AMBER, GREEN, PHASE_COLORS, note, style_fig


@st.cache_data(show_spinner=False)
def _corr(dataset: str) -> np.ndarray:
    df = load_class_data() if dataset == "classData.csv" else load_detect_data()
    return df[RAW_FEATURES].corr().to_numpy()


def _dist_figure(df, columns, title, unit) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                opacity=0.62,
                nbinsx=60,
                marker_color=PHASE_COLORS[col[-1].lower()],
            )
        )
    fig.update_layout(barmode="overlay", title=title, xaxis_title=unit, yaxis_title="Count")
    return style_fig(fig, height=330, showlegend=True)


def render() -> None:
    st.title("Data Explorer")
    st.caption("Exploratory analysis of the two datasets shipped with the repository.")

    dataset = st.radio(
        "Dataset",
        ["classData.csv", "detect_dataset.csv"],
        horizontal=True,
        captions=[
            "Labelled — six fault classes from G/C/B/A",
            "Binary — Output (S) fault flag",
        ],
    )

    df = load_class_data() if dataset == "classData.csv" else load_detect_data()
    label_col = "Fault_Type" if dataset == "classData.csv" else "Output (S)"

    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", df.shape[1])
    c.metric("Missing values", int(df.isna().sum().sum()))
    d.metric("Duplicate rows", int(df.duplicated().sum()))

    if dataset == "detect_dataset.csv":
        note(
            "The raw header of this file ends with two empty fields, which pandas reads as "
            "<code>Unnamed: 7</code> and <code>Unnamed: 8</code>. Both are entirely null and "
            "are dropped on load."
        )

    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Class distribution")
        counts = df[label_col].value_counts()
        labels = [str(i) for i in counts.index]
        if dataset == "detect_dataset.csv":
            labels = ["No fault (0)" if str(i) == "0" else "Fault (1)" for i in counts.index]
        fig = go.Figure(
            go.Bar(
                x=counts.values,
                y=labels,
                orientation="h",
                marker_color=[GREEN if "No" in str(l) else AMBER for l in labels],
                text=[f"{v:,}" for v in counts.values],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(xaxis_title="Samples")
        st.plotly_chart(style_fig(fig, height=300), width="stretch")

    with right:
        st.subheader("Correlation — raw measurements")
        matrix = _corr(dataset)
        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=RAW_FEATURES,
                y=RAW_FEATURES,
                colorscale="RdBu",
                zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in matrix],
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(thickness=12),
            )
        )
        st.plotly_chart(style_fig(fig, height=300), width="stretch")

    st.divider()
    st.subheader("Measurement distributions")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(
            _dist_figure(df, CURRENT_FEATURES, "Three-phase current", "Amperes"),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            _dist_figure(df, VOLTAGE_FEATURES, "Three-phase voltage", "Per unit"),
            width="stretch",
        )

    st.divider()
    st.subheader("Descriptive statistics")
    st.dataframe(df[RAW_FEATURES].describe().T, width="stretch")

    st.subheader("Raw data preview")
    n = st.slider("Rows to show", 5, 200, 25, step=5)
    cols = st.multiselect("Columns", list(df.columns), default=list(df.columns))
    if cols:
        st.dataframe(df[cols].head(n), width="stretch")
    else:
        st.info("Select at least one column to preview.")
