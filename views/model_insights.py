"""Page 5 - Model Insights."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.constants import (
    MODEL_REGISTRY,
    NOTEBOOK_ONLY_RESULTS,
    POLY_FEATURE_NAMES,
    TEST_SPLIT,
)
from utils.model_loader import model_availability, model_runtime_facts
from utils.prediction import feature_importance_frame
from views.ui import AMBER, BLUE, GREEN, RED, note, style_fig


def _accuracy_chart() -> go.Figure:
    names = [m["display_name"] for m in MODEL_REGISTRY.values()]
    accs = [m["test_accuracy"] for m in MODEL_REGISTRY.values()]
    colors = [GREEN if a >= 0.85 else RED for a in accs]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=accs,
            marker_color=colors,
            text=[f"{a:.2%}" for a in accs],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(title="Verified test accuracy — shipped artifacts", yaxis_title="Accuracy")
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
    return style_fig(fig, height=330)


def _importance_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=frame["Importance"],
            y=frame["Feature"],
            orientation="h",
            marker_color=AMBER,
            text=[f"{v:.3f}" for v in frame["Importance"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(title="Top feature importances", xaxis_title="Relative importance")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=430)


def render(model_key: str) -> None:
    st.title("Model Insights")
    st.caption(
        "Every number on this page was recomputed against the shipped artifacts by "
        "replaying the notebook's train/test split."
    )

    note(
        f"Split reproduced: <code>train_test_split(test_size={TEST_SPLIT['test_size']}, "
        f"random_state={TEST_SPLIT['random_state']}, stratify=y)</code> → "
        f"{TEST_SPLIT['n_train']:,} train / {TEST_SPLIT['n_test']:,} test rows. "
        "These are test metrics only; training metrics are not mixed in."
    )

    st.plotly_chart(_accuracy_chart(), width="stretch")

    st.divider()
    st.subheader("Selected model")

    info = MODEL_REGISTRY[model_key]
    status = model_availability()[model_key]

    if not status["ok"]:
        st.error(f"**{info['display_name']}** could not be loaded.\n\n{status['error']}")
        return

    facts = model_runtime_facts(model_key)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", info["display_name"])
    c2.metric("Input features", facts["n_features_in"] or "—")
    c3.metric("Test accuracy", f"{info['test_accuracy']:.2%}")
    c4.metric("Probabilities", "Yes" if facts["supports_proba"] else "No")

    detail = {
        "Python type": facts["python_type"],
        "Serialization": f"{info['loader']} → {info['family']}",
        "feature_names_in_": (
            "not set (trained on a NumPy array, so names live in the pipeline)"
            if facts["feature_names_in"] is None
            else ", ".join(map(str, facts["feature_names_in"]))
        ),
        "classes_": (
            "not applicable (Keras softmax output)"
            if facts["classes"] is None
            else str(list(facts["classes"]))
        ),
        "Output shape": str(facts["output_shape"]) if facts["output_shape"] else "—",
        "Trained on": info["trained_on"],
    }
    if info["cv_accuracy"] is not None:
        detail["5-fold CV accuracy (notebook)"] = f"{info['cv_accuracy']:.4f}"

    st.dataframe(
        pd.DataFrame({"Property": list(detail), "Value": list(detail.values())}),
        width="stretch",
        hide_index=True,
    )

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("**Strengths**")
        for item in info["strengths"]:
            st.markdown(f"- {item}")
    with s2:
        st.markdown("**Limitations**")
        for item in info["limitations"]:
            st.markdown(f"- {item}")

    if model_key == "deep_learning":
        st.warning(
            "**This model underperforms badly.** Verified test accuracy is 38.53%, which "
            "reproduces the notebook's own printed result exactly. It is kept in the app "
            "for an honest comparison rather than removed to make the project look better.",
            icon="⚠️",
        )
        st.info(
            "The README describes a hybrid CNN-LSTM. The saved artifact is a Dense MLP "
            "(27 → 128 → 64 → 32 → 6, BatchNorm + Dropout 0.5). The architecture reported "
            "here comes from the artifact itself, not the README.",
            icon="ℹ️",
        )

    st.divider()
    st.subheader("Feature importance")

    if info["has_feature_importances"]:
        top_n = st.slider("Features to show", 5, 27, 12)
        frame = feature_importance_frame(model_key, top_n=top_n)
        col_a, col_b = st.columns([1.5, 1], gap="large")
        with col_a:
            st.plotly_chart(_importance_chart(frame), width="stretch")
        with col_b:
            st.dataframe(frame, width="stretch", hide_index=True, height=430)
        note(
            f"Names come from the {len(POLY_FEATURE_NAMES)}-term polynomial expansion, which "
            "is asserted against the live transformer at startup — index <i>i</i> genuinely "
            "is the <i>i</i>-th training column."
        )
    else:
        st.info(
            f"{info['display_name']} does not expose `feature_importances_`. "
            "Neural networks have no equivalent built-in attribute; a permutation-importance "
            "study would be needed instead."
        )

    st.divider()
    st.subheader("Comparison with notebook-only models")
    st.caption(
        "These were trained later in the notebook on a wider 37-feature pipeline "
        "(27 polynomial + 10 domain features) and saved under filenames that are **not** "
        "in this repository. They are shown for context; the app cannot run inference "
        "with them."
    )

    table = pd.DataFrame(NOTEBOOK_ONLY_RESULTS, columns=["Model", "CV accuracy", "Test accuracy"])
    st.dataframe(
        table.style.format({"CV accuracy": "{:.4f}", "Test accuracy": "{:.4f}"}),
        width="stretch",
        hide_index=True,
        height=460,
    )
    st.warning(
        "Those near-perfect scores are not directly comparable to the shipped models. The "
        "notebook fits its MinMaxScaler on the full dataset before splitting, so test-set "
        "statistics leak into training. Reported ceilings of 100% should be read with that "
        "in mind.",
        icon="⚠️",
    )
