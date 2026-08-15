"""Page 3 - Batch Prediction."""

from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.constants import MODEL_REGISTRY, RAW_FEATURES
from utils.prediction import PredictionError, predict_frame
from utils.preprocessing import PreprocessingError, load_class_data
from views.ui import AMBER, GREEN, RED, note, style_fig

MAX_ROWS = 20_000


@st.cache_data(show_spinner=False)
def _template_csv() -> bytes:
    """A ready-to-use template drawn from real dataset rows, so the demo always works."""
    sample = load_class_data().sample(n=25, random_state=11)[RAW_FEATURES]
    return sample.to_csv(index=False).encode("utf-8")


def _read_upload(upload) -> pd.DataFrame:
    """Parse an uploaded CSV defensively; every failure returns a readable message."""
    raw = upload.getvalue()
    if not raw.strip():
        raise PreprocessingError("The uploaded file is empty.")
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise PreprocessingError("The uploaded file has no parsable rows.") from exc
        except pd.errors.ParserError as exc:
            raise PreprocessingError(
                "The file could not be parsed as CSV. Check for inconsistent column "
                f"counts or stray delimiters.\n\nParser said: {exc}"
            ) from exc
    raise PreprocessingError("Could not decode the file. Save it as UTF-8 CSV and retry.")


def render(model_key: str) -> None:
    st.title("Batch Prediction")
    st.caption(
        f"Score a whole CSV of measurements with "
        f"**{MODEL_REGISTRY[model_key]['display_name']}**."
    )

    head = st.columns([2.2, 1])
    with head[0]:
        st.markdown(
            "**Expected schema** — exactly these six columns, in any order, numeric:  \n"
            f"`{', '.join(RAW_FEATURES)}`"
        )
        note(
            "Extra columns are ignored. Rows with missing or non-numeric values are "
            "reported and skipped rather than silently guessed."
        )
    with head[1]:
        st.download_button(
            "Download template CSV",
            data=_template_csv(),
            file_name="power_grid_batch_template.csv",
            mime="text/csv",
            width="stretch",
        )

    upload = st.file_uploader("Upload a CSV", type=["csv"], label_visibility="collapsed")
    if upload is None:
        st.info("Upload a CSV to begin, or download the template above to see the format.")
        return

    # ------------------------------------------------------------------ parse
    try:
        df = _read_upload(upload)
    except PreprocessingError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not read the file.\n\n{type(exc).__name__}: {exc}")
        return

    st.subheader("Uploaded data")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    missing = [col for col in RAW_FEATURES if col.lower() not in
               {str(x).strip().lower() for x in df.columns}]
    c.metric("Required columns", f"{len(RAW_FEATURES) - len(missing)} of {len(RAW_FEATURES)}")
    st.dataframe(df.head(10), width="stretch")

    if missing:
        st.error(
            f"Missing required column(s): **{', '.join(missing)}**. "
            f"The file must contain {', '.join(RAW_FEATURES)}."
        )
        return

    if len(df) > MAX_ROWS:
        st.warning(f"File has {len(df):,} rows. Scoring the first {MAX_ROWS:,} only.")
        df = df.head(MAX_ROWS)

    # ------------------------------------------------------------------ predict
    try:
        with st.spinner(f"Scoring {len(df):,} rows…"):
            results = predict_frame(model_key, df)
    except (PredictionError, PreprocessingError) as exc:
        st.error(f"Batch prediction failed.\n\n{exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected error during batch inference.\n\n{type(exc).__name__}: {exc}")
        return

    dropped = results.attrs.get("n_dropped", 0)
    if dropped:
        st.warning(
            f"Skipped **{dropped}** row(s) containing missing or non-numeric values. "
            f"Row indices: {results.attrs.get('dropped_index', [])[:20]}"
            + (" …" if dropped > 20 else "")
        )

    st.divider()
    st.subheader("Results")

    faults = int((results["Status"] == "FAULT").sum())
    normals = int((results["Status"] == "NORMAL").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows scored", f"{len(results):,}")
    m2.metric("Faults", f"{faults:,}")
    m3.metric("Normal", f"{normals:,}")
    if results.attrs.get("has_probabilities"):
        m4.metric("Mean confidence", f"{results['Confidence'].mean():.1%}")
    else:
        m4.metric("Confidence", "n/a")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.dataframe(results, width="stretch", height=380)
    with right:
        counts = results["Predicted_Fault"].value_counts()
        fig = go.Figure(
            go.Bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
                marker_color=[GREEN if n == "No Fault" else AMBER for n in counts.index],
                text=counts.values,
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(title="Predicted class distribution", xaxis_title="Rows")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, height=380), width="stretch")

    if not results.attrs.get("has_probabilities"):
        st.info("Probability scores are not available for this model.")

    st.download_button(
        "Download Predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model_key}.csv",
        mime="text/csv",
        type="primary",
        width="stretch",
    )
