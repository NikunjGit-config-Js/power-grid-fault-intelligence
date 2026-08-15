"""Page 6 - About Project."""

from __future__ import annotations

import streamlit as st

from utils.constants import CLASS_LABELS, N_MODEL_FEATURES, POLY_FEATURE_NAMES, RAW_FEATURES
from views.ui import note

PIPELINE = """Electrical Measurements     Ia, Ib, Ic, Va, Vb, Vc
            &darr;
Data Preparation            validation, type coercion, finite checks
            &darr;
Feature Engineering         degree-2 polynomial expansion &rarr; 27 terms
            &darr;
Preprocessing               min-max scaling (scaler refit from classData.csv)
            &darr;
ML / DL Models              Decision Tree | Random Forest | Dense MLP
            &darr;
Fault Classification        encoded class 0-5 &rarr; human-readable fault"""


def render() -> None:
    st.title("About This Project")

    st.subheader("Problem")
    st.markdown(
        "Transmission-line faults — a conductor touching ground, two phases shorting, or "
        "all three collapsing together — have to be identified in milliseconds. Protection "
        "relays must know not just *that* something is wrong but *which* fault it is, "
        "because the correct switching response differs by type. This project classifies "
        "the fault directly from three-phase current and voltage measurements."
    )

    st.subheader("Pipeline")
    st.markdown(f'<div class="pg-pipe">{PIPELINE}</div>', unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("Dataset")
        st.markdown(
            "Simulated in MATLAB with four 11 kV generators feeding a transmission line, "
            "with faults injected at the line midpoint.\n\n"
            "- `classData.csv` — 7,861 labelled rows. Four indicator bits `G, C, B, A` "
            "combine into a 4-character code; six of those codes actually occur.\n"
            "- `detect_dataset.csv` — 12,001 rows with a binary `Output (S)` fault flag.\n\n"
            "The six classes are: " + ", ".join(f"`{c}`" for c in CLASS_LABELS) + "."
        )

        st.subheader("Feature engineering")
        st.markdown(
            f"The {len(RAW_FEATURES)} raw measurements are expanded to "
            f"**{N_MODEL_FEATURES}** features by a degree-2 polynomial transform: the six "
            "originals, six squared terms, and fifteen pairwise products.\n\n"
            "The products matter physically. `Ia Va` approximates instantaneous power on "
            "phase A, and cross terms like `Ia Ib` capture the phase imbalance that "
            "distinguishes a two-phase fault from a symmetrical three-phase one — "
            "relationships a model cannot recover from the six raw channels alone."
        )
        with st.expander("All 27 features, in model order"):
            st.code(", ".join(POLY_FEATURE_NAMES), language=None)

    with c2:
        st.subheader("Models")
        st.markdown(
            "- **Decision Tree** — 89.89% test accuracy. Best of the three and fully "
            "interpretable.\n"
            "- **Random Forest** — 88.30%. Steadier probabilities, 16 MB artifact.\n"
            "- **Dense MLP** — 38.53%. Included for honesty, not performance.\n\n"
            "All three take the identical 27-feature vector. Both tree models were trained "
            "on a SMOTE-balanced training split."
        )

        st.subheader("Why it matters")
        st.markdown(
            "Faster and more specific fault classification shortens outages, narrows the "
            "section of line that has to be de-energised, and reduces equipment damage from "
            "sustained fault current. A model that runs on measurements a substation "
            "already collects needs no new sensors to deploy."
        )

    st.divider()
    st.subheader("Limitations — read these before trusting any number here")

    st.markdown(
        "**Preprocessing leakage.** The notebook fits its MinMaxScaler on the entire "
        "dataset before splitting, so test-set minima and maxima influence the training "
        "transform. This app reproduces that behaviour deliberately: the shipped models "
        "were trained under it, and fitting the scaler on the training split alone would "
        "shift every feature and corrupt predictions. It does mean the reported accuracies "
        "are optimistic.\n\n"
        "**Simulated data.** Every row comes from a MATLAB simulation. Real substation "
        "telemetry carries sensor noise, CT saturation, and harmonics that this dataset "
        "does not contain.\n\n"
        "**Single-snapshot inference.** Each prediction uses one instant of measurement "
        "with no time context, so genuinely sequential behaviour — inrush, fault evolution, "
        "breaker operation — is invisible to the model.\n\n"
        "**Two questionable class names.** Codes `1001` and `0110` carry notebook labels "
        "that disagree with the source dataset's own documentation. The names are kept "
        "because the models were trained under them, and the app flags them at the point "
        "of prediction.\n\n"
        "**Documentation mismatch.** The README describes a hybrid CNN-LSTM. The saved "
        "artifact is a Dense MLP. This app reports what the artifact actually is."
    )

    st.subheader("What I would do next")
    st.markdown(
        "1. Refit the scaler inside a `Pipeline` on the training split only, and republish "
        "honest metrics — the leak is the single biggest correctness issue in the project.\n"
        "2. Ship the fitted scaler and label encoder as artifacts so inference does not "
        "have to reconstruct them from the CSV.\n"
        "3. Retrain the neural network with far lighter dropout; 0.5 after every layer on a "
        "14.5k-parameter network is why it collapses to one class.\n"
        "4. Add the LightGBM and CatBoost artifacts, which the notebook trains but never "
        "saves into this repository.\n"
        "5. Move to windowed, sequential inputs so the model can use fault dynamics rather "
        "than a single snapshot."
    )

    st.divider()
    note(
        "Built on the public repository "
        "<code>Marvellous-Udoye/power-grid-fault-detection-and-classification</code>. "
        "The original notebook, datasets, and trained artifacts are unmodified — this "
        "application adds the inference layer and interface around them."
    )
