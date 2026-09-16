import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from fraud_detection.config import (  # noqa: E402
    DECISION_THRESHOLD,
    FEATURE_NAMES,
    METRICS_PATH,
)
from fraud_detection.predict import predict_batch, predict_transaction  # noqa: E402

# Representative example transactions (index 0 = Time, 1-28 = V1-V28, 29 = Amount).
EXAMPLE_LEGITIMATE = [0.0] * 30
EXAMPLE_LEGITIMATE[0] = 406.0
EXAMPLE_LEGITIMATE[29] = 45.0

EXAMPLE_FRAUD = [0.0] * 30
EXAMPLE_FRAUD[0] = 406.0
EXAMPLE_FRAUD[29] = 0.0
for idx, value in {
    1: -2.31, 2: 1.95, 3: -1.61, 4: 3.99, 5: -0.52,
    6: -1.43, 7: -2.54, 8: 1.39, 9: -2.77, 10: -2.77,
}.items():
    EXAMPLE_FRAUD[idx] = value


@st.cache_resource(show_spinner="Loading model...")
def _warm_model():
    """Touch the model/scaler once so later predictions don't pay the load cost."""
    from fraud_detection.predict import _load_artifacts

    return _load_artifacts()


@st.cache_data
def _load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH).iloc[0]
    return None


st.set_page_config(
    page_title="Argus - Intelligent Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }

    .hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e2a4a 0%, #3a1f47 100%);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0 0 0.35rem 0; font-size: 2rem; }
    .hero p { margin: 0; color: rgba(255,255,255,0.75); font-size: 1.02rem; }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 0.5rem;
    }
    .result-card.fraud {
        background: rgba(255, 75, 75, 0.12);
        border: 1px solid rgba(255, 75, 75, 0.45);
    }
    .result-card.legit {
        background: rgba(60, 200, 130, 0.12);
        border: 1px solid rgba(60, 200, 130, 0.45);
    }
    .result-card .verdict { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
    .result-card .prob { font-size: 2.4rem; font-weight: 800; margin: 0.25rem 0; }
    .result-card .sub { color: rgba(255,255,255,0.65); font-size: 0.9rem; }

    .example-btn-row { margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Argus</h1>
        <p style="font-weight:600; font-size:1.05rem; margin-bottom:0.5rem;">
        Intelligent Credit Card Fraud Detection System</p>
        <p>An artificial neural network trained on anonymized, PCA-transformed
        transaction data — class-weighted and threshold-tuned to catch fraud in a
        dataset where it's just 0.17% of all transactions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    _warm_model()
except FileNotFoundError as exc:
    st.error(
        "Model artifacts are missing, so predictions can't run.\n\n"
        f"{exc}"
    )
    st.stop()

metrics = _load_metrics()

with st.sidebar:
    st.header("Model Performance")
    st.caption("Test-set results at the F1-optimal decision threshold")
    if metrics is not None:
        m1, m2 = st.columns(2)
        m1.metric("Precision", f"{metrics['Precision']:.1%}")
        m2.metric("Recall", f"{metrics['Recall']:.1%}")
        m3, m4 = st.columns(2)
        m3.metric("F1-Score", f"{metrics['F1-Score']:.1%}")
        m4.metric("PR-AUC", f"{metrics['PR-AUC']:.3f}")
        st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
    else:
        st.info("Run evaluation to populate metrics.")

    st.divider()
    st.caption("Built with TensorFlow/Keras + Streamlit.")

if "features" not in st.session_state:
    st.session_state.features = [0.0] * 30

tab_single, tab_batch = st.tabs(["🔎 Single Transaction", "📂 Batch Upload"])

with tab_single:
    left, right = st.columns([1.3, 1], gap="large")

    with left:
        st.subheader("Try an example")
        col_a, col_b = st.columns(2)
        if col_a.button("✅ Load legitimate example", use_container_width=True):
            st.session_state.features = list(EXAMPLE_LEGITIMATE)
        if col_b.button("⚠️ Load suspicious example", use_container_width=True):
            st.session_state.features = list(EXAMPLE_FRAUD)

        st.subheader("Transaction Details")
        time_val = st.number_input(
            "Time (seconds since first transaction)",
            min_value=0.0,
            value=st.session_state.features[0],
        )
        amount_val = st.number_input(
            "Amount ($)", min_value=0.0, value=st.session_state.features[29]
        )

        with st.expander("Transaction Features (V1-V28)", expanded=False):
            v_values = []
            cols = st.columns(4)
            for i in range(1, 29):
                with cols[(i - 1) % 4]:
                    v_values.append(
                        st.number_input(
                            f"V{i}",
                            value=st.session_state.features[i],
                            format="%.6f",
                            key=f"v_{i}",
                        )
                    )

        predict_clicked = st.button(
            "Predict Transaction", type="primary", use_container_width=True
        )

    with right:
        st.subheader("Result")
        if predict_clicked:
            transaction = [time_val, *v_values, amount_val]
            result, probability = predict_transaction(transaction)
            is_fraud = result == "Fraud"

            st.markdown(
                f"""
                <div class="result-card {'fraud' if is_fraud else 'legit'}">
                    <div class="verdict">{'⚠️ FRAUD DETECTED' if is_fraud else '✅ LEGITIMATE'}</div>
                    <div class="prob">{probability * 100:.2f}%</div>
                    <div class="sub">fraud probability &middot; decision threshold {DECISION_THRESHOLD * 100:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(probability, 1.0))
        else:
            st.info("Fill in transaction details and click **Predict Transaction** to see a result here.")

with tab_batch:
    st.subheader("Batch Prediction")
    st.write(
        "Upload a CSV with columns "
        f"`{', '.join(FEATURE_NAMES)}` to score multiple transactions at once."
    )

    uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            scored_df = predict_batch(input_df)
        except ValueError as exc:
            st.error(str(exc))
        else:
            n_fraud = int((scored_df["prediction"] == "Fraud").sum())
            n_total = len(scored_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total transactions", f"{n_total:,}")
            c2.metric("Flagged as fraud", f"{n_fraud:,}")
            c3.metric("Flag rate", f"{(n_fraud / n_total if n_total else 0):.2%}")

            st.dataframe(scored_df, use_container_width=True)

            st.download_button(
                "⬇️ Download results as CSV",
                data=scored_df.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )
