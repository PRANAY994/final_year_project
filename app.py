# =========================================================
# STREAMLIT DASHBOARD – MULTIMODAL CLIMATE DATA FUSION
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Multimodal Climate Data Fusion Dashboard",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================
APP_DIR = Path(__file__).resolve().parent
BASE = APP_DIR / "fusion_project"

METRICS_CSV = BASE / "results" / "metrics" / "metrics.csv"
PREDICTIONS_DIR = BASE / "results" / "predictions"

# =========================================================
# FUSION LABEL MAP
# =========================================================
FUSION_MAP = {
    "Early Fusion": "early",
    "Intermediate Fusion": "intermediate",
    "Late Fusion": "late",
    "Transformer Fusion": "transformer",
    "Confidence Fusion": "confidence_aware",
}

# =========================================================
# PREDICTION FILES
# =========================================================
PREDICTION_FILES = {
    "Early – LSTM": PREDICTIONS_DIR / "early_lstm.csv",
    "Early – XGBoost": PREDICTIONS_DIR / "early_xgboost.csv",

    "Intermediate – LSTM+TCN": PREDICTIONS_DIR / "intermediate_lstm_tcn.csv",
    "Intermediate – GRU+CNN (Gated)": PREDICTIONS_DIR / "intermediate_gru_cnn_gated.csv",

    "Late – Average": PREDICTIONS_DIR / "late_avg.csv",
    "Late – Meta": PREDICTIONS_DIR / "late_meta.csv",

    "Transformer Fusion": PREDICTIONS_DIR / "transformer_fusion.csv",
    "Confidence Fusion": PREDICTIONS_DIR / "confidence_fusion.csv",
}

# =========================================================
# DATA LOADERS
# =========================================================
@st.cache_data
def load_metrics():
    df = pd.read_csv(METRICS_CSV)
    df["fusion"] = df["fusion"].str.lower().str.strip()
    return df

@st.cache_data
def load_predictions(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    return df

metrics_df = load_metrics()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
# 🌍 Multimodal Climate Data Fusion Dashboard  
**ERA5 + Earth Observation | SPEI Prediction**
""")
st.markdown("---")

# ==========================================================
# SIDEBAR
# =========================================================
fusion_type = st.sidebar.selectbox(
    "Fusion Strategy",
    list(FUSION_MAP.keys())
)

fusion_key = FUSION_MAP[fusion_type]

# =========================================================
# MAIN TABS
# =========================================================
tab_arch, tab_perf, tab_ts = st.tabs(
    ["Architecture", "Model Performance", "SPEI Time-Series"]
)

# =========================================================
# TAB 1 — ARCHITECTURE
# =========================================================
with tab_arch:
    st.subheader("Fusion Architecture Overview")
    st.info(
        "This section describes the conceptual architecture of each fusion strategy.\n\n"
        "• Early Fusion: Feature-level concatenation\n"
        "• Intermediate Fusion: Learned temporal representations\n"
        "• Late Fusion: Decision-level aggregation\n"
        "• Transformer Fusion: Cross-modal attention\n"
        "• Confidence-Aware Fusion: Uncertainty-weighted fusion"
    )

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab_perf:
    st.subheader(f"{fusion_type} – Model Performance")

    subset = metrics_df[metrics_df["fusion"] == fusion_key]

    if subset.empty:
        st.warning("No metrics available for this fusion strategy.")
    else:
        model = st.selectbox("Select Model", subset["model"].unique())
        row = subset[subset["model"] == model].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{row.RMSE:.4f}")
        c2.metric("MAE",  f"{row.MAE:.4f}")
        c3.metric("R²",   f"{row.R2:.4f}")

        fig = px.bar(
            subset.sort_values("RMSE"),
            x="model",
            y="RMSE",
            color="model",
            title=f"{fusion_type} – RMSE Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 — SPEI TIME-SERIES
# =========================================================
with tab_ts:
    st.subheader("SPEI Prediction Over Time")

    model_name = st.selectbox(
        "Select Model",
        list(PREDICTION_FILES.keys())
    )

    df = load_predictions(PREDICTION_FILES[model_name])

    if df is None:
        st.error("Prediction file not found.")
    else:
        time_col = next((c for c in df.columns if "time" in c), None)
        pred_col = next((c for c in df.columns if "pred" in c), None)
        true_col = next((c for c in df.columns if "true" in c), None)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[pred_col],
                mode="lines",
                name="Prediction"
            )
        )

        if true_col:
            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=df[true_col],
                    mode="lines",
                    name="Ground Truth",
                    line=dict(color="black", dash="dot")
                )
            )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="SPEI",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Final Year Project • Multimodal Climate Data Fusion • ERA5 + Earth Observation"
)
