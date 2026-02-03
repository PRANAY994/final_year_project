# =========================================================
# STREAMLIT DASHBOARD – MULTIMODAL CLIMATE DATA FUSION
# =========================================================

import streamlit as st
import pandas as pd
import json
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
# PATHS (MATCH YOUR PROJECT TREE)
# =========================================================
APP_DIR = Path(__file__).resolve().parent
BASE = APP_DIR / "fusion_project"

METRICS_CSV = BASE / "results" / "metrics" / "metrics.csv"

TRANSFORMER_METRICS = BASE / "models" / "transformer_fusion" / "metrics.json"
CONFIDENCE_METRICS  = BASE / "models" / "confidence_fusion" / "metrics.json"

PREDICTIONS_DIR = BASE / "results" / "predictions"

PREDICTION_FILES = {
    "Early – LSTM": PREDICTIONS_DIR / "early_lstm.csv",
    "Early – XGBoost": PREDICTIONS_DIR / "early_xgboost.csv",

    "Intermediate – LSTM+TCN": PREDICTIONS_DIR / "intermediate_lstm_tcn.csv",
    "Intermediate – GRU+CNN (Gated)": PREDICTIONS_DIR / "intermediate_gru_cnn_gated.csv",

    "Late – Average": PREDICTIONS_DIR / "late_avg.csv",
    "Late – Meta": PREDICTIONS_DIR / "late_meta.csv",

    "Transformer Fusion": BASE / "models" / "transformer_fusion" / "predictions.csv",
    "Confidence Fusion": BASE / "models" / "confidence_fusion" / "predictions.csv",
}

# =========================================================
# HELPERS
# =========================================================
def pred_key(name: str) -> str:
    return f"predicted_{name}"

# =========================================================
# DATA LOADERS (ROBUST)
# =========================================================
@st.cache_data
def load_metrics_csv():
    rows = []

    with open(METRICS_CSV, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append(parts[:5])

    df = pd.DataFrame(rows, columns=["fusion", "model", "RMSE", "MAE", "R2"])
    df[["RMSE", "MAE", "R2"]] = df[["RMSE", "MAE", "R2"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df["fusion"] = df["fusion"].str.lower()
    return df.dropna()

@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_prediction_csv(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    return df

# =========================================================
# LOAD DATA
# =========================================================
metrics_df = load_metrics_csv()
transformer_metrics = load_json(TRANSFORMER_METRICS)
confidence_metrics  = load_json(CONFIDENCE_METRICS)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
# 🌍 Multimodal Climate Data Fusion Dashboard  
**ERA5 + Earth Observation | SPEI Prediction**
""")
st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================
fusion_type = st.sidebar.selectbox(
    "Fusion Strategy",
    [
        "Early Fusion",
        "Intermediate Fusion",
        "Late Fusion",
        "Transformer Fusion",
        "Confidence Fusion"
    ]
)

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
        "Architecture diagrams can be displayed here.\n\n"
        "This tab is intentionally descriptive and visual-only."
    )

# =========================================================
# TAB 2 — MODEL PERFORMANCE (WITH PREDICT BUTTON)
# =========================================================
with tab_perf:
    st.subheader(f"{fusion_type} – Model Performance")

    # ---------------------------------------------
    # EARLY / INTERMEDIATE / LATE
    # ---------------------------------------------
    if fusion_type in ["Early Fusion", "Intermediate Fusion", "Late Fusion"]:
        fusion_key = fusion_type.split()[0].lower()
        subset = metrics_df[metrics_df["fusion"] == fusion_key]

        model = st.selectbox("Select Model", subset["model"].unique())
        key = pred_key(model)

        if st.button("🔮 Predict", key=f"btn_{key}"):
            st.session_state[key] = True

        if st.session_state.get(key, False):
            row = subset[subset["model"] == model].iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{row.RMSE:.4f}")
            c2.metric("MAE",  f"{row.MAE:.4f}")
            c3.metric("R²",   f"{row.R2:.4f}")

            fig = px.bar(
                subset,
                x="model",
                y="RMSE",
                color="model",
                title="RMSE Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click **Predict** to compute metrics.")

    # ---------------------------------------------
    # TRANSFORMER
    # ---------------------------------------------
    elif fusion_type == "Transformer Fusion":
        key = pred_key("transformer")

        if st.button("🔮 Predict Transformer"):
            st.session_state[key] = True

        if st.session_state.get(key, False) and transformer_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{transformer_metrics.get('rmse', 0):.4f}")
            c2.metric("MAE",  f"{transformer_metrics.get('mae', 0):.4f}")
            c3.metric("R²",   f"{transformer_metrics.get('r2', 0):.4f}")
            st.json(transformer_metrics)
        else:
            st.info("Click **Predict Transformer**.")

    # ---------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------
    elif fusion_type == "Confidence Fusion":
        key = pred_key("confidence")

        if st.button("🔮 Predict Confidence Fusion"):
            st.session_state[key] = True

        if st.session_state.get(key, False) and confidence_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{confidence_metrics.get('rmse', 0):.4f}")
            c2.metric("MAE",  f"{confidence_metrics.get('mae', 0):.4f}")
            c3.metric("R²",   f"{confidence_metrics.get('r2', 0):.4f}")
            st.json(confidence_metrics)
        else:
            st.info("Click **Predict Confidence Fusion**.")

# =========================================================
# TAB 3 — SPEI TIME-SERIES (WITH PREDICT BUTTON)
# =========================================================
with tab_ts:
    st.subheader("SPEI Prediction Over Time")

    model_name = st.selectbox(
        "Select Model",
        list(PREDICTION_FILES.keys())
    )

    key = pred_key(model_name)

    if st.button("🔮 Predict SPEI", key=f"ts_{key}"):
        st.session_state[key] = True

    if not st.session_state.get(key, False):
        st.info("Click **Predict SPEI** to visualize predictions.")
    else:
        df = load_prediction_csv(PREDICTION_FILES[model_name])

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
                    name=model_name
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
