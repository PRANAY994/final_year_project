# =========================================================
# STREAMLIT DASHBOARD – MULTIMODAL CLIMATE DATA FUSION
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

# Try to import tensorflow/keras for model loading
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Multimodal Climate Data Fusion Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATHS
# =========================================================
APP_DIR = Path(__file__).resolve().parent
BASE = APP_DIR / "fusion_project"

METRICS_CSV = BASE / "results" / "metrics" / "metrics.csv"
PREDICTIONS_DIR = BASE / "results" / "predictions"
MODELS_DIR = BASE / "models"

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
    "Late – Hybrid (Meta)": PREDICTIONS_DIR / "late_hybrid_meta.csv",
    "Late – Hybrid (Avg)": PREDICTIONS_DIR / "late_hybrid_avg.csv",
    "Late – TCN (Average)": PREDICTIONS_DIR / "late_tcn_avg.csv",
    "Late – TCN (Meta)": PREDICTIONS_DIR / "late_tcn_meta.csv",
    "Late – TCN (Earth)": PREDICTIONS_DIR / "late_tcn_earth.csv",
    "Late – TCN (Era)": PREDICTIONS_DIR / "late_tcn_era.csv",
    "Late – Hybrid (Earth+TCN)": PREDICTIONS_DIR / "late_hybrid_earth_tcn.csv",
    "Late – Hybrid (Era+LSTM)": PREDICTIONS_DIR / "late_hybrid_era_lstm.csv",
    "Transformer Fusion": PREDICTIONS_DIR / "transformer_fusion.csv",
    "Confidence Fusion": PREDICTIONS_DIR / "confidence_fusion.csv",
}

# =========================================================
# MODEL ARCHITECTURES
# =========================================================
MODEL_ARCHITECTURES = {
    "Early – LSTM": {
        "type": "🔀 Early Fusion",
        "description": "Feature-level fusion at input stage",
        "architecture": """
        ┌─────────────────────────────────────────┐
        │  Earth Observation + ERA5 Features      │
        │      (Concatenated Input)               │
        └────────────────┬────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │  LSTM Layer (128 units)                 │
        │  - Processes fused temporal sequences   │
        └────────────────┬────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │  Dense Layers (64 → 32 → 1)             │
        │  - Regression output (SPEI prediction)  │
        └─────────────────────────────────────────┘
        """,
        "pros": ["✅ Simple & interpretable", "✅ Fast training", "✅ All features available from start"],
        "cons": ["❌ No modality-specific processing", "❌ Large input dimension"],
        "params": "~150K parameters"
    },
    
    "Early – XGBoost": {
        "type": "🔀 Early Fusion + Gradient Boosting",
        "description": "Feature-level fusion with XGBoost ensemble",
        "architecture": """
        ┌─────────────────────────────────────────┐
        │  Earth Observation + ERA5 Features      │
        │      (Concatenated Input)               │
        └────────────────┬────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │  XGBoost Ensemble (500 trees)           │
        │  - Sequential decision tree building    │
        │  - Gradient boosting optimization       │
        └────────────────┬────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │  SPEI Prediction (Regression)           │
        └─────────────────────────────────────────┘
        """,
        "pros": ["✅ Handles non-linear relationships", "✅ Feature importance available", "✅ Robust to outliers"],
        "cons": ["❌ Not deep learning", "❌ Less flexibility"],
        "params": "~500 trees"
    },
    
    "Intermediate – LSTM+TCN": {
        "type": "🔗 Intermediate Fusion",
        "description": "Learned temporal representations from both modalities",
        "architecture": """
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Earth Observation   │     │  ERA5 Climate Data   │
        │      Features        │     │      Features        │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   ▼                             ▼
        ┌──────────────────────┐     ┌──────────────────────┐
        │  LSTM (64 units)     │     │  TCN (Temporal CNN)  │
        │  - Temporal memory   │     │  - Dilated convolution
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Concatenate Features   │
                    │   (Intermediate fusion) │
                    └──────────────┬──────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │  Dense (64 → 32 → 1)   │
                    │  - SPEI Prediction      │
                    └─────────────────────────┘
        """,
        "pros": ["✅ Learns modality-specific features", "✅ Better temporal understanding", "✅ Balanced complexity"],
        "cons": ["❌ More parameters to tune", "❌ Moderate training time"],
        "params": "~220K parameters"
    },
    
    "Intermediate – GRU+CNN (Gated)": {
        "type": "🔗 Intermediate Fusion (Gated)",
        "description": "Learned gated fusion of GRU and CNN representations",
        "architecture": """
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Earth Observation   │     │  ERA5 Climate Data   │
        │      Features        │     │      Features        │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   ▼                             ▼
        ┌──────────────────────┐     ┌──────────────────────┐
        │  GRU (64 units)      │     │  CNN (1D Convolution)
        │  - Temporal modeling │     │  - Spatial patterns  │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Gated Fusion Layer     │
                    │  σ(W₁*x₁ + W₂*x₂ + b)  │
                    │  - Learned attention    │
                    └──────────────┬──────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │  Dense (64 → 32 → 1)   │
                    │  - SPEI Prediction      │
                    └─────────────────────────┘
        """,
        "pros": ["✅ Gated attention mechanism", "✅ Adaptive feature weighting", "✅ Better fusion control"],
        "cons": ["❌ Complex architecture", "❌ Requires more data"],
        "params": "~280K parameters"
    },
    
    "Late – Average": {
        "type": "🔚 Late Fusion (Simple Average)",
        "description": "Simple ensemble averaging of individual modality predictions",
        "architecture": """
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Earth Observation   │     │  ERA5 Climate Data   │
        │      Features        │     │      Features        │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   ▼                             ▼
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Model_1 (LSTM)      │     │  Model_2 (LSTM)      │
        │  Pred_1              │     │  Pred_2              │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Average Pooling        │
                    │  (Pred_1 + Pred_2) / 2 │
                    └──────────────┬──────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │  Final SPEI Prediction  │
                    └─────────────────────────┘
        """,
        "pros": ["✅ Simplest approach", "✅ Combines complementary views", "✅ Very fast inference"],
        "cons": ["❌ No learned fusion", "❌ Equal weighting may not be optimal"],
        "params": "No additional parameters"
    },
    
    "Late – Meta": {
        "type": "🔚 Late Fusion (Meta-Learner)",
        "description": "Learned meta-learner combines individual model predictions optimally",
        "architecture": """
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Earth Observation   │     │  ERA5 Climate Data   │
        │      Features        │     │      Features        │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   ▼                             ▼
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Model_1 (LSTM)      │     │  Model_2 (LSTM)      │
        │  Pred_1              │     │  Pred_2              │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Meta-Learner Network   │
                    │  (Dense: 32 → 16 → 1)  │
                    │  Learns optimal weights │
                    └──────────────┬──────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │  Final SPEI Prediction  │
                    │  y = σ(w₁*Pred_1 +     │
                    │       w₂*Pred_2 + b)   │
                    └─────────────────────────┘
        """,
        "pros": ["✅ Learns optimal fusion weights", "✅ Better than simple average", "✅ Good balance"],
        "cons": ["❌ Requires training data", "❌ Slight increase in parameters"],
        "params": "~1K additional parameters"
    },
    
    "Transformer Fusion": {
        "type": "⚡ Transformer Fusion",
        "description": "Cross-modal attention mechanism for fusion",
        "architecture": """
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Earth Observation   │     │  ERA5 Climate Data   │
        │  Embedding (64d)     │     │  Embedding (64d)     │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Multi-Head Attention   │
                    │  (4 heads × 16d)        │
                    │  - Cross-modal queries  │
                    │  - Self-attention       │
                    └──────────────┬──────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │  Feed-Forward Network   │
                    │  (256 → 128 → 64)       │
                    └──────────────┬──────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │  Output Layer           │
                    │  (1 → SPEI)             │
                    └─────────────────────────┘
        """,
        "pros": ["✅ Attention mechanism", "✅ Captures complex interactions", "✅ State-of-the-art"],
        "cons": ["❌ Requires more data", "❌ Complex to interpret"],
        "params": "~450K parameters"
    },
    
    "Confidence Fusion": {
        "type": "🎯 Confidence-Aware Fusion",
        "description": "Uncertainty-weighted fusion based on model confidence",
        "architecture": """
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Earth Observation   │     │  ERA5 Climate Data   │
        │      Features        │     │      Features        │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   ▼                             ▼
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Pred_1 + Var_1      │     │  Pred_2 + Var_2      │
        │  (Prediction +       │     │  (Prediction +       │
        │   Uncertainty)       │     │   Uncertainty)       │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Confidence Weighting   │
                    │  w₁ = 1/Var_1           │
                    │  w₂ = 1/Var_2           │
                    │  Normalize: w₁/(w₁+w₂) │
                    └──────────────┬──────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │  Weighted Average       │
                    │  Output = w₁*Pred_1 +  │
                    │           w₂*Pred_2    │
                    └─────────────────────────┘
        """,
        "pros": ["✅ Accounts for uncertainty", "✅ Adaptive weighting", "✅ More robust"],
        "cons": ["❌ Requires variance estimates", "❌ Sensitive to noise"],
        "params": "Dynamic weighting scheme"
    },
}

# Fill in remaining models with generic descriptions
for model_name in PREDICTION_FILES.keys():
    if model_name not in MODEL_ARCHITECTURES:
        fusion_type = "Late" if "Late" in model_name else "Intermediate" if "Intermediate" in model_name else "Early"
        MODEL_ARCHITECTURES[model_name] = {
            "type": f"🔚 {fusion_type} Fusion Variant",
            "description": f"Specialized {fusion_type.lower()} fusion architecture",
            "architecture": f"Custom {model_name} fusion network",
            "pros": ["✅ Specialized design", "✅ Optimized for specific task"],
            "cons": ["❌ Model-specific implementation"],
            "params": "Varies by model"
        }

# =========================================================
# DATA LOADERS (CACHED)
# =========================================================
@st.cache_data
def load_metrics():
    try:
        df = pd.read_csv(METRICS_CSV)
        df["fusion"] = df["fusion"].str.lower().str.strip()
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_predictions(path: Path):
    """Load prediction CSV file"""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Handle different time formats
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        elif 'date' in df.columns:
            df['time'] = pd.to_datetime(df['date'])
        elif 'year' in df.columns and 'month' in df.columns:
            # Combine year and month into datetime
            df['time'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
        else:
            # Create a dummy time column if none exists
            df['time'] = pd.to_datetime('2019-01-01')
        
        return df
    except Exception as e:
        st.error(f"Error loading {path.name}: {str(e)}")
        return None

@st.cache_resource
def load_model(model_path):
    """Load trained Keras model"""
    try:
        if KERAS_AVAILABLE and model_path.exists():
            model = keras.models.load_model(model_path)
            return model
        return None
    except Exception as e:
        st.warning(f"Could not load model: {str(e)}")
        return None

# =========================================================
# COMPUTE METRICS
# =========================================================
def compute_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"RMSE": 0, "MAE": 0, "R²": 0, "MAPE": 0}
    
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    }

# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================
def plot_actual_vs_predicted(df):
    """Scatter plot: Actual vs Predicted"""
    fig = px.scatter(
        df,
        x='y_true',
        y='y_pred',
        color='model',
        hover_data=['latitude', 'longitude', 'time'],
        title="Actual vs Predicted SPEI",
        labels={'y_true': 'Actual (Ground Truth)', 'y_pred': 'Predicted'},
        opacity=0.6
    )
    # Add diagonal line (perfect prediction)
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='black', dash='dash', width=2)
        )
    )
    fig.update_layout(height=600, template='plotly_white')
    return fig

def plot_timeseries(df, location_lat=None, location_lon=None):
    """Time-series plot with actual vs predicted"""
    if location_lat is not None and location_lon is not None:
        df = df[(df['latitude'] == location_lat) & (df['longitude'] == location_lon)]
    
    df = df.sort_values('time')
    
    fig = go.Figure()
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['y_pred'],
            mode='lines+markers',
            name='Predicted SPEI',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=5)
        )
    )
    
    # Add ground truth
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['y_true'],
            mode='lines+markers',
            name='Actual SPEI',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            marker=dict(size=5)
        )
    )
    
    title = f"SPEI Time Series (Lat: {location_lat:.1f}, Lon: {location_lon:.1f})" if location_lat else "SPEI Time Series"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="SPEI Value",
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    return fig

def plot_residuals(df):
    """Residual plot"""
    df = df.copy()
    df['residual'] = df['y_true'] - df['y_pred']
    
    fig = px.scatter(
        df,
        x='y_pred',
        y='residual',
        color='model',
        title="Residual Plot (Actual - Predicted)",
        labels={'y_pred': 'Predicted SPEI', 'residual': 'Residual'},
        opacity=0.6
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
    fig.update_layout(height=600, template='plotly_white')
    
    return fig

def plot_error_distribution(df):
    """Histogram of prediction errors"""
    df = df.copy()
    df['abs_error'] = np.abs(df['y_true'] - df['y_pred'])
    
    fig = px.histogram(
        df,
        x='abs_error',
        nbins=50,
        color='model',
        title="Distribution of Absolute Errors",
        labels={'abs_error': 'Absolute Error', 'count': 'Frequency'},
        barmode='overlay',
        opacity=0.7
    )
    fig.update_layout(height=600, template='plotly_white')
    return fig

def plot_metrics_radar(metrics_dict):
    """Radar chart for multiple metrics"""
    if not metrics_dict:
        return None
    
    models = list(metrics_dict.keys())
    metrics = ['RMSE', 'MAE', 'MAPE']
    
    # Normalize metrics for better visualization
    normalized = {}
    for metric in metrics:
        values = [metrics_dict[m].get(metric, 0) for m in models]
        max_val = max(values) if values else 1
        normalized[metric] = [v / max_val if max_val > 0 else 0 for v in values]
    
    fig = go.Figure()
    
    for i, model in enumerate(models):
        values = [normalized[metric][i] for metric in metrics]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model
            )
        )
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=600,
        title="Model Comparison (Normalized Metrics)"
    )
    return fig

# =========================================================
# HEADER
# =========================================================
st.markdown("""
# 🌍 Multimodal Climate Data Fusion Dashboard  
**ERA5 + Earth Observation | SPEI Prediction**

*Explore and visualize predictions from deep learning models fusing satellite and reanalysis data.*
""")

metrics_df = load_metrics()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("## 📍 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Overview", "🔍 Model Predictions", "📈 Performance Analysis", "🗺️ Spatial Analysis", "🎯 Quick Comparison", "🔮 Make Predictions"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info("""
**SPEI**: Standardized Precipitation-Evapotranspiration Index

Predicting drought/flood conditions by fusing:
- 🛰️ Earth Observation (vegetation, surface temp)
- 📊 ERA5 Climate (temperature, precipitation, pressure)
""")

# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if page == "📊 Overview":
    st.header("Welcome to Prediction Explorer 👋")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Total Models", len(PREDICTION_FILES))
    with col2:
        st.metric("🔄 Fusion Strategies", len(FUSION_MAP))
    with col3:
        st.metric("📊 Predictions", "~1,363 per model")
    
    st.markdown("---")
    
    st.markdown("""
    ## What is This Dashboard?
    
    This interactive dashboard displays **SPEI predictions** from multiple deep learning models that combine:
    - **Earth Observation Data**: Satellite-derived vegetation indices, surface temperature, moisture
    - **ERA5 Climate Reanalysis**: Historical temperature, precipitation, pressure, and wind data
    
    ### 🎯 Why Visualize Predictions?
    
    1. **Model Comparison** – Identify which fusion strategy works best
    2. **Error Analysis** – Understand prediction failures and biases
    3. **Spatial Patterns** – See regional performance variations
    4. **Temporal Trends** – Track accuracy across time periods
    5. **Decision Support** – Build confidence in model predictions for drought/flood forecasting
    
    ### 📊 Understanding the Metrics
    
    | Metric | Best | Interpretation |
    |--------|------|-----------------|
    | **RMSE** | ↓ Lower | Root Mean Squared Error – average prediction error magnitude |
    | **MAE** | ↓ Lower | Mean Absolute Error – typical error size |
    | **R²** | ↑ Higher | Coefficient of determination (0-1) – explains variance in data |
    | **MAPE** | ↓ Lower | Mean Absolute Percentage Error – relative error in percent |
    
    ### 🚀 Get Started
    
    Use the sidebar to explore:
    - **🔍 Model Predictions**: View detailed predictions for any single model
    - **📈 Performance Analysis**: Compare all models side-by-side
    - **🗺️ Spatial Analysis**: See performance by geographic location
    - **🎯 Quick Comparison**: Compare 2-3 models directly
    """)

# =========================================================
# PAGE 2: MODEL PREDICTIONS
# =========================================================
elif page == "🔍 Model Predictions":
    st.header("🔍 Interactive Prediction Explorer")
    
    model_name = st.selectbox(
        "🎯 Select Model to Analyze",
        list(PREDICTION_FILES.keys()),
        help="Choose a specific model to visualize its predictions"
    )
    
    df = load_predictions(PREDICTION_FILES[model_name])
    
    if df is None:
        st.error(f"❌ Prediction file not found")
        st.stop()
    
    st.success(f"✅ Loaded {len(df):,} predictions from {len(df['latitude'].unique())} locations")
    
    # Compute overall metrics
    metrics = compute_metrics(df['y_true'], df['y_pred'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"{metrics['RMSE']:.4f}", delta="Lower is better", delta_color="inverse")
    col2.metric("MAE", f"{metrics['MAE']:.4f}", delta="Lower is better", delta_color="inverse")
    col3.metric("R²", f"{metrics['R²']:.4f}", delta="Higher is better")
    col4.metric("MAPE", f"{metrics['MAPE']:.2f}%", delta="Lower is better", delta_color="inverse")
    
    st.markdown("---")
    
    # Display Model Architecture
    if model_name in MODEL_ARCHITECTURES:
        arch_info = MODEL_ARCHITECTURES[model_name]
        
        st.subheader(f"🏗️ Model Architecture: {arch_info['type']}")
        
        arch_col1, arch_col2 = st.columns([2, 1])
        
        with arch_col1:
            st.markdown(f"**Description:** {arch_info['description']}")
            st.code(arch_info['architecture'], language="text")
        
        with arch_col2:
            st.markdown("**Strengths:**")
            for pro in arch_info['pros']:
                st.write(pro)
            
            st.markdown("**Limitations:**")
            for con in arch_info['cons']:
                st.write(con)
            
            st.markdown(f"**Parameters:** {arch_info['params']}")
        
        st.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Scatter Plot", "⏱️ Time Series", "📉 Residuals", "📊 Error Distribution", "📋 Data"]
    )
    
    with tab1:
        st.subheader("Actual vs Predicted Values")
        st.plotly_chart(plot_actual_vs_predicted(df), use_container_width=True)
        st.info("""
        💡 **How to interpret:**
        - Points on the diagonal line = perfect predictions
        - Points above diagonal = model overestimated
        - Points below diagonal = model underestimated
        - Tight clustering = consistent, reliable model
        """)
    
    with tab2:
        st.subheader("SPEI Time Series at Selected Location")
        
        locations = sorted(df[['latitude', 'longitude']].drop_duplicates().values.tolist())
        if locations:
            selected_loc = st.selectbox(
                "📍 Pick a location",
                locations,
                format_func=lambda x: f"Lat: {x[0]:.1f}°, Lon: {x[1]:.1f}°"
            )
            st.plotly_chart(
                plot_timeseries(df, selected_loc[0], selected_loc[1]),
                use_container_width=True
            )
            
            # Stats for this location
            loc_data = df[(df['latitude'] == selected_loc[0]) & (df['longitude'] == selected_loc[1])]
            loc_metrics = compute_metrics(loc_data['y_true'], loc_data['y_pred'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{loc_metrics['RMSE']:.4f}")
            col2.metric("MAE", f"{loc_metrics['MAE']:.4f}")
            col3.metric("R²", f"{loc_metrics['R²']:.4f}")
            col4.metric("Samples", len(loc_data))
        else:
            st.warning("No location data available")
    
    with tab3:
        st.subheader("Residual Analysis")
        st.plotly_chart(plot_residuals(df), use_container_width=True)
        st.info("""
        💡 **How to interpret:**
        - Random scatter around zero line = good model
        - Patterns/trends = systematic bias
        - Large residuals = outliers or problem periods
        """)
    
    with tab4:
        st.subheader("Error Distribution")
        st.plotly_chart(plot_error_distribution(df), use_container_width=True)
        st.info("""
        💡 **How to interpret:**
        - Left-skewed distribution = consistently accurate
        - Right tail = occasional large errors
        - Multiple peaks = different error modes
        """)
    
    with tab5:
        st.subheader("Prediction Data Table")
        
        col1, col2 = st.columns(2)
        with col1:
            n_rows = st.slider("Rows to display", 10, min(500, len(df)), 50)
        with col2:
            if st.checkbox("Show all columns", value=False):
                columns_to_show = df.columns.tolist()
            else:
                columns_to_show = ['latitude', 'longitude', 'time', 'y_true', 'y_pred']
        
        display_df = df[columns_to_show].head(n_rows).copy()
        display_df['error'] = np.abs(display_df['y_true'] - display_df['y_pred'])
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download displayed data as CSV",
            data=csv,
            file_name=f"{model_name}_predictions.csv",
            mime="text/csv"
        )

# =========================================================
# PAGE 3: PERFORMANCE ANALYSIS
# =========================================================
elif page == "📈 Performance Analysis":
    st.header("📈 Compare All Models")
    
    st.info("Loading and comparing all prediction models...")
    
    # Load all predictions
    all_metrics = {}
    for model_name, path in PREDICTION_FILES.items():
        df = load_predictions(path)
        if df is not None:
            all_metrics[model_name] = compute_metrics(df['y_true'], df['y_pred'])
    
    if not all_metrics:
        st.error("No prediction data available")
        st.stop()
    
    st.success(f"✅ Loaded {len(all_metrics)} models")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(all_metrics).T.reset_index()
    comparison_df.columns = ['Model', 'RMSE', 'MAE', 'R²', 'MAPE']
    comparison_df = comparison_df.sort_values('RMSE')
    
    st.markdown("---")
    
    # Ranking tables
    st.subheader("🏆 Model Rankings")
    
    rank_col1, rank_col2, rank_col3, rank_col4 = st.columns(4)
    
    with rank_col1:
        st.subheader("🥇 Best RMSE")
        best_rmse = comparison_df.nsmallest(5, 'RMSE')[['Model', 'RMSE']].reset_index(drop=True)
        best_rmse.index = best_rmse.index + 1
        st.dataframe(best_rmse, use_container_width=True)
    
    with rank_col2:
        st.subheader("🥇 Best MAE")
        best_mae = comparison_df.nsmallest(5, 'MAE')[['Model', 'MAE']].reset_index(drop=True)
        best_mae.index = best_mae.index + 1
        st.dataframe(best_mae, use_container_width=True)
    
    with rank_col3:
        st.subheader("🥇 Best R²")
        best_r2 = comparison_df.nlargest(5, 'R²')[['Model', 'R²']].reset_index(drop=True)
        best_r2.index = best_r2.index + 1
        st.dataframe(best_r2, use_container_width=True)
    
    with rank_col4:
        st.subheader("🥇 Best MAPE")
        best_mape = comparison_df.nsmallest(5, 'MAPE')[['Model', 'MAPE']].reset_index(drop=True)
        best_mape.index = best_mape.index + 1
        st.dataframe(best_mape, use_container_width=True)
    
    st.markdown("---")
    
    # Comparison charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_rmse = px.bar(
            comparison_df.nsmallest(10, 'RMSE'),
            x='RMSE',
            y='Model',
            orientation='h',
            color='RMSE',
            color_continuous_scale='RdYlGn_r',
            title="Top 10 Models by RMSE (Lower is Better)"
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with chart_col2:
        fig_r2 = px.bar(
            comparison_df.nlargest(10, 'R²'),
            x='R²',
            y='Model',
            orientation='h',
            color='R²',
            color_continuous_scale='Greens',
            title="Top 10 Models by R² (Higher is Better)"
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # All models table
    st.subheader("📊 All Models Detailed Metrics")
    st.dataframe(
        comparison_df.sort_values('RMSE').reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
    
    # Radar chart comparison (top 5)
    st.subheader("⭐ Top 5 Models – Radar Comparison")
    top_5_models = comparison_df.nsmallest(5, 'RMSE')['Model'].tolist()
    top_5_metrics = {model: all_metrics[model] for model in top_5_models}
    
    fig_radar = plot_metrics_radar(top_5_metrics)
    if fig_radar:
        st.plotly_chart(fig_radar, use_container_width=True)

# =========================================================
# PAGE 4: SPATIAL ANALYSIS
# =========================================================
elif page == "🗺️ Spatial Analysis":
    st.header("🗺️ Geographic Performance Analysis")
    
    model_name = st.selectbox("📍 Select Model", list(PREDICTION_FILES.keys()))
    df = load_predictions(PREDICTION_FILES[model_name])
    
    if df is None:
        st.error("Prediction file not found")
        st.stop()
    
    # Compute spatial metrics
    spatial_data = []
    for (lat, lon), group in df.groupby(['latitude', 'longitude']):
        metrics = compute_metrics(group['y_true'], group['y_pred'])
        metrics['latitude'] = lat
        metrics['longitude'] = lon
        spatial_data.append(metrics)
    
    spatial_df = pd.DataFrame(spatial_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric_choice = st.selectbox(
            "📊 Metric to Display",
            ['RMSE', 'MAE', 'R²', 'MAPE']
        )
    with col2:
        size_metric = st.selectbox(
            "📏 Size indicator",
            ['RMSE', 'MAE', 'MAPE', 'None']
        )
    
    # Create map
    size_col = None if size_metric == 'None' else size_metric
    
    fig = px.scatter_mapbox(
        spatial_df,
        lat='latitude',
        lon='longitude',
        color=metric_choice,
        size=size_col,
        hover_data=['RMSE', 'MAE', 'R²', 'MAPE'],
        color_continuous_scale='RdYlGn_r' if metric_choice != 'R²' else 'Greens',
        zoom=2,
        center={"lat": spatial_df['latitude'].mean(), "lon": spatial_df['longitude'].mean()},
        title=f"Model Performance: {metric_choice} by Location",
        size_max=30
    )
    fig.update_layout(mapbox_style="open-street-map", height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Spatial statistics
    st.subheader("📊 Spatial Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Highest RMSE", f"{spatial_df['RMSE'].max():.4f}")
    col2.metric("Lowest RMSE", f"{spatial_df['RMSE'].min():.4f}")
    col3.metric("Mean RMSE", f"{spatial_df['RMSE'].mean():.4f}")
    col4.metric("Std Dev RMSE", f"{spatial_df['RMSE'].std():.4f}")
    
    st.dataframe(
        spatial_df.sort_values('RMSE'),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# PAGE 5: QUICK COMPARISON
# =========================================================
elif page == "🎯 Quick Comparison":
    st.header("🎯 Side-by-Side Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model1 = st.selectbox("📌 Model 1", list(PREDICTION_FILES.keys()), key="m1")
    with col2:
        model2 = st.selectbox("📌 Model 2", list(PREDICTION_FILES.keys()), key="m2", 
                             index=1 if len(PREDICTION_FILES) > 1 else 0)
    with col3:
        model3 = st.selectbox("📌 Model 3", ["None"] + list(PREDICTION_FILES.keys()), key="m3")
    
    models_to_compare = [m for m in [model1, model2, model3] if m != "None"]
    
    # Load data
    comparison_data = {}
    for model_name in models_to_compare:
        df = load_predictions(PREDICTION_FILES[model_name])
        if df is not None:
            comparison_data[model_name] = compute_metrics(df['y_true'], df['y_pred'])
    
    if not comparison_data:
        st.error("Could not load prediction data")
        st.stop()
    
    # Comparison table
    comp_df = pd.DataFrame(comparison_data).T
    st.subheader("📊 Metrics Comparison")
    st.dataframe(comp_df, use_container_width=True)
    
    # Visual comparison
    st.subheader("📈 Visual Comparison")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        fig_comp_rmse = px.bar(
            x=list(comparison_data.keys()),
            y=[comparison_data[m]['RMSE'] for m in comparison_data.keys()],
            labels={'x': 'Model', 'y': 'RMSE'},
            title="RMSE Comparison",
            color=list(comparison_data.keys())
        )
        st.plotly_chart(fig_comp_rmse, use_container_width=True)
    
    with comp_col2:
        fig_comp_r2 = px.bar(
            x=list(comparison_data.keys()),
            y=[comparison_data[m]['R²'] for m in comparison_data.keys()],
            labels={'x': 'Model', 'y': 'R² Score'},
            title="R² Comparison",
            color=list(comparison_data.keys())
        )
        st.plotly_chart(fig_comp_r2, use_container_width=True)

# =========================================================
# PAGE 6: MAKE PREDICTIONS (NEW)
# =========================================================
elif page == "🔮 Make Predictions":
    st.header("🔮 Real-Time SPEI Prediction")
    
    if not KERAS_AVAILABLE:
        st.error("❌ TensorFlow/Keras not installed. Install it with: `pip install tensorflow`")
        st.info("Once installed, you'll be able to make predictions using trained models.")
        st.stop()
    
    # Model selection
    model_select = st.selectbox(
        "🤖 Select Model",
        [
            "Early – LSTM",
            "Intermediate – LSTM+TCN",
            "Intermediate – GRU+CNN (Gated)",
            "Late – Meta",
            "Transformer Fusion"
        ]
    )
    
    # Map to model paths
    model_paths = {
        "Early – LSTM": MODELS_DIR / "early" / "lstm_early_fusion.keras",
        "Intermediate – LSTM+TCN": MODELS_DIR / "intermediate" / "lstm_tcn_intermediate.keras",
        "Intermediate – GRU+CNN (Gated)": MODELS_DIR / "intermediate" / "gru_cnn_gated_intermediate.keras",
        "Late – Meta": MODELS_DIR / "late" / "earth_lstm.keras",  # Example
        "Transformer Fusion": MODELS_DIR / "transformer_fusion" / "transformer_fusion.keras",
    }
    
    model_path = model_paths.get(model_select)
    
    st.markdown("---")
    
    st.subheader("📥 Input Features")
    st.info("""
    Enter the feature values for SPEI prediction:
    - **Earth Observation**: Satellite-derived indices (typically normalized -2 to 2)
    - **ERA5 Climate**: Reanalysis climate variables (typically normalized -3 to 3)
    """)
    
    # Feature input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🛰️ Earth Observation Features")
        eo_ndvi = st.slider("NDVI (Vegetation Index)", -1.0, 1.0, 0.5, 0.01, help="Normalized Difference Vegetation Index")
        eo_lst = st.slider("Land Surface Temp", -3.0, 3.0, 0.0, 0.1, help="Normalized Land Surface Temperature")
        eo_sm = st.slider("Soil Moisture", -2.0, 2.0, 0.0, 0.1, help="Normalized Soil Moisture")
        eo_lai = st.slider("Leaf Area Index", -2.0, 2.0, 0.0, 0.1, help="Normalized LAI")
    
    with col2:
        st.subheader("📊 ERA5 Climate Features")
        era_temp = st.slider("Temperature (2m)", -3.0, 3.0, 0.0, 0.1, help="Normalized 2m Temperature")
        era_precip = st.slider("Precipitation", -2.0, 3.0, 0.0, 0.1, help="Normalized Precipitation")
        era_pressure = st.slider("Surface Pressure", -2.0, 2.0, 0.0, 0.1, help="Normalized Pressure")
        era_wind = st.slider("Wind Speed", -2.0, 2.0, 0.0, 0.1, help="Normalized Wind Speed")
    
    with col3:
        st.subheader("📍 Geospatial Info")
        latitude = st.number_input("Latitude", -90.0, 90.0, 14.9, 0.1)
        longitude = st.number_input("Longitude", -180.0, 180.0, 76.9, 0.1)
        month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        year = st.number_input("Year", 2015, 2025, 2023, 1)
    
    st.markdown("---")
    
    # Prepare features
    features = np.array([
        [eo_ndvi, eo_lst, eo_sm, eo_lai,
         era_temp, era_precip, era_pressure, era_wind]
    ])
    
    # Make prediction button
    col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 2])
    
    with col_pred1:
        predict_btn = st.button("🚀 Get Prediction", use_container_width=True)
    
    with col_pred2:
        clear_btn = st.button("🔄 Reset", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    if predict_btn:
        # Try to load and use model
        model = load_model(model_path)
        
        if model is None:
            st.warning(f"⚠️ Model file not found at: {model_path}")
            st.info("""
            For now, showing simulated prediction based on input features.
            To use actual models, ensure they are saved in the models directory.
            """)
            # Simulate prediction
            pred_spei = eo_ndvi * 0.3 + era_precip * 0.25 - era_temp * 0.15 + eo_sm * 0.2 + np.random.normal(0, 0.3)
        else:
            try:
                # Reshape for model input
                features_reshaped = features.reshape(1, -1)
                pred_spei = model.predict(features_reshaped, verbose=0)[0][0]
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                pred_spei = None
        
        if pred_spei is not None:
            st.markdown("---")
            
            # Display prediction
            col_display1, col_display2 = st.columns([1.5, 1])
            
            with col_display1:
                st.subheader("🎯 SPEI Prediction Result")
                
                # Color code based on SPEI value
                if pred_spei > 1.5:
                    condition = "🟢 Extremely Wet"
                    color = "green"
                elif pred_spei > 1.0:
                    condition = "🟦 Very Wet"
                    color = "lightblue"
                elif pred_spei > 0.5:
                    condition = "🟩 Moderately Wet"
                    color = "lightgreen"
                elif pred_spei > -0.5:
                    condition = "🟨 Near Normal"
                    color = "yellow"
                elif pred_spei > -1.0:
                    condition = "🟧 Moderately Dry"
                    color = "orange"
                elif pred_spei > -1.5:
                    condition = "🟥 Severely Dry"
                    color = "red"
                else:
                    condition = "⬛ Extremely Dry"
                    color = "darkred"
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2>SPEI Value: {pred_spei:.3f}</h2>
                    <h3>{condition}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                **SPEI Scale Interpretation:**
                - **> 1.5**: Extremely Wet
                - **1.0 to 1.5**: Very Wet
                - **0.5 to 1.0**: Moderately Wet
                - **-0.5 to 0.5**: Near Normal
                - **-1.0 to -0.5**: Moderately Dry
                - **-1.5 to -1.0**: Severely Dry
                - **< -1.5**: Extremely Dry
                """)
            
            with col_display2:
                st.subheader("📊 Prediction Input Summary")
                summary = {
                    "Location": f"{latitude:.1f}°N, {longitude:.1f}°E",
                    "Date": f"{month:02d}/{year}",
                    "Model": model_select,
                    "NDVI": f"{eo_ndvi:.2f}",
                    "Soil Moisture": f"{eo_sm:.2f}",
                    "Precipitation": f"{era_precip:.2f}",
                    "Temperature": f"{era_temp:.2f}",
                }
                
                summary_df = pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Visualization
            st.markdown("---")
            
            st.subheader("📈 Feature Contribution Analysis")
            
            features_names = ["NDVI", "LST", "Soil Moisture", "LAI", "Temperature", "Precipitation", "Pressure", "Wind"]
            features_values = features[0]
            
            fig = px.bar(
                x=features_names,
                y=np.abs(features_values),
                color=features_values,
                color_continuous_scale='RdBu',
                title="Input Feature Values (Absolute Magnitude)",
                labels={'y': 'Feature Value (Normalized)', 'x': 'Features'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Historical comparison
            st.markdown("---")
            st.subheader("📚 Historical Context")
            
            # Load some sample data for context
            sample_df = load_predictions(PREDICTION_FILES[list(PREDICTION_FILES.keys())[0]])
            if sample_df is not None:
                # Get predictions from same location if available
                same_loc = sample_df[(sample_df['latitude'] == latitude) & (sample_df['longitude'] == longitude)]
                if len(same_loc) > 0:
                    avg_pred = same_loc['y_pred'].mean()
                    st.info(f"💡 Average prediction at this location (historical): **{avg_pred:.3f}**")
                    
                    # Comparison chart
                    comparison_vals = [avg_pred, pred_spei]
                    comparison_labels = ["Historical Avg", "Your Prediction"]
                    
                    fig_comp = px.bar(
                        x=comparison_labels,
                        y=comparison_vals,
                        color=comparison_labels,
                        title="Your Prediction vs Historical Average",
                        labels={'y': 'SPEI Value', 'x': ''}
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>🌍 Multimodal Climate Data Fusion Dashboard • Final Year Project</p>
    <p>Fusing ERA5 Climate Data + Earth Observation for SPEI Prediction</p>
</div>
""", unsafe_allow_html=True)
