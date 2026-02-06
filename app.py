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
import joblib

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

        # Normalize prediction column names to expected `y_true` and `y_pred`
        col_map = {}
        if 'spei_true' in df.columns and 'spei_pred' in df.columns:
            col_map['spei_true'] = 'y_true'
            col_map['spei_pred'] = 'y_pred'
        # common alternatives
        if 'true' in df.columns and 'pred' in df.columns:
            col_map.setdefault('true', 'y_true')
            col_map.setdefault('pred', 'y_pred')
        if 'actual' in df.columns and 'predicted' in df.columns:
            col_map.setdefault('actual', 'y_true')
            col_map.setdefault('predicted', 'y_pred')
        # apply mapping
        if col_map:
            df = df.rename(columns=col_map)

        # Ensure expected columns exist and provide sensible defaults
        if 'y_true' not in df.columns and 'y_pred' not in df.columns:
            # nothing to plot; return as-is
            pass

        # Add a model column if missing (use filename stem)
        if 'model' not in df.columns:
            try:
                df['model'] = path.stem
            except Exception:
                df['model'] = 'model'
        
        return df
    except Exception as e:
        st.error(f"Error loading {path.name}: {str(e)}")
        return None


@st.cache_data
def load_training_data():
    """Load original dataset used for training (SPEI series)."""
    try:
        data_path = APP_DIR / "datasets" / "early_fusion_dataset.csv"
        if not data_path.exists():
            return pd.DataFrame()
        tdf = pd.read_csv(data_path)
        # normalize column names
        tdf.columns = [c.lower().strip() for c in tdf.columns]
        if 'valid_time' in tdf.columns:
            tdf['time'] = pd.to_datetime(tdf['valid_time'])
        elif 'time' in tdf.columns:
            tdf['time'] = pd.to_datetime(tdf['time'])
        # target column name in dataset is 'spei6_new'
        if 'spei6_new' in tdf.columns:
            tdf['y_train'] = tdf['spei6_new']
        else:
            # fallback: try SPEI column
            tdf['y_train'] = tdf.get('spei', np.nan)
        return tdf
    except Exception as e:
        st.warning(f"Could not load training dataset: {e}")
        return pd.DataFrame()

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


@st.cache_resource
def load_xgb_model(model_path):
    """Load a joblib XGBoost model if present"""
    try:
        if model_path.exists():
            return joblib.load(str(model_path))
        return None
    except Exception as e:
        st.warning(f"Could not load XGBoost model: {str(e)}")
        return None


@st.cache_resource
def load_feature_names(path):
    """Load feature names (joblib) used by the XGBoost model"""
    try:
        if path.exists():
            return joblib.load(str(path))
        return None
    except Exception as e:
        st.warning(f"Could not load feature names: {str(e)}")
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
    
    # attempt to load training series (original dataset)
    train_df = load_training_data()

    fig = go.Figure()

    # Determine training and validation cutoffs (training: <=2015, validation: 2016-2018)
    train_cutoff = pd.Timestamp('2015-12-31')
    val_cutoff = pd.Timestamp('2018-12-31')

    # Plot training + validation series from original dataset if available
    if not train_df.empty and location_lat is not None and location_lon is not None:
        tloc = train_df[(train_df['latitude'] == location_lat) & (train_df['longitude'] == location_lon)].copy()
        if not tloc.empty:
            tloc = tloc.sort_values('time')
            t_train = tloc[tloc['time'] <= train_cutoff]
            t_val = tloc[(tloc['time'] > train_cutoff) & (tloc['time'] <= val_cutoff)]
            if not t_train.empty:
                fig.add_trace(
                    go.Scatter(
                        x=t_train['time'],
                        y=t_train['y_train'],
                        mode='lines+markers',
                        name='Training (Actual SPEI)',
                        line=dict(color='gray', width=2),
                        marker=dict(size=4),
                        hovertemplate='%{x|%Y-%m-%d}: %{y:.3f}<extra>Training</extra>'
                    )
                )
            if not t_val.empty:
                fig.add_trace(
                    go.Scatter(
                        x=t_val['time'],
                        y=t_val['y_train'],
                        mode='lines+markers',
                        name='Validation (Actual SPEI)',
                        line=dict(color='lightgray', width=2, dash='dot'),
                        marker=dict(size=4),
                        hovertemplate='%{x|%Y-%m-%d}: %{y:.3f}<extra>Validation</extra>'
                    )
                )

    # Add red vertical separator at the end of validation using a shape (datetime-safe)
    cutoff_str = str(val_cutoff)
    fig.add_shape(
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=cutoff_str,
            x1=cutoff_str,
            y0=0,
            y1=1,
            line=dict(color='red', width=2, dash='dash')
        )
    )

    fig.add_annotation(
        x=cutoff_str,
        y=1.02,
        xref='x',
        yref='paper',
        text='Train/Test Split',
        showarrow=False,
        xanchor='left',
        font=dict(color='red')
    )

    # Plot post-training actual and predicted from the predictions dataframe
    post_df = df[df['time'] > train_cutoff]
    # If there is any pre-split data in the predictions df (rare), we still plot it as 'Actual (full)'
    if not df.empty and (df['time'] <= train_cutoff).any():
        pre_df = df[df['time'] <= train_cutoff]
        fig.add_trace(
            go.Scatter(
                x=pre_df['time'],
                y=pre_df['y_true'],
                mode='lines+markers',
                name='Actual (Full)',
                line=dict(color='#ff7f0e', width=2, dash='dot'),
                marker=dict(size=5)
            )
        )

    if not post_df.empty:
        fig.add_trace(
            go.Scatter(
                x=post_df['time'],
                y=post_df['y_true'],
                mode='lines+markers',
                name='Actual (Post-Train)',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=6)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=post_df['time'],
                y=post_df['y_pred'],
                mode='lines+markers',
                name='Predicted (Post-Train)',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            )
        )

    # If no training dataset was available, fall back to previous behavior (plot all actual & pred)
    if train_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['y_pred'], mode='lines+markers', name='Predicted SPEI', line=dict(color='#1f77b4', width=2), marker=dict(size=5))
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['y_true'], mode='lines+markers', name='Actual SPEI', line=dict(color='#ff7f0e', width=2, dash='dot'), marker=dict(size=5))
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


def plot_spatial_error_map(df):
    """Map showing mean absolute error per location (lat/lon)."""
    df2 = df.copy()
    if 'latitude' not in df2.columns or 'longitude' not in df2.columns:
        return None

    df2['abs_error'] = np.abs(df2['y_true'] - df2['y_pred'])
    grouped = (
        df2.groupby(['latitude', 'longitude'], as_index=False)
        .agg(mean_abs_error=('abs_error', 'mean'), samples=('y_true', 'size'))
    )

    if grouped.empty:
        return None

    fig = px.scatter_mapbox(
        grouped,
        lat='latitude',
        lon='longitude',
        color='mean_abs_error',
        size='mean_abs_error',
        hover_data=['mean_abs_error', 'samples'],
        color_continuous_scale='RdYlGn_r',
        title='Mean Absolute Error by Location',
        size_max=18,
        zoom=2
    )
    fig.update_layout(mapbox_style='open-street-map', height=600)
    return fig


def plot_error_overview(df):
    """Single combined figure: spatial map (mean abs error) + error distribution histogram."""
    import plotly.subplots as sp

    df2 = df.copy()
    if 'latitude' not in df2.columns or 'longitude' not in df2.columns:
        return plot_error_distribution(df2)

    df2['abs_error'] = np.abs(df2['y_true'] - df2['y_pred'])
    grouped = (
        df2.groupby(['latitude', 'longitude'], as_index=False)
        .agg(mean_abs_error=('abs_error', 'mean'), samples=('y_true', 'size'), std_error=('abs_error', 'std'))
    )

    # create subplots: map (left) + histogram (right)
    fig = sp.make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], specs=[[{"type": "mapbox"}, {"type": "xy"}]], horizontal_spacing=0.02)

    # map trace
    if not grouped.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=grouped['latitude'],
                lon=grouped['longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=np.clip(grouped['mean_abs_error'] * 8, 6, 30),
                    color=grouped['mean_abs_error'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title='Mean |Error|')
                ),
                hovertemplate='Lat: %{lat}<br>Lon: %{lon}<br>Mean |err|: %{marker.color:.3f}<br>Samples: %{customdata[0]}<extra></extra>',
                customdata=np.stack([grouped['samples'].values, grouped['mean_abs_error'].values, grouped['std_error'].fillna(0).values], axis=1),
                name='Mean Abs Error'
            ),
            row=1, col=1
        )

    # histogram of absolute errors (all points)
    fig.add_trace(
        go.Histogram(
            x=df2['abs_error'],
            nbinsx=50,
            marker_color='indianred',
            name='Absolute Error Distribution',
            opacity=0.8
        ),
        row=1, col=2
    )

    # layout for mapbox - center on data
    if not grouped.empty:
        center = dict(lat=grouped['latitude'].mean(), lon=grouped['longitude'].mean())
    else:
        center = dict(lat=df2['latitude'].mean(), lon=df2['longitude'].mean()) if 'latitude' in df2.columns else dict(lat=0, lon=0)

    fig.update_layout(
        mapbox=dict(style='open-street-map', center=center, zoom=3),
        height=600,
        title='Spatial Error Overview — Map (left) + Error Distribution (right)'
    )

    # tidy axes for histogram
    fig.update_xaxes(title_text='Absolute Error', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)

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
        st.subheader("Error Distribution & Spatial Hotspots")
        # Compute spatial error stats
        df_err = df.copy()
        df_err['abs_error'] = np.abs(df_err['y_true'] - df_err['y_pred'])
        spatial_stats = (
            df_err.groupby(['latitude', 'longitude'], as_index=False)
            .agg(
                mean_error=('abs_error', 'mean'),
                max_error=('abs_error', 'max'),
                min_error=('abs_error', 'min'),
                std_error=('abs_error', 'std'),
                samples=('y_true', 'size')
            )
        )
        
        # Single unified map with all error details
        fig = px.scatter_mapbox(
            spatial_stats,
            lat='latitude',
            lon='longitude',
            color='mean_error',
            size='mean_error',
            hover_data={
                'latitude': ':.2f',
                'longitude': ':.2f',
                'mean_error': ':.4f',
                'max_error': ':.4f',
                'min_error': ':.4f',
                'std_error': ':.4f',
                'samples': True
            },
            color_continuous_scale='RdYlGn_r',
            title='Error Distribution by Location (Size & Color = Mean Absolute Error)',
            size_max=25,
            zoom=2,
            center={'lat': spatial_stats['latitude'].mean(), 'lon': spatial_stats['longitude'].mean()}
        )
        fig.update_layout(mapbox_style='open-street-map', height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        💡 **How to interpret:**
        - **Color & Size**: Darker red = higher mean absolute error; larger circles = higher error
        - **Hover**: Shows mean, max, min, std deviation of errors & sample count at each location
        - **High-error zones** (red hotspots) indicate regions where the model struggles most
        """)
        
        st.markdown("---")
        st.subheader("📊 Detailed Error Visualization by Location & SPEI Values")
        
        # Create comprehensive scatter plot with all information
        df_detailed = df.copy()
        df_detailed['error'] = np.abs(df_detailed['y_true'] - df_detailed['y_pred'])
        
        # Sort by error descending
        df_detailed = df_detailed.sort_values('error', ascending=False).reset_index(drop=True)
        
        # Create scatter plot: Latitude vs Error, with Longitude as color, size as error magnitude
        fig_detailed = px.scatter(
            df_detailed,
            x='latitude',
            y='error',
            color='longitude',
            size='error',
            hover_data={
                'latitude': ':.2f',
                'longitude': ':.2f',
                'y_true': ':.4f',
                'y_pred': ':.4f',
                'error': ':.4f',
                'time': True
            },
            labels={
                'latitude': 'Latitude',
                'error': 'Absolute Error',
                'y_true': 'Actual SPEI',
                'y_pred': 'Predicted SPEI',
                'longitude': 'Longitude'
            },
            color_continuous_scale='Viridis',
            title='Error Distribution: Latitude vs Error (size & hover show all details)',
            height=600
        )
        fig_detailed.update_layout(template='plotly_white')
        st.plotly_chart(fig_detailed, use_container_width=True)
        
        st.write("**Hover over points to see: Latitude, Longitude, Actual SPEI, Predicted SPEI, Error, and Time**")
        
        st.markdown("---")
        
        # Alternative view: Actual vs Predicted with error coloring
        fig_alt = px.scatter(
            df_detailed,
            x='y_true',
            y='y_pred',
            color='error',
            size='error',
            hover_data={
                'latitude': ':.2f',
                'longitude': ':.2f',
                'y_true': ':.4f',
                'y_pred': ':.4f',
                'error': ':.4f'
            },
            labels={
                'y_true': 'Actual SPEI',
                'y_pred': 'Predicted SPEI',
                'error': 'Absolute Error'
            },
            color_continuous_scale='RdYlGn_r',
            title='Actual vs Predicted SPEI (colored & sized by error; high error = red & large)',
            height=600
        )
        # Add diagonal line for perfect prediction
        min_val = min(df_detailed['y_true'].min(), df_detailed['y_pred'].min())
        max_val = max(df_detailed['y_true'].max(), df_detailed['y_pred'].max())
        fig_alt.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                      name='Perfect Prediction', line=dict(color='black', dash='dash', width=2))
        )
        fig_alt.update_layout(template='plotly_white')
        st.plotly_chart(fig_alt, use_container_width=True)
        
        st.write("**Points far from diagonal = high error; hover to see latitude, longitude, and error details**")
    
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
    st.header("🗺️ Spatial-Temporal Analysis")
    
    model_name = st.selectbox("📍 Select Model", list(PREDICTION_FILES.keys()))
    df = load_predictions(PREDICTION_FILES[model_name])
    
    if df is None:
        st.error("Prediction file not found")
        st.stop()
    
    # Line plot: Predicted SPEI over time for each location
    df_time = df.copy()
    df_time = df_time.sort_values('time')
    
    # Create location label (lat/lon)
    df_time['location'] = df_time['latitude'].round(2).astype(str) + '°N, ' + df_time['longitude'].round(2).astype(str) + '°E'
    
    # Line plot showing predicted SPEI over time
    fig_temporal = px.line(
        df_time,
        x='time',
        y='y_pred',
        color='location',
        hover_data={
            'latitude': ':.2f',
            'longitude': ':.2f',
            'y_true': ':.4f',
            'y_pred': ':.4f',
            'time': '|%Y-%m-%d'
        },
        title='Spatial-Temporal Analysis: Predicted SPEI Over Time by Location',
        labels={'time': 'Date', 'y_pred': 'Predicted SPEI', 'location': 'Location'},
        height=700
    )
    fig_temporal.update_layout(
        hovermode='x unified',
        template='plotly_white',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01, bgcolor='rgba(255,255,255,0.8)')
    )
    st.plotly_chart(fig_temporal, use_container_width=True)
    
    st.info("""
    💡 **How to interpret:**
    - **Each line = one location (lat/lon)**
    - **Y-axis**: Predicted SPEI value
    - **X-axis**: Time (from 2019 to present)
    - **Hover**: Shows latitude, longitude, actual SPEI, predicted SPEI, and date
    - **Patterns**: Compare SPEI trends across different geographic locations over time
    """)

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
            "Early – XGBoost",
            "Intermediate – LSTM+TCN",
            "Intermediate – GRU+CNN (Gated)",
            "Late – Meta",
            "Transformer Fusion"
        ]
    )
    
    # Map to model paths
    model_paths = {
        "Early – LSTM": MODELS_DIR / "early" / "lstm_early_fusion.keras",
        "Early – XGBoost": MODELS_DIR / "early" / "xgboost_early_fusion.pkl",
        "Intermediate – LSTM+TCN": MODELS_DIR / "intermediate" / "lstm_tcn_intermediate.keras",
        "Intermediate – GRU+CNN (Gated)": MODELS_DIR / "intermediate" / "gru_cnn_gated_intermediate.keras",
        "Late – Meta": MODELS_DIR / "late" / "earth_lstm.keras",  # Example
        "Transformer Fusion": MODELS_DIR / "transformer_fusion" / "transformer_fusion.keras",
    }
    
    model_path = model_paths.get(model_select)
    
    st.markdown("---")

    # If user selected Early - XGBoost, show form-based inputs and use joblib model
    if model_select == "Early – XGBoost":
        xgb_model_path = MODELS_DIR / "early" / "xgboost_early_fusion.pkl"
        feat_path = MODELS_DIR / "early" / "feature_names.pkl"
        xgb_model = load_xgb_model(xgb_model_path)
        feature_names = load_feature_names(feat_path)

        if xgb_model is None or feature_names is None:
            st.warning("⚠️ XGBoost model or feature names not found. Please run the training script first (train_early_fusion.py).")
        else:
            st.subheader("📥 Input Features (Early - XGBoost)")
            st.info("Enter exact values for each feature; press Predict when ready.")

            # Form for inputs
            with st.form(key="xgb_predict_form"):
                cols = st.columns(3)
                xgb_inputs = {}
                for idx, feat in enumerate(feature_names):
                    with cols[idx % 3]:
                        xgb_inputs[feat] = st.text_input(feat, value="0.0", key=f"xgb_{idx}")

                submit_xgb = st.form_submit_button("🔮 Predict SPEI6")

            if submit_xgb:
                # convert and predict
                try:
                    converted = {}
                    for k, v in xgb_inputs.items():
                        converted[k] = float(v)
                except ValueError as e:
                    st.error(f"Invalid numeric input: {e}")
                else:
                    input_df = pd.DataFrame([converted])[feature_names]
                    try:
                        pred = xgb_model.predict(input_df)[0]
                        st.success("✓ Prediction successful!")
                        st.metric("Predicted SPEI6", f"{pred:.4f}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    else:
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
