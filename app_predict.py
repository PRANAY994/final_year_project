"""
Streamlit UI for Early Fusion XGBoost Model Predictions
Users can input feature values and get predictions without running the training code
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Early Fusion SPEI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Early Fusion - SPEI6 Prediction")
st.markdown("---")

# ================================
# PATHS
# ================================
BASE = "c:\\Users\\hp\\OneDrive\\Desktop\\final_year_project\\fusion_project"
MODEL_PATH = f"{BASE}\\models\\early\\xgboost_early_fusion.pkl"
FEATURES_PATH = f"{BASE}\\models\\early\\feature_names.pkl"

# ================================
# LOAD MODEL AND FEATURES
# ================================
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}")
        st.info("Please run `python train_early_fusion.py` first to train the model.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_feature_names():
    """Load feature names used in training"""
    if not os.path.exists(FEATURES_PATH):
        st.error(f"❌ Feature names not found at {FEATURES_PATH}")
        st.stop()
    return joblib.load(FEATURES_PATH)

# Load model and features
try:
    model = load_model()
    feature_names = load_feature_names()
    st.success("✓ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ================================
# SIDEBAR - MODEL INFO
# ================================
with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"""
    **Model Type:** XGBoost Regressor
    
    **Features:** {len(feature_names)}
    
    **Target:** SPEI6 (Standardized Precipitation Evapotranspiration Index)
    
    **Fusion Type:** Early Fusion
    """)
    
    st.markdown("---")
    st.subheader("📋 Feature Categories")
    
    # Categorize features for better organization
    temporal_features = ['year', 'month']
    spatial_features = ['latitude', 'longitude']
    
    vegetation_features = [f for f in feature_names if any(x in f for x in ['NDVI', 'VCI', 'TCI', 'VHI'])]
    temperature_features = [f for f in feature_names if any(x in f for x in ['LST', 't2m', 'd2m'])]
    precipitation_features = [f for f in feature_names if any(x in f for x in ['tp', 'ssrd', 'swvl'])]
    spei_features = [f for f in feature_names if 'SPEI6' in f]
    
    st.caption(f"🕐 Temporal: {len(temporal_features)}")
    st.caption(f"📍 Spatial: {len(spatial_features)}")
    st.caption(f"🌿 Vegetation: {len(vegetation_features)}")
    st.caption(f"🌡️ Temperature: {len(temperature_features)}")
    st.caption(f"💧 Precipitation: {len(precipitation_features)}")
    st.caption(f"📈 SPEI (Lag): {len(spei_features)}")

# ================================
# INPUT FORM
# ================================
st.header("📝 Enter Feature Values")
st.markdown("Fill in all the required values below and click **Predict SPEI6**:")

# Create form for all inputs
with st.form(key="prediction_form"):
    
    # ================================
    # SECTION 1: TEMPORAL & SPATIAL
    # ================================
    st.subheader("🕐 Temporal & Spatial Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        year_val = st.text_input("Year", value="2020", key="year_input")
    with col2:
        month_val = st.text_input("Month", value="6", key="month_input")
    with col3:
        lat_val = st.text_input("Latitude", value="35.0", key="lat_input")
    with col4:
        lon_val = st.text_input("Longitude", value="70.0", key="lon_input")
    
    st.markdown("---")
    
    # ================================
    # SECTION 2: VEGETATION INDICES
    # ================================
    st.subheader("🌿 Vegetation Indices")
    vegetation_cols = sorted([f for f in feature_names if any(x in f for x in ['NDVI', 'VCI', 'TCI', 'VHI'])])
    
    veg_inputs = {}
    if vegetation_cols:
        cols = st.columns(3)
        for idx, feature in enumerate(vegetation_cols):
            with cols[idx % 3]:
                default_val = "0.5" if "NDVI" in feature or "VCI" in feature or "TCI" in feature or "VHI" in feature else "0.0"
                veg_inputs[feature] = st.text_input(
                    feature,
                    value=default_val,
                    key=f"veg_{idx}"
                )
    
    st.markdown("---")
    
    # ================================
    # SECTION 3: TEMPERATURE
    # ================================
    st.subheader("🌡️ Temperature Variables")
    temp_cols = sorted([f for f in feature_names if any(x in f for x in ['LST', 't2m', 'd2m'])])
    
    temp_inputs = {}
    if temp_cols:
        cols = st.columns(3)
        for idx, feature in enumerate(temp_cols):
            with cols[idx % 3]:
                temp_inputs[feature] = st.text_input(
                    feature,
                    value="20.0",
                    key=f"temp_{idx}"
                )
    
    st.markdown("---")
    
    # ================================
    # SECTION 4: PRECIPITATION & SPEI
    # ================================
    st.subheader("💧 Precipitation & SPEI Values")
    precip_spei_cols = sorted([f for f in feature_names if any(x in f for x in ['tp', 'ssrd', 'swvl', 'SPEI6'])])
    
    precip_inputs = {}
    if precip_spei_cols:
        cols = st.columns(3)
        for idx, feature in enumerate(precip_spei_cols):
            with cols[idx % 3]:
                default_val = "100.0" if any(x in feature for x in ['tp', 'ssrd']) else "0.5"
                precip_inputs[feature] = st.text_input(
                    feature,
                    value=default_val,
                    key=f"precip_{idx}"
                )
    
    st.markdown("---")
    
    # ================================
    # SUBMIT BUTTON
    # ================================
    submit_button = st.form_submit_button(
        "🔮 Predict SPEI6",
        use_container_width=True,
        type="primary"
    )
    
    # Collect all input values
    input_values = {
        'year': year_val,
        'month': month_val,
        'latitude': lat_val,
        'longitude': lon_val,
    }
    input_values.update(veg_inputs)
    input_values.update(temp_inputs)
    input_values.update(precip_inputs)

# ================================
# PROCESS PREDICTION
# ================================
if submit_button:
    # Prepare input data
    try:
        # Convert string inputs to floats
        converted_inputs = {}
        for key, value in input_values.items():
            try:
                converted_inputs[key] = float(value)
            except ValueError:
                st.error(f"❌ Invalid value for {key}: '{value}' is not a number")
                st.stop()
        
        # Create a dataframe with the input values
        input_df = pd.DataFrame([converted_inputs])
        
        # Ensure all features are present and in correct order
        missing_features = [f for f in feature_names if f not in converted_inputs]
        
        if missing_features:
            st.warning(f"⚠️ Missing {len(missing_features)} features. Using default values.")
            for feature in missing_features:
                input_df[feature] = 0.0
        
        # Reorder columns to match training order
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display prediction with styling
        st.success("✓ Prediction successful!")
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        st.markdown("---")
        
        # Create metric cards
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label="Predicted SPEI6",
                value=f"{prediction:.4f}",
                delta=None
            )
        
        with metric_col2:
            # Interpretation
            if prediction < -2:
                severity = "🔴 Extreme Drought"
            elif prediction < -1.5:
                severity = "🟠 Severe Drought"
            elif prediction < -1:
                severity = "🟡 Moderate Drought"
            elif prediction < 0:
                severity = "🟢 Mild Drought"
            else:
                severity = "🔵 Wet Conditions"
            
            st.metric(
                label="Condition Severity",
                value=severity
            )
        
        st.markdown("---")
        
        # Additional info
        st.info("""
        **SPEI6 Interpretation:**
        - **< -2:** Extreme Drought
        - **-2 to -1.5:** Severe Drought
        - **-1.5 to -1:** Moderate Drought
        - **-1 to 0:** Mild Drought
        - **≥ 0:** Normal/Wet Conditions
        """)
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            summary_df = pd.DataFrame({
                "Feature": list(converted_inputs.keys()),
                "Value": list(converted_inputs.values())
            }).sort_values("Feature")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Early Fusion XGBoost Model | SPEI6 Prediction System</p>
    <p>Final Year Project - Drought Forecasting</p>
</div>
""", unsafe_allow_html=True)
