"""
Early Fusion XGBoost Model Training Script
Trains and saves the model for use in the prediction UI
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================================
# 1. PATH SETUP
# ================================
BASE = "c:\\Users\\hp\\OneDrive\\Desktop\\final_year_project\\fusion_project"
os.makedirs(f"{BASE}\\models\\early", exist_ok=True)
os.makedirs(f"{BASE}\\results\\predictions", exist_ok=True)
os.makedirs(f"{BASE}\\results\\metrics", exist_ok=True)
os.makedirs(f"{BASE}\\results\\feature_importance", exist_ok=True)

CSV_PATH = "c:\\Users\\hp\\OneDrive\\Desktop\\final_year_project\\datasets\\early_fusion_dataset.csv"

# ================================
# 2. LOAD DATA
# ================================
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Original shape: {df.shape}")

df["valid_time"] = pd.to_datetime(df["valid_time"])
df["year"] = df["valid_time"].dt.year
df["month"] = df["valid_time"].dt.month

# ================================
# 3. LAG FEATURE ENGINEERING
# ================================
lag_features = [
    "SPEI6_new",
    "NDVI_mean", "VCI", "TCI", "VHI",
    "LST_mean_C",
    "tp", "ssrd", "t2m", "d2m", "swvl2", "swvl3"
]

lag_steps = [1, 2, 3]

df = df.sort_values(["latitude", "longitude", "valid_time"])

print("Engineering lag features...")
for feature in lag_features:
    for lag in lag_steps:
        df[f"{feature}_lag{lag}"] = (
            df.groupby(["latitude", "longitude"])[feature].shift(lag)
        )

df = df.dropna().reset_index(drop=True)
print(f"After adding lag features: {df.shape}")

# ================================
# 4. DROP UNUSED COLUMNS
# ================================
drop_cols = ["valid_time", "SPEI3_new"]
df = df.drop(columns=drop_cols, errors="ignore")

# ================================
# 5. TEMPORAL SPLIT
# ================================
train_df = df[df["year"] <= 2015]
val_df   = df[(df["year"] > 2015) & (df["year"] <= 2018)]
test_df  = df[df["year"] > 2018]

print(f"Train: {train_df.shape}")
print(f"Val: {val_df.shape}")
print(f"Test: {test_df.shape}")

TARGET = "SPEI6_new"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# Save feature names for prediction UI
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, f"{BASE}\\models\\early\\feature_names.pkl")
print(f"Saved {len(feature_names)} feature names")

# ================================
# 6. TRAIN XGBOOST (EARLY FUSION)
# ================================
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ================================
# 7. TEST EVALUATION
# ================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("EARLY FUSION – XGBOOST (TEST PERFORMANCE)")
print("-"*50)
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print("="*50)

# ================================
# 8. SAVE TRAINED MODEL
# ================================
model_path = f"{BASE}\\models\\early\\xgboost_early_fusion.pkl"
joblib.dump(model, model_path)
print(f"\n✓ Saved model → {model_path}")

# ================================
# 9. SAVE PREDICTIONS
# ================================
pred_df = pd.DataFrame({
    "latitude": test_df["latitude"].values,
    "longitude": test_df["longitude"].values,
    "year": test_df["year"].values,
    "month": test_df["month"].values,
    "fusion": "early",
    "model": "xgboost_lag",
    "y_true": y_test.values,
    "y_pred": y_pred
})

pred_path = f"{BASE}\\results\\predictions\\early_xgboost.csv"
pred_df.to_csv(pred_path, index=False)
print(f"✓ Saved predictions → {pred_path}")

# ================================
# 10. SAVE METRICS
# ================================
metrics_df = pd.DataFrame([{
    "fusion": "early",
    "model": "xgboost_lag",
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2
}])

metrics_path = f"{BASE}\\results\\metrics\\metrics.csv"
if os.path.exists(metrics_path):
    existing_metrics = pd.read_csv(metrics_path)
    metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)

metrics_df.to_csv(metrics_path, index=False)
print(f"✓ Saved metrics → {metrics_path}")

# ================================
# 11. FEATURE IMPORTANCE
# ================================
importance = model.get_booster().get_score(importance_type="gain")

imp_df = pd.DataFrame({
    "feature": list(importance.keys()),
    "importance": list(importance.values())
}).sort_values("importance", ascending=False)

imp_path = f"{BASE}\\results\\feature_importance\\early_xgboost_importance.csv"
imp_df.to_csv(imp_path, index=False)
print(f"✓ Saved feature importance → {imp_path}")

print("\n✓ Training completed successfully!")
print(f"Use the model in the Streamlit app for predictions.")
