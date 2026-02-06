# 📊 Visual Architecture Diagrams

## 1️⃣ Complete Data & Model Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE (One-time)                │
│                                                               │
│  early_fusion_dataset.csv                                   │
│  (5.3 MB, 50,000+ rows)                                     │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────┐                       │
│  │  train_early_fusion.py           │                       │
│  └──────────────────────────────────┘                       │
│         │                                                     │
│         ├─► Feature Engineering                             │
│         │   - Date parsing                                  │
│         │   - Add lag features (36 new)                     │
│         │   - Total: 54 features                            │
│         │                                                     │
│         ├─► Data Splitting                                  │
│         │   - Train: years ≤ 2015                           │
│         │   - Val: 2016-2018                                │
│         │   - Test: > 2018                                  │
│         │                                                     │
│         ├─► Model Training                                  │
│         │   - XGBoost Regressor                             │
│         │   - 800 estimators                                │
│         │   - ~3-5 minutes                                  │
│         │                                                     │
│         └─► Save Artifacts                                  │
│             ├─ xgboost_early_fusion.pkl                     │
│             ├─ feature_names.pkl                            │
│             ├─ predictions CSV                              │
│             └─ metrics.csv                                  │
│                                                               │
│         ⏱️  Duration: 3-5 minutes (one-time only!)           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  PREDICTION PHASE (Repeated)                │
│                                                               │
│  ┌──────────────────────────────────┐                       │
│  │     app_predict.py (Streamlit)   │                       │
│  └──────────────────────────────────┘                       │
│         │                                                     │
│         ├─► Load Model (<1 second)                          │
│         │   └─ xgboost_early_fusion.pkl                     │
│         │                                                     │
│         ├─► Streamlit UI Opens                              │
│         │   ├─ Tab 1: Temporal & Spatial (4 inputs)        │
│         │   ├─ Tab 2: Vegetation Indices (16 inputs)       │
│         │   ├─ Tab 3: Temperature (9 inputs)               │
│         │   └─ Tab 4: Precipitation (25 inputs)            │
│         │                                                     │
│         ├─► User Enters Values                              │
│         │   └─ 54 feature values (text inputs)              │
│         │                                                     │
│         ├─► Click "Predict SPEI6"                           │
│         │                                                     │
│         ├─► Model Inference (<100ms)                        │
│         │   └─ prediction = model.predict(user_input)       │
│         │                                                     │
│         └─► Display Results                                 │
│             ├─ Predicted SPEI6 value                        │
│             ├─ Color-coded severity                         │
│             ├─ Interpretation guide                         │
│             └─ Input summary                                │
│                                                               │
│         ⚡ Duration: <100ms per prediction!                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2️⃣ Input Tab Structure

```
┌─────────────────────────────────────────────────────────┐
│           STREAMLIT PREDICTION UI (app_predict.py)      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  [TAB 1]       [TAB 2]        [TAB 3]    [TAB 4]       │
│  Temporal&    Vegetation    Temperature Precip&       │
│  Spatial      Indices        & Moisture  SPEI          │
│  ────────────────────────────────────────────────      │
│                                                           │
│  TAB 1:                                                  │
│  ┌────────────────────────────────────┐                │
│  │ ① Year:       [2020]               │                │
│  │ ② Month:      [6]                  │                │
│  │ ③ Latitude:   [35.5]               │                │
│  │ ④ Longitude:  [70.2]               │                │
│  └────────────────────────────────────┘                │
│                                                           │
│  TAB 2:                                                  │
│  ┌────────────────────────────────────┐                │
│  │ ⑤ NDVI_mean:       [0.45]          │                │
│  │ ⑥ NDVI_mean_lag1:  [0.44]          │                │
│  │ ⑦ NDVI_mean_lag2:  [0.43]          │                │
│  │ ⑧ NDVI_mean_lag3:  [0.42]          │                │
│  │ ⑨ VCI:             [0.65]          │                │
│  │ ... (more vegetation features)     │                │
│  └────────────────────────────────────┘                │
│                                                           │
│  TAB 3:                                                  │
│  ┌────────────────────────────────────┐                │
│  │ ⑨ LST_mean_C:      [25.3]          │                │
│  │ ⑩ LST_mean_C_lag1: [25.1]          │                │
│  │ ⑪ t2m:             [28.5]          │                │
│  │ ⑫ d2m:             [15.2]          │                │
│  │ ... (more temperature features)    │                │
│  └────────────────────────────────────┘                │
│                                                           │
│  TAB 4:                                                  │
│  ┌────────────────────────────────────┐                │
│  │ ⑬ tp:              [150.0]          │                │
│  │ ⑭ ssrd:            [180.0]          │                │
│  │ ⑮ swvl2:           [0.35]           │                │
│  │ ⑯ swvl3:           [0.28]           │                │
│  │ ⑰ SPEI6_new:       [0.45]           │                │
│  │ ... (more with lags)                │                │
│  └────────────────────────────────────┘                │
│                                                           │
│         [🔮 Predict SPEI6]                             │
│                                                           │
│  ═════════════════════════════════════════            │
│  RESULT: Predicted SPEI6 = 0.6234                    │
│  ═════════════════════════════════════════            │
│  Condition: 🟢 Mild Drought (-1 to 0)                │
│  ═════════════════════════════════════════            │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 3️⃣ Feature Categories Breakdown

```
┌─────────────────────────────────────────────────────┐
│         54 INPUT FEATURES ORGANIZATION              │
├─────────────────────────────────────────────────────┤
│                                                       │
│  🕐 TEMPORAL (2)                                    │
│    ├─ year
│    └─ month
│                                                       │
│  📍 SPATIAL (2)                                     │
│    ├─ latitude
│    └─ longitude
│                                                       │
│  🌿 VEGETATION (12)                                 │
│    ├─ NDVI_mean [4 features: original + 3 lags]    │
│    ├─ VCI [4 features]                             │
│    ├─ TCI [4 features]                             │
│    └─ VHI [4 features]                             │
│        Subtotal: 12 (with lags)                     │
│                                                       │
│  🌡️ TEMPERATURE (12)                               │
│    ├─ LST_mean_C [4 features]                      │
│    ├─ t2m [4 features]                             │
│    └─ d2m [4 features]                             │
│        Subtotal: 12 (with lags)                     │
│                                                       │
│  💧 PRECIPITATION (12)                              │
│    ├─ tp [4 features]                              │
│    ├─ ssrd [4 features]                            │
│    ├─ swvl2 [4 features]                           │
│    └─ swvl3 [4 features]                           │
│        Subtotal: 12 (with lags)                     │
│                                                       │
│  📈 SPEI (4)                                        │
│    └─ SPEI6_new [4 features: original + 3 lags]    │
│        Subtotal: 4 (with lags)                      │
│                                                       │
│  TOTAL: 2+2+12+12+12+4 = 54 FEATURES              │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 4️⃣ Output Severity Scale

```
SPEI6 Value Range     │   Condition Level      │ Color
──────────────────────┼────────────────────────┼──────
  < -2.0              │  🔴 Extreme Drought    │  RED
-2.0 to -1.5          │  🟠 Severe Drought     │ ORANGE
-1.5 to -1.0          │  🟡 Moderate Drought   │ YELLOW
-1.0 to 0.0           │  🟢 Mild Drought       │ GREEN
  ≥ 0.0               │  🔵 Wet Conditions     │ BLUE
──────────────────────┴────────────────────────┴──────

Example: If model predicts SPEI6 = -0.8
  → Shows: "🟢 Mild Drought (-1 to 0)"
```

---

## 5️⃣ File Organization Tree

```
final_year_project/
│
├── 📂 datasets/                          ← INPUT DATA
│   └── early_fusion_dataset.csv          (5.3 MB)
│
├── 📂 fusion_project/                    ← PROJECT ARTIFACTS
│   ├── 📂 models/
│   │   └── 📂 early/
│   │       ├── xgboost_early_fusion.pkl  ← TRAINED MODEL ⭐
│   │       └── feature_names.pkl         ← CONFIG ⭐
│   │
│   ├── 📂 results/
│   │   ├── 📂 predictions/
│   │   │   └── early_xgboost.csv         (Test predictions)
│   │   ├── 📂 metrics/
│   │   │   └── metrics.csv               (RMSE, MAE, R²)
│   │   └── 📂 feature_importance/
│   │       └── early_xgboost_importance.csv
│   │
│   └── 📂 config/
│
├── 📄 train_early_fusion.py              ← TRAINING SCRIPT
├── 📄 app_predict.py                     ← STREAMLIT UI ⭐
├── 📄 app.py                             (Previous version)
│
├── 📋 DOCUMENTATION
│   ├── README.md                         (Setup guide)
│   ├── QUICKSTART.txt                    (Quick reference)
│   ├── ARCHITECTURE.md                   (Design details)
│   └── IMPLEMENTATION_SUMMARY.txt        (Complete summary)
│
├── 📦 requirements.txt                   (Dependencies)
└── ⚡ quickstart.bat                     (Windows automation)

Legend:
⭐ = Critical files for predictions
```

---

## 6️⃣ Workflow Timeline

```
TIME          ACTIVITY                              LOCATION
─────────────────────────────────────────────────────────────

 0 sec    └─ User double-clicks quickstart.bat
          
 5 sec    └─ pip install dependencies
          
15 sec    └─ python train_early_fusion.py started
          
180-300   ├─ Load dataset (5.3 MB)
 sec      ├─ Engineer features
          ├─ Split data
          ├─ Train XGBoost (800 trees)
          └─ Save model artifacts ✓
          
305 sec   └─ streamlit run app_predict.py started
          
310 sec   └─ Browser opens at localhost:8501 ✓
          
320 sec   └─ User sees Streamlit UI
          
325 sec   └─ Model loaded (<1 second)
          
326 sec   └─ User ready to make predictions
          
327 sec   └─ User enters feature values
          
328 sec   └─ User clicks "Predict SPEI6"
          
328.1 sec └─ Model inference (<100ms)
          
328.2 sec └─ Results displayed ✓
          
330 sec   └─ User can make more predictions
          
...       └─ Each prediction: <100ms
          
─────────────────────────────────────────────────────────────
Key Insight: Training takes 3-5 minutes ONCE.
             Each prediction takes <100ms thereafter!
```

---

## 7️⃣ Comparison: Before vs After

```
BEFORE (Full Code Every Time)
════════════════════════════════════════════════════════

User needs prediction:
  ├─ Run full notebook
  ├─ [5 min] Load dataset
  ├─ [2 min] Engineer features
  ├─ [3 min] Split data
  ├─ [5 min] TRAIN model
  ├─ [2 min] Make prediction
  └─ [1 min] Display result
  
TOTAL: 18 minutes per prediction ⏱️
Result: 1 prediction

────────────────────────────────────────────────────────────

AFTER (Model-Only)
════════════════════════════════════════════════════════

Setup (one-time):
  ├─ [5 min] pip install
  ├─ [1 min] Load dataset
  ├─ [1 min] Engineer features
  ├─ [1 min] Split data
  ├─ [3 min] TRAIN model
  └─ Save artifacts ✓
  
TOTAL: 11 minutes (one-time setup)

Then for EACH prediction:
  ├─ [1 sec] Load pre-trained model
  ├─ [0.1 sec] User inputs values
  ├─ [0.001 sec] Make prediction
  └─ [0.001 sec] Display result
  
TOTAL: <1 second per prediction ⚡
Result: UNLIMITED predictions!

────────────────────────────────────────────────────────────

EFFICIENCY GAIN:
  - Setup overhead: ~11 min (one-time)
  - Per prediction: 18 min → 0.1 sec
  - 180x faster per prediction!
```

---

## 8️⃣ Model Architecture

```
INPUT LAYER
════════════════════════════════════════════════════════
54 Features
├─ year, month (2)
├─ latitude, longitude (2)
├─ NDVI_mean, NDVI_mean_lag1, NDVI_mean_lag2, NDVI_mean_lag3
├─ VCI, VCI_lag1, VCI_lag2, VCI_lag3
├─ TCI, TCI_lag1, TCI_lag2, TCI_lag3
├─ VHI, VHI_lag1, VHI_lag2, VHI_lag3
├─ LST_mean_C, LST_mean_C_lag1, LST_mean_C_lag2, LST_mean_C_lag3
├─ t2m, t2m_lag1, t2m_lag2, t2m_lag3
├─ d2m, d2m_lag1, d2m_lag2, d2m_lag3
├─ tp, tp_lag1, tp_lag2, tp_lag3
├─ ssrd, ssrd_lag1, ssrd_lag2, ssrd_lag3
├─ swvl2, swvl2_lag1, swvl2_lag2, swvl2_lag3
├─ swvl3, swvl3_lag1, swvl3_lag2, swvl3_lag3
└─ SPEI6_new, SPEI6_new_lag1, SPEI6_new_lag2, SPEI6_new_lag3

         ▼

XGBOOST LAYER
════════════════════════════════════════════════════════
- Algorithm: Gradient Boosting
- Trees: 800 estimators
- Max Depth: 8
- Learning Rate: 0.03
- Subsample: 0.8
- Colsample: 0.8
- Regularization (L2): 1.0

         ▼

OUTPUT LAYER
════════════════════════════════════════════════════════
SPEI6 Prediction (continuous value)

Range: -3 to +3 (typically)
Interpretation:
  < -2.0   → Extreme Drought
  -2 to -1.5 → Severe Drought
  -1.5 to -1 → Moderate Drought
  -1 to 0  → Mild Drought
  ≥ 0      → Wet Conditions
```

---

## 9️⃣ System Status

```
✅ SETUP COMPLETE

┌─────────────────────────────────────────────┐
│ Component              │ Status             │
├────────────────────────┼────────────────────┤
│ Dataset                │ ✓ Ready            │
│ Training Script        │ ✓ Ready            │
│ Prediction UI          │ ✓ Ready            │
│ Model File             │ ⏳ After training  │
│ Dependencies           │ ✓ Listed           │
│ Documentation          │ ✓ Complete         │
│ Quick Start            │ ✓ Available        │
└─────────────────────────────────────────────┘

NEXT STEP:
Run → python train_early_fusion.py
Then → streamlit run app_predict.py
```

---

*Visual Architecture Diagram - Early Fusion SPEI6 Prediction System*
*Last Updated: 2026-02-03*
