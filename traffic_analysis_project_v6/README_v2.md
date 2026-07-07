# 🚦 Egypt Real-Time Traffic Intelligence System — v2.0

**Business Analytics & Machine Learning** project built on Cairo traffic stream data.  
Refactored for simplicity, correctness, and real-world usability.

---

## 📁 Simplified Project Structure

```
traffic_analysis_project/
├── data/
│   ├── traffic.csv               ← Raw data
│   └── traffic_clean.csv         ← Cleaned data
├── src/
│   ├── data_preparation.py       ← Load, clean, EDA
│   └── clustering.py             ← KMeans + DBSCAN
├── train_models.py               ← ✅ NEW: Unified training pipeline (all models)
├── streamlit_app.py              ← ✅ UPDATED: Dashboard (no vehicle_count input)
├── models/
│   ├── best_model.pkl            ← ✅ Single best model (auto-selected)
│   ├── scaler.pkl                ← StandardScaler
│   ├── loc_encoder.pkl           ← LabelEncoder (location names)
│   ├── target_encoder.pkl        ← LabelEncoder (traffic_status)
│   └── meta.json                 ← Model metadata (name, features, timestamp)
├── outputs/
│   ├── plots/                    ← All visualisation images
│   └── reports/
│       ├── model_comparison.csv  ← ✅ Full model comparison table
│       └── training_report.txt   ← Full training report
├── requirements.txt
└── README.md
```

**Removed files** (replaced by unified pipeline):
- ~~`src/preprocessing.py`~~ → merged into `train_models.py`
- ~~`src/classification.py`~~ → merged into `train_models.py`
- ~~`src/evaluation.py`~~ → merged into `train_models.py`
- ~~`models/logistic_regression.pkl`~~ → only best model saved
- ~~`models/random_forest.pkl`~~ → only best model saved
- ~~`models/xgboost.pkl`~~ → only best model saved
- ~~`models/best_regressor.pkl`~~ → regression handled separately

---

## 🗂️ Dataset

| Column | Description |
|---|---|
| `timestamp` | Reading datetime |
| `location_name` | Cairo road/junction name |
| `coordinates` | Lat, Lon |
| `speed` | Average speed in km/h |
| `vehicle_count` | Number of vehicles observed |

**Locations:** Ring Road – Maadi, October Bridge, 26th July Corridor, Abbas El Akkad, Galaa Square

---

## 🔍 Feature Audit — What Changed & Why

### ❌ REMOVED Features

| Feature | Reason |
|---|---|
| `vehicle_count` | **Not realistic.** A user at home cannot know real-time vehicle counts. Removing it makes the system genuinely usable before travel. |
| `speed` | **Data leakage.** Speed is the direct proxy for congestion — using it as a feature means the model is essentially told the answer. Also unknown before travel. |
| `traffic_density` | **Derived from removed features** (`vehicle_count / speed`). Removed automatically. |

### ✅ KEPT Features

| Feature | Justification |
|---|---|
| `hour` | User knows their planned departure time |
| `minute` | Same rationale |
| `day_of_week` | User knows what day it is |
| `is_weekend` | Derived from day_of_week — always known |
| `is_peak_hour` | Derived from hour — flags 7–9 AM and 4–7 PM rush windows |
| `lat`, `lon` | Fixed per road; user picks from a dropdown |
| `loc_encoded` | Label-encoded location — same spatial info |

---

## 🤖 Unified ML Pipeline — `train_models.py`

### Models Trained & Compared

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable non-linear |
| Random Forest | Ensemble, robust to noise |
| Gradient Boosting | Sequential boosting |
| KNN | Distance-based |
| SVM (RBF) | Kernel-based, good for small datasets |
| XGBoost | Fast gradient boosting (if installed) |

### Selection Criterion
Best model = **highest weighted F1-Score** (not raw accuracy, because traffic classes may be imbalanced — Congested can be under-represented).

### Pipeline Steps
1. Load & clean CSV
2. Engineer temporal features (`hour`, `is_weekend`, `is_peak_hour`, etc.)
3. Encode location names, encode target labels
4. Train/test split (80/20, stratified)
5. StandardScaler fit on training set only
6. Optional SMOTE on training set for class balance
7. Train all 6–7 models with 5-fold cross-validation
8. Compare: Accuracy, Precision, Recall, F1, ROC-AUC, CV mean±std
9. Save only the best model + encoders + metadata

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unified training pipeline
python train_models.py --data data/traffic_clean.csv

# Options:
#   --no-smote    Disable SMOTE oversampling
#   --cv 5        Number of cross-validation folds

# 3. Launch dashboard
streamlit run streamlit_app.py
```

---

## 🎨 Dashboard Pages

| Page | Content |
|---|---|
| 🏠 Overview | KPI cards, status distribution, model status card |
| 📊 EDA | Speed distributions, hourly trends, day-of-week breakdown, location analysis |
| 🔵 Clustering | KMeans clusters (hour vs speed), saved plots |
| 🤖 Model Training | Comparison table (from CSV), metric bar chart, heatmap, confusion matrices, feature importances |
| 📈 Regression | Actual vs predicted speed, predictions CSV download |
| 🔮 Live Prediction | **Pre-trip form** — location + departure time only (no vehicle count!) |
| ℹ️ Feature Audit | Full justification of kept/removed features + improvement recommendations |

---

## 💡 Recommendations for Future Improvement

1. **Weather data** — Rain/fog significantly affects Cairo traffic; fetchable via free API before travel
2. **Public holidays / events** — Eid, national days, football matches cause atypical congestion
3. **Historical road averages** — "Average speed on Ring Road at 8 AM on Thursdays" — computable from past data, fully pre-trip
4. **Seasonal feature** — Ramadan, summer school holidays, winter patterns differ meaningfully
5. **Route-level metadata** — Number of intersections, road type (highway vs arterial)

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
xgboost
streamlit
joblib
```

---

*Egypt Traffic Analytics Project — Refactored 2026*
