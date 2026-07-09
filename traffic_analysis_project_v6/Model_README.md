# 🚦 Egypt Traffic Intelligence System

**Real-time Cairo traffic monitoring, machine-learning-based congestion classification, and a live dashboard** — built on the TomTom Traffic API, a self-healing ingestion pipeline, and an auto-retraining ML backend.

---

## 📁 Project Structure

```
traffic_analysis_project/
├── app/
│   ├── streamlit_app.py       # Dashboard entry point
│   ├── style.css              # Dark glassmorphism theme (Syne + IBM Plex Mono)
│   └── logo.svg                # Sidebar logo
├── config/
│   └── config.py             # Paths, TomTom API settings, locations, training config
├── data/
│   ├── raw/                  # Timestamped raw pulls, e.g. traffic_2026_07_03_14_22.csv
│   └── processed/
│       └── traffic_clean.csv # Cleaned + feature-engineered dataset
├── ingestion/
│   └── data_ingestion.py     # TomTom API polling + simulation fallback
├── models/
│   ├── train_models.py       # Unified training pipeline (validate → clean → engineer → train → compare)
│   ├── best_model.pkl        # Best-performing classifier (auto-selected)
│   ├── scaler.pkl            # StandardScaler fit on training data
│   ├── loc_encoder.pkl       # LabelEncoder for location names
│   ├── target_encoder.pkl    # LabelEncoder for traffic_status
│   └── meta.json             # Model name, features, training timestamp
├── outputs/
│   ├── plots/                # Comparison charts, confusion matrices, feature importance
│   └── reports/              # model_comparison.csv, training_report.txt
├── api.py                    # FastAPI inference service
├── requirements.txt
└── README.md
```

---

## 🧭 System Overview

```
TomTom Traffic API
       │
       ▼
ingestion/data_ingestion.py  ──►  data/raw/traffic_<timestamp>.csv
       │  (timestamped, append-only, never overwritten)
       ▼
train_models.py
   ├── validate raw data (schema, ranges, timestamp formats)
   ├── clean + engineer features  ──►  data/processed/traffic_clean.csv
   ├── encode locations & target
   ├── train/test split (stratified) + StandardScaler
   ├── optional SMOTE class balancing
   ├── train & cross-validate 6–7 classifiers
   ├── select best model by weighted F1
   └── save models/*.pkl + meta.json + outputs/reports
       │
       ├──► api.py (FastAPI)         → /predict, /locations
       └──► app/ (Streamlit)         → live dashboard, auto-reloads model on new artifacts
```

Re-running `train_models.py` on a schedule (cron, Task Scheduler, or a background process) keeps the deployed model current with incoming traffic patterns — the dashboard hot-reloads automatically whenever new model artifacts are written, no restart required.

---

## 🗺️ Coverage

17 monitored Cairo road segments/junctions, including Ring Road (Maadi, Marg, Moneeb), October Bridge, 26th July Corridor, Tahrir Square, Corniche El Nil, Salah Salem, Autostrad–Sheraton, Rod El Farag Axis, and others — full list and coordinates in `config/config.py`.

---

## 🔍 Feature Design

The model only uses information a user actually has **before** starting a trip — no real-time inputs like current speed or vehicle count, which would leak the answer.

| Feature | Why it's included |
|---|---|
| `hour`, `minute` | Planned departure time |
| `day_of_week`, `is_weekend` | Known in advance |
| `is_peak_hour` | Flags 7–10 AM / 4–7 PM rush windows |
| `lat`, `lon` | Fixed per road |
| `loc_encoded` | Encoded location selected from dropdown |

`speed` and `vehicle_count` are collected during ingestion (for EDA/history) but excluded from the feature set, since they're both unknown pre-trip and would directly leak the congestion label.

---

## 🤖 Machine Learning Pipeline (`train_models.py`)

**Data quality handling:**
- Robust parsing of mixed ISO-8601 timestamp formats (an earlier version silently dropped ~32% of rows as `NaT` — now fixed with explicit multi-format parsing and validation).
- Schema and range validation before cleaning.
- SMOTE oversampling on the training split only, to correct class imbalance (Congested is typically under-represented).

**Models trained & compared:**

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable |
| Random Forest | Ensemble, robust to noise |
| Gradient Boosting | Sequential boosting |
| KNN | Distance-based |
| SVM (RBF) | Kernel-based |
| XGBoost | If installed |

**Selection metric:** highest weighted F1-score across 5-fold cross-validation (not raw accuracy, to account for class imbalance).

**Outputs:** `models/best_model.pkl`, `scaler.pkl`, `loc_encoder.pkl`, `target_encoder.pkl`, `meta.json`, plus comparison tables/plots in `outputs/`.

---

## 🎛️ Dashboard (`app/streamlit_app.py`)

Dark, glassmorphism-styled UI (`style.css`, Syne + IBM Plex Mono, animated gradient background) with a branded sidebar (`logo.svg`) and `streamlit_option_menu` navigation across four pages:

- **Overview** — live KPI cards (record count, locations, avg speed, % free-flow, % congested), status distribution and trend charts
- **EDA & Viz** — speed distributions, hourly/day-of-week trends, per-location breakdowns, map view via `pydeck`
- **Prediction** — pre-trip form (location + departure time only, no live speed/vehicle count), comparison cards across times of day, best-hour badge
- **Feature Audit** — rationale for kept/removed features

**Data source:** reads live from an Azure SQL table (`dbo.traffic`) via SQLAlchemy by default, with column aliasing (`region`→`location_name`, `avg_speed`→`speed`) to match the Stream Analytics output schema; a sidebar CSV uploader lets you override it with a local file. If neither is reachable, the app degrades gracefully to an empty frame instead of crashing.

**Hot-reload:** both the dataset and the model artifacts are cached by file-modification timestamp (`@st.cache_data` / `@st.cache_resource` keyed on mtime), and the app auto-reruns every 60 seconds — so new data or a freshly retrained model are picked up automatically, with no manual restart.

---

## 🔌 API (`api.py`)

FastAPI service exposing:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/predict` | POST | Predict traffic status for a location + time |
| `/locations` | GET | List all valid location names |

**`/predict` request body:**
```json
{
  "location": "Ring Road - Maadi",
  "hour": 8,
  "minute": 30,
  "day_of_week": 2,
  "lat": 29.9544,
  "lon": 31.2858
}
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (live API or simulation)
python ingestion/data_ingestion.py --rows 5000
python ingestion/data_ingestion.py --simulate      # no API key needed

# 3. Train models
python train_models.py --data data/processed/traffic_clean.csv
#   --no-smote     Disable SMOTE oversampling
#   --cv 5         Number of cross-validation folds

# 4. Launch the dashboard (reads from Azure SQL by default, or upload a CSV in the sidebar)
streamlit run app/streamlit_app.py

# 5. (Optional) Serve predictions via API
uvicorn api:app --reload
```

---

## 💡 Future Improvements

1. **Weather data** — rain/fog materially affects Cairo traffic and is fetchable pre-trip.
2. **Public holidays / events** — Eid, national holidays, football matches cause atypical congestion.
3. **Historical road averages** — e.g. "average speed on Ring Road at 8 AM on Thursdays."
4. **Seasonal features** — Ramadan, school holidays, and winter patterns differ meaningfully.
5. **Route-level metadata** — intersection count, road type (highway vs. arterial).

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
streamlit-option-menu
pydeck
sqlalchemy
fastapi
uvicorn
joblib
requests
```

---

*Egypt Traffic Intelligence System — Cairo real-time traffic classification*
