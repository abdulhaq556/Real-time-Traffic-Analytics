
import os
import sys
from pathlib import Path

# ── Project root resolution ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = str(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ══════════════════════════════════════════════════════════════════════════════
PATHS = {
    # Raw data directory — one timestamped file per ingestion run (never overwritten)
    "raw_dir":       os.path.join(BASE_DIR, "data", "raw"),
    # Cleaned + feature-engineered data ready for training
    "processed_dir": os.path.join(BASE_DIR, "data", "processed"),
    # Convenience processed paths used by training pipeline
    "clean_csv":     os.path.join(BASE_DIR, "data", "processed", "traffic_clean.csv"),
    "train_csv":     os.path.join(BASE_DIR, "data", "processed", "traffic_train.csv"),
    # Outputs
    "models_dir":    os.path.join(BASE_DIR, "models"),
    "plots_dir":     os.path.join(BASE_DIR, "outputs", "plots"),
    "reports_dir":   os.path.join(BASE_DIR, "outputs", "reports"),
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
COLLECTION = {
    "api_key":      "OEZCfjkL5k4imwQiQdjLVEWLI981FV1s",
    "base_url":     "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
    "unit":         "KMPH",
    "zoom":         10,
    "style":        "absolute",

    "rows_to_fetch":        5000,
    "rows_per_location":    None,

    "polling_interval_sec": 1.0,
    "log_every_n":          500,

    "request_timeout_sec":  10,
    "max_retries":          3,
    "retry_backoff_sec":    2.0,

    "simulation_mode":      False,
}


# ══════════════════════════════════════════════════════════════════════════════
# CAIRO ROAD LOCATIONS  — single source of truth for all modules
# ══════════════════════════════════════════════════════════════════════════════
LOCATIONS = [
    # ── Original 9 locations ──────────────────────────────────────────────
    {"name": "Ring Road - Maadi",      "lat": 29.9544, "lon": 31.2858},
    {"name": "October Bridge",          "lat": 30.0526, "lon": 31.2372},
    {"name": "26th July Corridor",      "lat": 30.0381, "lon": 31.0264},
    {"name": "Abbas El Akkad",          "lat": 30.0631, "lon": 31.3341},
    {"name": "Galaa Square",            "lat": 30.0398, "lon": 31.2188},
    {"name": "Salah Salem",             "lat": 30.0645, "lon": 31.2820},
    {"name": "Autostrad - Sheraton",    "lat": 30.1065, "lon": 31.3711},
    {"name": "Tahrir Square",           "lat": 30.0444, "lon": 31.2357},
    {"name": "Corniche El Nil",         "lat": 30.0263, "lon": 31.2268},
    # ── New 8 locations ───────────────────────────────────────────────────
    {"name": "Mosheer Tantawy Axis",    "lat": 30.0242, "lon": 31.3486},
    {"name": "Rod El Farag Axis",       "lat": 30.0881, "lon": 31.2185},
    {"name": "Ring Road - Marg",        "lat": 30.1554, "lon": 31.3323},
    {"name": "Ring Road - Moneeb",      "lat": 29.9881, "lon": 31.2227},
    {"name": "Gamaat El Dowal St",      "lat": 30.0543, "lon": 31.2005},
    {"name": "Faisal Street",           "lat": 30.0051, "lon": 31.1574},
    {"name": "North 90th Street",       "lat": 30.0308, "lon": 31.4721},
    {"name": "South 90th Street",       "lat": 30.0195, "lon": 31.4326},
]

# Derived lookup: name → (lat, lon)  — used by Streamlit and prediction code
LOCATION_COORDS: dict[str, tuple[float, float]] = {
    loc["name"]: (loc["lat"], loc["lon"]) for loc in LOCATIONS
}

# Ordered list of location names — used for dropdowns / encoders
LOCATION_NAMES: list[str] = [loc["name"] for loc in LOCATIONS]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
TRAINING = {
    "test_size":        0.20,
    "random_state":     42,
    "cv_folds":         5,
    "use_smote":        True,
    "selection_metric": "f1",
}

FEATURE_COLS = [
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "lat",
    "lon",
    "loc_encoded",
]
TARGET_COL = "traffic_status"


# ══════════════════════════════════════════════════════════════════════════════
# STYLING  (shared across plots)
# ══════════════════════════════════════════════════════════════════════════════
STYLE = {
    "palette":  ["#00d4ff", "#7b2fff", "#ff6b6b", "#ffd93d", "#6bcb77"],
    "bg_color": "#0e1117",
    "card_bg":  "#1c1f26",
}


# ── Auto-create required directories on import ─────────────────────────────────
for _path in PATHS.values():
    if not str(_path).endswith(".csv"):
        os.makedirs(_path, exist_ok=True)
