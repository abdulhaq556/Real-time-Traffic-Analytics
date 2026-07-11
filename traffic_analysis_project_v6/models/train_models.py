import os
import sys
import time
import warnings
import argparse
import json
import logging
from datetime import datetime
from typing import Optional

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("xgboost not installed — XGBClassifier skipped.")

# ── Central config ─────────────────────────────────────────────────────────────
from config.config import PATHS, TRAINING, FEATURE_COLS, TARGET_COL, LOCATION_NAMES

MODELS_DIR  = PATHS["models_dir"]
PLOTS_DIR   = PATHS["plots_dir"]
REPORTS_DIR = PATHS["reports_dir"]

for _d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Cairo geographic bounding box (generous margin) ────────────────────────────
_CAIRO_LAT_MIN, _CAIRO_LAT_MAX = 29.7,  30.4
_CAIRO_LON_MIN, _CAIRO_LON_MAX = 30.8,  31.8

# ── Valid traffic status labels ────────────────────────────────────────────────
_VALID_STATUSES = {"Free_Flow", "Moderate", "Congested"}

# ── Minimum rows required to attempt training ──────────────────────────────────
_MIN_ROWS_FOR_TRAINING = 50

# ── Feature audit (printed at the start of every training run) ─────────────────
FEATURE_AUDIT = """
FEATURE AUDIT — Pre-Trip Realism Analysis
==========================================
KEPT:
  hour          ✅ User knows when they plan to travel
  minute        ✅ Same rationale as hour
  day_of_week   ✅ User knows what day it is
  is_weekend    ✅ Derived from day_of_week; always known
  is_peak_hour  ✅ Derived from hour; captures rush-hour patterns
  lat / lon     ✅ User selects a road from a dropdown; coords are fixed per road
  loc_encoded   ✅ Same location info, encoded for models

REMOVED:
  vehicle_count   ❌ User has NO way to know real-time vehicle counts before leaving
  speed           ❌ Speed IS what determines congestion (target leak proxy);
                     also unknown before travelling
  traffic_density ❌ Derived from removed features
"""


# ══════════════════════════════════════════════════════════════════════════════
# VERBOSE REPORTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pct(part, whole):
    return (part / whole * 100) if whole else 0.0


def print_validation_summary(df: pd.DataFrame, findings: dict) -> str:
    n_rows = len(df)
    n_cols = len(df.columns)

    lines = []
    lines.append("\n## Validation Summary")
    lines.append(f"\nRows: {n_rows:,}")
    lines.append(f"Columns: {n_cols}")

    null_counts = findings.get("null_counts", {})
    non_zero_nulls = {c: v for c, v in null_counts.items() if v > 0}
    lines.append("\nMissing Values:")
    if non_zero_nulls:
        for col, cnt in non_zero_nulls.items():
            lines.append(f"{col:<20s}{cnt:>6,} ({_pct(cnt, n_rows):.2f}%)")
    else:
        lines.append("None")

    if "bad_timestamps" in findings:
        lines.append(f"\nInvalid Timestamps: {findings['bad_timestamps']:,}")

    unknown_locs = findings.get("unknown_locations", [])
    lines.append("\nUnknown Locations:")
    lines.append(f"{len(unknown_locs)}")
    if unknown_locs:
        lines.append(f"Names: {', '.join(map(str, unknown_locs))}")

    lines.append(f"\nDuplicate Rows:\n{findings.get('duplicate_rows', 0):,}")

    inf_map = findings.get("infinite_values", {})
    total_inf = sum(inf_map.values()) if inf_map else 0
    lines.append(f"\nInfinite Values: {total_inf:,}")
    if total_inf:
        for col, cnt in inf_map.items():
            if cnt:
                lines.append(f"  {col:<20s}{cnt:>6,}")

    bad_lat = findings.get("out_of_range_lat", 0)
    bad_lon = findings.get("out_of_range_lon", 0)
    lines.append(f"\nInvalid Coordinates: {bad_lat + bad_lon:,}  (lat={bad_lat:,}, lon={bad_lon:,})")

    lines.append(f"\nInvalid Traffic Status Values: {findings.get('invalid_traffic_status', 0):,}")
    lines.append(f"Negative Vehicle Counts: {findings.get('negative_vehicle_count', 0):,}")
    lines.append(f"Invalid Speeds: {findings.get('non_positive_speed', 0):,}")

    warning_total = (
        len(non_zero_nulls) + findings.get("bad_timestamps", 0)
        + len(unknown_locs) + findings.get("duplicate_rows", 0)
        + total_inf + bad_lat + bad_lon
        + findings.get("invalid_traffic_status", 0)
        + findings.get("negative_vehicle_count", 0)
        + findings.get("non_positive_speed", 0)
    )
    verdict = "PASS" if warning_total == 0 else "PASS WITH WARNINGS"
    lines.append(f"\nValidation Result:\n{verdict}")

    report = "\n".join(lines)
    print(report)
    return report


def print_cleaning_summary(n_before: int, n_after: int, removed: dict) -> str:
    lines = []
    lines.append("\n## Cleaning Summary")
    lines.append(f"\nRows Before Cleaning: {n_before:,}")
    lines.append(f"Rows After Cleaning : {n_after:,}")

    lines.append("\nRemoved:")
    any_removed = False
    for label, cnt in removed.items():
        if cnt:
            any_removed = True
            lines.append(f"* {cnt:,} {label}")
    if not any_removed:
        lines.append("* None")

    total_removed = n_before - n_after
    lines.append(f"\nTotal Removed:\n{total_removed:,} rows")

    retention = _pct(n_after, n_before)
    lines.append(f"\nRetention Rate:\n{retention:.2f}%")

    report = "\n".join(lines)
    print(report)
    return report


def print_feature_snapshot(df: pd.DataFrame, generated_features: list) -> None:
    print("\nGenerated Features:")
    for feat in generated_features:
        print(feat)

    present = [f for f in generated_features if f in df.columns]
    print("\n## Feature Snapshot:\n")
    print(df[present].head(5).to_string(index=False))

    print("\nData Types:")
    print(df[present].dtypes.to_string())

    print("\nNull Counts After Feature Engineering:")
    nulls = df[present].isnull().sum()
    if nulls.sum() == 0:
        print("None")
    else:
        print(nulls[nulls > 0].to_string())


def print_location_audit(df: pd.DataFrame) -> None:
    configured = set(LOCATION_NAMES)
    found = set(df["location_name"].dropna().unique()) if "location_name" in df.columns else set()

    missing = configured - found
    new = found - configured

    print("\n## Location Audit")
    print(f"\nConfigured Locations: {len(configured)}")
    print(f"Found Locations: {len(found)}")
    print("\nMissing:")
    print("None" if not missing else ", ".join(sorted(missing)))
    print("\nNew:")
    print("None" if not new else ", ".join(sorted(new)))


def print_target_distribution(df: pd.DataFrame, target_col: str = "traffic_status") -> None:
    counts = df[target_col].value_counts()
    total = counts.sum()

    print("\n## Target Distribution")
    print()
    for label, cnt in counts.items():
        print(f"{label:<12s} {cnt:>6,} ({_pct(cnt, total):.1f}%)")

    if len(counts) >= 2 and counts.min() > 0:
        imbalance_ratio = counts.max() / counts.min()
        print(f"\nImbalance Ratio (max:min): {imbalance_ratio:.2f} : 1")


def print_split_summary(
    X_train, X_test, y_train, y_test, target_encoder: LabelEncoder,
    y_full=None,
) -> None:
    class_names = target_encoder.classes_

    print("\n## Train / Test Split")
    print(f"\nTrain Shape: {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")
    print(f"Feature Count: {X_train.shape[1]}")

    if y_full is not None:
        full_counts = pd.Series(target_encoder.inverse_transform(y_full)).value_counts()
        print("\nClass Distribution Before Split:")
        print(full_counts.to_string())

    train_counts = pd.Series(target_encoder.inverse_transform(y_train)).value_counts()
    test_counts = pd.Series(target_encoder.inverse_transform(y_test)).value_counts()

    print("\nClass Distribution After Split (Train):")
    print(train_counts.to_string())
    print("\nClass Distribution After Split (Test):")
    print(test_counts.to_string())


def print_feature_audit(df: pd.DataFrame, feature_cols: list) -> None:
    print("\n## Feature Audit")
    print(f"\n{'Column':<20s}{'Nulls':>10s}{'Unique':>10s}")

    constant_cols = []
    near_constant_cols = []
    nan_cols = []
    inf_cols = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        s = df[col]
        nulls = int(s.isnull().sum())
        unique = int(s.nunique(dropna=True))
        print(f"{col:<20s}{nulls:>10,}{unique:>10,}")

        if unique <= 1:
            constant_cols.append(col)
        elif s.dtype.kind in "if":
            top_freq = s.value_counts(normalize=True, dropna=True)
            if not top_freq.empty and top_freq.iloc[0] >= 0.95:
                near_constant_cols.append(col)

        if nulls > 0:
            nan_cols.append(col)

        if s.dtype.kind == "f" and np.isinf(s).sum() > 0:
            inf_cols.append(col)

    print("\nConstant Columns:")
    print("None" if not constant_cols else ", ".join(constant_cols))
    print("\nNear-Constant Columns (>=95% single value):")
    print("None" if not near_constant_cols else ", ".join(near_constant_cols))
    print("\nColumns With NaN:")
    print("None" if not nan_cols else ", ".join(nan_cols))
    print("\nColumns With Infinite Values:")
    print("None" if not inf_cols else ", ".join(inf_cols))


def print_smote_report(before_counts: dict, after_counts: dict = None, failure_reason: str = None,
                        nan_report: dict = None) -> None:
    print("\n## SMOTE Resampling")
    print("\nBefore SMOTE:\n")
    for label, cnt in before_counts.items():
        print(f"{label:<10s}: {cnt:,}")

    if failure_reason:
        print(f"\nSMOTE FAILED: {failure_reason}")
        if nan_report:
            print("\nColumns With NaN:")
            for col, cnt in nan_report.items():
                if cnt:
                    print(f"{col:<20s}{cnt:,}")
        return

    if after_counts:
        print("\nAfter SMOTE:\n")
        for label, cnt in after_counts.items():
            print(f"{label:<10s}: {cnt:,}")
        rows_added = sum(after_counts.values()) - sum(before_counts.values())
        print(f"\nRows Added:\n{rows_added:,}")


def print_ranking_table(results: dict, metric: str = "f1") -> str:
    ranked = sorted(results.items(), key=lambda kv: kv[1][metric], reverse=True)

    lines = ["\n## Rank   Model                 " + metric.upper()]
    for i, (name, r) in enumerate(ranked, start=1):
        lines.append(f"{i:<6d} {name:<20s}  {r[metric]:.4f}")

    winner_name = ranked[0][0]
    lines.append(f"\nWinner:\n{winner_name}")
    lines.append(f"\nReason:\nHighest weighted {metric.upper()}-score.")

    report = "\n".join(lines)
    print(report)
    return report


def print_artifact_sizes(paths: list) -> None:
    print("\n## Artifacts Saved")
    for p in paths:
        if os.path.exists(p):
            size_kb = os.path.getsize(p) / 1024
            print(f"{p}  ({size_kb:,.1f} KB)")
        else:
            print(f"{p}  (NOT FOUND)")


def print_final_report(
    input_rows: int, rows_removed: int, rows_used: int,
    feature_cols: list, target_col: str,
    models_trained: int, best_name: str, best_f1: float,
    training_time_sec: float, artifact_paths: list, status: str = "SUCCESS",
) -> str:
    lines = []
    lines.append("\n" + "=" * 48)
    lines.append("PIPELINE EXECUTION SUMMARY")
    lines.append("=" * 26)
    lines.append(f"\nInput Rows:\n{input_rows:,}")
    lines.append(f"\nRows Removed:\n{rows_removed:,}")
    lines.append(f"\nRows Used:\n{rows_used:,}")
    lines.append(f"\nFeatures:\n{', '.join(feature_cols)}")
    lines.append(f"\nTarget:\n{target_col}")
    lines.append(f"\nModels Trained:\n{models_trained}")
    lines.append(f"\nBest Model:\n{best_name}")
    lines.append(f"\nBest F1:\n{best_f1:.4f}")
    lines.append(f"\nTraining Time:\n{training_time_sec:.2f} sec")
    lines.append("\nArtifacts Saved:\n" + ", ".join(os.path.basename(p) for p in artifact_paths))
    lines.append(f"\nPipeline Status:\n{status}")

    report = "\n".join(lines)
    print(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0-A: TIMESTAMP PARSING
# ══════════════════════════════════════════════════════════════════════════════
from pathlib import Path

def get_latest_raw_file():
    raw_dir = Path(PATHS["raw_dir"])

    csv_files = list(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in raw data folder")

    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)

    return str(latest_file)



def _parse_timestamp(series: pd.Series) -> pd.Series:
    
    try:
        # pandas ≥ 2.0: format='ISO8601' tolerates any valid ISO-8601 string
        parsed = pd.to_datetime(series, format="ISO8601", errors="coerce")
    except (ValueError, TypeError):
        # pandas 1.x fallback
        parsed = pd.to_datetime(series, format="mixed", utc=True, errors="coerce")

    nat_count = parsed.isnull().sum()
    if nat_count > 0:
        log.warning(
            f"Timestamp parsing: {nat_count} rows could not be parsed "
            f"({nat_count / len(series) * 100:.1f}% of total). "
            "These rows will be dropped at the cleaning stage."
        )
    else:
        log.info(f"Timestamp parsing: all {len(series):,} rows parsed successfully.")

    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0-B: RAW DATA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_raw_data(df: pd.DataFrame) -> dict:
    
    findings = {}
    n = len(df)
    log.info(f"[Validation] Starting raw data audit on {n:,} rows, {len(df.columns)} columns.")

    # ── 1. Missing values ──────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    findings["null_counts"] = null_counts.to_dict()
    if null_cols.empty:
        log.info("[Validation] ✅ No missing values in any column.")
    else:
        for col, cnt in null_cols.items():
            log.warning(f"[Validation] ⚠ Column '{col}': {cnt} null values ({cnt/n*100:.1f}%)")

    # ── 2. Timestamp parseability ──────────────────────────────────────────
    if "timestamp" in df.columns:
        ts_parsed     = _parse_timestamp(df["timestamp"])
        bad_ts        = ts_parsed.isnull().sum()
        findings["bad_timestamps"] = int(bad_ts)
        if bad_ts > 0:
            log.warning(f"[Validation] ⚠ {bad_ts} unparseable timestamps ({bad_ts/n*100:.1f}%)")

    # ── 3. Empty location names ────────────────────────────────────────────
    if "location_name" in df.columns:
        empty_loc = df["location_name"].isnull().sum() + (df["location_name"].astype(str).str.strip() == "").sum()
        findings["empty_location_names"] = int(empty_loc)
        if empty_loc > 0:
            log.warning(f"[Validation] ⚠ {empty_loc} empty location names.")

    # ── 4. Unknown locations ───────────────────────────────────────────────
    if "location_name" in df.columns:
        known        = set(LOCATION_NAMES)
        data_locs    = set(df["location_name"].dropna().unique())
        unknown_locs = data_locs - known
        findings["unknown_locations"] = sorted(unknown_locs)
        if unknown_locs:
            log.warning(
                f"[Validation] ⚠ {len(unknown_locs)} location(s) not in config.LOCATION_NAMES: "
                f"{unknown_locs}. These rows will be dropped."
            )
        else:
            log.info("[Validation] ✅ All locations match config.LOCATION_NAMES.")

    # ── 5. Coordinate range ────────────────────────────────────────────────
    for col, (lo, hi) in [("lat", (_CAIRO_LAT_MIN, _CAIRO_LAT_MAX)),
                           ("lon", (_CAIRO_LON_MIN, _CAIRO_LON_MAX))]:
        if col in df.columns:
            out = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            cnt = out.sum()
            findings[f"out_of_range_{col}"] = int(cnt)
            if cnt:
                log.warning(f"[Validation] ⚠ {cnt} rows with {col} outside Cairo bounds [{lo}, {hi}].")
            else:
                log.info(f"[Validation] ✅ All {col} values within Cairo bounds.")

    # ── 6. Speed sanity ────────────────────────────────────────────────────
    if "speed" in df.columns:
        bad_speed = (df["speed"].notna()) & (df["speed"] <= 0)
        cnt       = bad_speed.sum()
        findings["non_positive_speed"] = int(cnt)
        if cnt:
            log.warning(f"[Validation] ⚠ {cnt} rows with speed ≤ 0.")

    # ── 7. Vehicle count sanity ────────────────────────────────────────────
    if "vehicle_count" in df.columns:
        bad_vc = (df["vehicle_count"].notna()) & (df["vehicle_count"] < 0)
        cnt    = bad_vc.sum()
        findings["negative_vehicle_count"] = int(cnt)
        if cnt:
            log.warning(f"[Validation] ⚠ {cnt} rows with vehicle_count < 0.")

    # ── 8. Traffic status labels ───────────────────────────────────────────
    if "traffic_status" in df.columns:
        bad_status = ~df["traffic_status"].isin(_VALID_STATUSES) & df["traffic_status"].notna()
        cnt        = bad_status.sum()
        findings["invalid_traffic_status"] = int(cnt)
        if cnt:
            bad_vals = df.loc[bad_status, "traffic_status"].unique()
            log.warning(f"[Validation] ⚠ {cnt} rows with invalid traffic_status: {bad_vals}")

    # ── 9. Duplicate rows ──────────────────────────────────────────────────
    dup_count = df.duplicated().sum()
    findings["duplicate_rows"] = int(dup_count)
    if dup_count:
        log.warning(f"[Validation] ⚠ {dup_count} exact duplicate rows.")
    else:
        log.info("[Validation] ✅ No duplicate rows.")

    # ── 10. Infinite values ────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {col: int(np.isinf(df[col]).sum()) for col in num_cols}
    total_inf  = sum(inf_counts.values())
    findings["infinite_values"] = inf_counts
    if total_inf:
        for col, cnt in inf_counts.items():
            if cnt:
                log.warning(f"[Validation] ⚠ {cnt} infinite values in column '{col}'.")
    else:
        log.info("[Validation] ✅ No infinite values in numeric columns.")

    log.info("[Validation] Raw data audit complete.")
    print_validation_summary(df, findings)
    return findings


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1-A: STRUCTURED CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    
    n_start = len(df)
    log.info(f"[Cleaning] Starting with {n_start:,} rows.")
    removed = {
        "duplicate rows": 0,
        "invalid timestamps": 0,
        "unknown locations": 0,
        "invalid coordinate records": 0,
        "infinite value rows": 0,
        "invalid speed rows": 0,
        "invalid vehicle_count rows": 0,
        "invalid traffic_status rows": 0,
    }

    # ── Column aliases (Azure SQL schema compatibility) ────────────────────
    rename_map = {}
    if "region" in df.columns and "location_name" not in df.columns:
        rename_map["region"] = "location_name"
    if "avg_speed" in df.columns and "speed" not in df.columns:
        rename_map["avg_speed"] = "speed"
    if rename_map:
        df = df.rename(columns=rename_map)
        log.info(f"[Cleaning] Column aliases applied: {rename_map}")

    # ── Replace infinities before any numeric check ────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_mask = df[num_cols].isin([np.inf, -np.inf]).any(axis=1)
    if inf_mask.sum():
        df = df[~inf_mask]
        removed["infinite value rows"] += int(inf_mask.sum())
        log.warning(f"[Cleaning] Dropped {inf_mask.sum()} rows containing infinite values. "
                    f"Remaining: {len(df):,}")

    # ── Exact duplicates ───────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        removed["duplicate rows"] += dropped
        log.info(f"[Cleaning] Dropped {dropped} exact duplicate rows. Remaining: {len(df):,}")

    # ── Timestamp: parse with ISO-8601-tolerant parser, then drop NaT ─────
    if "timestamp" in df.columns:
        before = len(df)
        df["timestamp"] = _parse_timestamp(df["timestamp"])
        df = df.dropna(subset=["timestamp"])
        dropped = before - len(df)
        if dropped:
            removed["invalid timestamps"] += dropped
            log.warning(f"[Cleaning] Dropped {dropped} rows with unparseable timestamps. "
                        f"Remaining: {len(df):,}")

    # ── Location name: drop empty or unknown ──────────────────────────────
    if "location_name" in df.columns:
        before = len(df)
        df = df[df["location_name"].notna()]
        df = df[df["location_name"].astype(str).str.strip() != ""]
        unknown_mask = ~df["location_name"].isin(set(LOCATION_NAMES))
        if unknown_mask.sum():
            bad_locs = df.loc[unknown_mask, "location_name"].unique()
            log.warning(
                f"[Cleaning] Dropping {unknown_mask.sum()} rows with unknown locations: {bad_locs}. "
                "Add them to LOCATIONS in config/config.py to include them in training."
            )
            df = df[~unknown_mask]
        dropped = before - len(df)
        if dropped:
            removed["unknown locations"] += dropped
            log.info(f"[Cleaning] Dropped {dropped} rows with invalid location names. "
                     f"Remaining: {len(df):,}")

    # ── Coordinates from 'coordinates' string column (when lat/lon absent) ─
    if "coordinates" in df.columns and "lat" not in df.columns:
        coords    = df["coordinates"].str.extract(r"([\d.]+),\s*([\d.]+)")
        df["lat"] = coords[0].astype(float)
        df["lon"] = coords[1].astype(float)
        log.info("[Cleaning] lat/lon derived from 'coordinates' string column.")

    # ── Coordinate range ───────────────────────────────────────────────────
    for col, lo, hi in [("lat", _CAIRO_LAT_MIN, _CAIRO_LAT_MAX),
                         ("lon", _CAIRO_LON_MIN, _CAIRO_LON_MAX)]:
        if col in df.columns:
            before   = len(df)
            bad_mask = df[col].isnull() | (df[col] < lo) | (df[col] > hi)
            df       = df[~bad_mask]
            dropped  = before - len(df)
            if dropped:
                removed["invalid coordinate records"] += dropped
                log.warning(f"[Cleaning] Dropped {dropped} rows with invalid {col}. "
                            f"Remaining: {len(df):,}")

    # ── Speed ──────────────────────────────────────────────────────────────
    if "speed" in df.columns:
        before = len(df)
        df     = df.dropna(subset=["speed"])
        df     = df[df["speed"] > 0]
        dropped = before - len(df)
        if dropped:
            removed["invalid speed rows"] += dropped
            log.warning(f"[Cleaning] Dropped {dropped} rows with missing or non-positive speed. "
                        f"Remaining: {len(df):,}")

    # ── Vehicle count ──────────────────────────────────────────────────────
    if "vehicle_count" in df.columns:
        before = len(df)
        df     = df.dropna(subset=["vehicle_count"])
        df     = df[df["vehicle_count"] >= 0]
        dropped = before - len(df)
        if dropped:
            removed["invalid vehicle_count rows"] += dropped
            log.warning(f"[Cleaning] Dropped {dropped} rows with invalid vehicle_count. "
                        f"Remaining: {len(df):,}")

    # ── Traffic status (if pre-labeled) ───────────────────────────────────
    if "traffic_status" in df.columns:
        before    = len(df)
        bad_label = ~df["traffic_status"].isin(_VALID_STATUSES) & df["traffic_status"].notna()
        if bad_label.sum():
            log.warning(f"[Cleaning] Dropping {bad_label.sum()} rows with invalid traffic_status labels.")
            df = df[~bad_label]
        dropped = before - len(df)
        if dropped:
            removed["invalid traffic_status rows"] += dropped
            log.info(f"[Cleaning] After label cleaning: {len(df):,} rows.")

    n_end   = len(df)
    n_dropped = n_start - n_end
    log.info(
        f"[Cleaning] Complete. {n_end:,} clean rows remain "
        f"(dropped {n_dropped:,} = {n_dropped/max(n_start,1)*100:.1f}%)."
    )
    print_cleaning_summary(n_start, n_end, removed)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1-B: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    
    log.info("[Engineering] Deriving temporal features from timestamp…")

    # ── Temporal features ──────────────────────────────────────────────────
    # timestamp is already parsed by _clean_raw; use .dt accessor directly
    df["hour"]         = df["timestamp"].dt.hour
    df["minute"]       = df["timestamp"].dt.minute
    df["day_of_week"]  = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0
    )

    # ── Target label (derive from speed when absent) ───────────────────────
    if "traffic_status" not in df.columns:
        if "speed" not in df.columns:
            raise ValueError(
                "Cannot derive traffic_status: neither 'traffic_status' nor 'speed' "
                "column is present in the cleaned dataset."
            )
        def _speed_to_status(s: float) -> str:
            if s < 20:   return "Congested"
            elif s < 50: return "Moderate"
            return "Free_Flow"
        df["traffic_status"] = df["speed"].apply(_speed_to_status)
        log.info("[Engineering] traffic_status derived from speed column.")
    else:
        log.info("[Engineering] traffic_status column already present — using as-is.")

    # ── Location encoder ───────────────────────────────────────────────────
    # Fitted on the canonical LOCATION_NAMES list from config, not on data.
    # This guarantees consistent encoding across training runs even when
    # not all locations appear in every batch.
    loc_encoder = LabelEncoder()
    loc_encoder.fit(LOCATION_NAMES)
    df["loc_encoded"] = loc_encoder.transform(df["location_name"])

    # ── Target encoder ─────────────────────────────────────────────────────
    target_encoder = LabelEncoder()
    df["target_enc"] = target_encoder.fit_transform(df["traffic_status"])

    log.info(
        f"[Engineering] Complete. Shape: {df.shape}. "
        f"Target distribution:\n{df['traffic_status'].value_counts().to_string()}"
    )

    generated_features = ["hour", "minute", "day_of_week", "is_weekend", "is_peak_hour"]
    print_feature_snapshot(df, generated_features)
    print_location_audit(df)
    print_target_distribution(df, target_col="traffic_status")

    return df, loc_encoder, target_encoder


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1-C: FEATURE MATRIX VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_feature_matrix(df: pd.DataFrame) -> None:
    
    log.info("[FeatureValidation] Validating feature matrix…")
    errors = []

    # ── Required columns present? ──────────────────────────────────────────
    missing_cols = [c for c in FEATURE_COLS + ["target_enc"] if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        raise RuntimeError(
            "[FeatureValidation] FATAL — cannot proceed:\n" + "\n".join(errors)
        )

    # ── NaN check ─────────────────────────────────────────────────────────
    feat_df     = df[FEATURE_COLS]
    nan_counts  = feat_df.isnull().sum()
    nan_cols    = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        for col, cnt in nan_cols.items():
            log.error(f"[FeatureValidation] ❌ NaN in feature '{col}': {cnt} rows")
        errors.append(f"NaN values remain in features after engineering: {nan_cols.to_dict()}")

    # ── Infinite check ─────────────────────────────────────────────────────
    inf_counts = feat_df.apply(lambda c: np.isinf(c).sum() if c.dtype.kind == "f" else 0)
    inf_cols   = inf_counts[inf_counts > 0]
    if not inf_cols.empty:
        for col, cnt in inf_cols.items():
            log.error(f"[FeatureValidation] ❌ Infinite values in feature '{col}': {cnt} rows")
        errors.append(f"Infinite values remain in features: {inf_cols.to_dict()}")

    # ── Target NaN ────────────────────────────────────────────────────────
    target_nan = df["target_enc"].isnull().sum()
    if target_nan:
        errors.append(f"NaN in target_enc: {target_nan} rows")

    # ── Raise on any errors ────────────────────────────────────────────────
    if errors:
        raise RuntimeError(
            "[FeatureValidation] FATAL — data has irrecoverable issues:\n"
            + "\n".join(errors)
        )

    # ── Minimum rows ───────────────────────────────────────────────────────
    if len(df) < _MIN_ROWS_FOR_TRAINING:
        raise RuntimeError(
            f"[FeatureValidation] Only {len(df)} rows remain after cleaning — "
            f"minimum required is {_MIN_ROWS_FOR_TRAINING}. "
            "Collect more data before training."
        )

    # ── Class distribution ─────────────────────────────────────────────────
    class_counts = df["traffic_status"].value_counts()
    n_classes    = len(class_counts)
    if n_classes < 2:
        raise RuntimeError(
            f"[FeatureValidation] Only {n_classes} class(es) present — "
            "need at least 2 to train a classifier."
        )

    log.info(f"[FeatureValidation] ✅ {len(df):,} rows, {n_classes} classes.")
    log.info(f"[FeatureValidation] Class distribution:\n{class_counts.to_string()}")
    for col in FEATURE_COLS:
        s = df[col]
        log.info(
            f"[FeatureValidation]   {col:15s}  "
            f"min={s.min():.4g}  max={s.max():.4g}  "
            f"mean={s.mean():.4g}  null={s.isnull().sum()}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT: LOAD + CLEAN + ENGINEER
# ══════════════════════════════════════════════════════════════════════════════

def load_and_engineer(
    csv_path: str,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder, int]:
    
    log.info(f"[Load] Reading: {csv_path}")
    # Do NOT pass parse_dates here — let _parse_timestamp handle the mixed format
    df = pd.read_csv(csv_path)
    n_raw_rows = len(df)
    log.info(f"[Load] {n_raw_rows:,} rows loaded, columns: {df.columns.tolist()}")

    # ── Stage 1: Raw data audit (non-destructive — logging only) ──────────
    validate_raw_data(df)

    # ── Stage 2: Structured cleaning ──────────────────────────────────────
    df = _clean_raw(df)

    if len(df) == 0:
        raise RuntimeError(
            "[Load] Zero rows remain after cleaning. "
            "Check the raw data source or cleaning thresholds."
        )

    # ── Stage 3: Feature engineering ──────────────────────────────────────
    df, loc_encoder, target_encoder = _engineer_features(df)

    # ── Stage 4: Feature matrix validation (destructive check — will raise) ─
    validate_feature_matrix(df)

    return df, loc_encoder, target_encoder, n_raw_rows


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PREPROCESSING  (split → scale → SMOTE)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, use_smote: bool = True, target_encoder: Optional[LabelEncoder] = None):
    
    X = df[FEATURE_COLS].values
    y = df["target_enc"].values

    # ── Pre-flight assertions (belt-and-suspenders) ────────────────────────
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0:
        raise ValueError(
            f"[Preprocess] X contains {nan_count} NaN value(s). "
            "This should have been caught by validate_feature_matrix()."
        )
    if inf_count > 0:
        raise ValueError(
            f"[Preprocess] X contains {inf_count} infinite value(s). "
            "This should have been caught by validate_feature_matrix()."
        )
    if len(np.unique(y)) < 2:
        raise ValueError(
            "[Preprocess] y contains fewer than 2 unique classes. Cannot train."
        )

    log.info(f"[Preprocess] X shape: {X.shape}  y classes: {np.unique(y)}")

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TRAINING["test_size"],
        random_state = TRAINING["random_state"],
        stratify     = y,
    )
    log.info(f"[Preprocess] Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    if target_encoder is not None:
        print_split_summary(X_train, X_test, y_train, y_test, target_encoder, y_full=y)

    # ── Scaling ────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── SMOTE ─────────────────────────────────────────────────────────────
    if use_smote:
        class_counts = np.bincount(y_train)
        min_class    = class_counts.min()
        n_classes    = len(class_counts)

        def _named(counts_array):
            if target_encoder is not None:
                return {target_encoder.inverse_transform([i])[0]: int(c) for i, c in enumerate(counts_array)}
            return {f"class_{i}": int(c) for i, c in enumerate(counts_array)}

        before_named = _named(class_counts)

        # SMOTE requires at least k_neighbors + 1 samples per class (default k=5 → need ≥6)
        smote_k = min(5, min_class - 1)
        if smote_k < 1:
            log.warning(
                f"[Preprocess] SMOTE skipped: smallest class has only {min_class} sample(s) "
                f"in training set — need at least 2. Using raw class distribution."
            )
            nan_report = {f"feature_{i}": int(np.isnan(X_train[:, i]).sum()) for i in range(X_train.shape[1])}
            print_smote_report(
                before_named,
                failure_reason=f"smallest class has only {min_class} sample(s); need at least 2.",
                nan_report=nan_report,
            )
        else:
            try:
                sm          = SMOTE(random_state=TRAINING["random_state"], k_neighbors=smote_k)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                after_named = _named(np.bincount(y_train))
                log.info(
                    f"[Preprocess] SMOTE complete (k={smote_k}). "
                    f"Train size after resampling: {X_train.shape[0]:,}"
                )
                print_smote_report(before_named, after_counts=after_named)
            except Exception as exc:
                log.warning(
                    f"[Preprocess] SMOTE failed: {exc}. "
                    "Continuing with raw class distribution."
                )
                nan_report = {f"feature_{i}": int(np.isnan(X_train[:, i]).sum()) for i in range(X_train.shape[1])}
                print_smote_report(before_named, failure_reason=str(exc), nan_report=nan_report)

    return X_train, X_test, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=-1),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=42
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
    return models


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    models: dict,
    X_train, X_test, y_train, y_test,
    target_encoder: LabelEncoder,
    cv_folds: int = 5,
) -> dict:
    
    class_names = target_encoder.classes_
    results     = {}

    # ── CV guard: cannot use more folds than smallest class has samples ────
    min_class_train = int(np.bincount(y_train).min())
    safe_cv_folds   = min(cv_folds, min_class_train)
    if safe_cv_folds < cv_folds:
        log.warning(
            f"[Training] CV folds reduced from {cv_folds} to {safe_cv_folds} "
            f"(smallest training class has only {min_class_train} sample(s))."
        )

    for name, model in models.items():
        log.info(f"[Training] Fitting {name}…")
        print(f"\n---\n\n## Training {name}\n")
        try:
            fit_start = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - fit_start
            print(f"Training Time: {fit_time:.2f} sec")

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            try:
                auc = (
                    roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                    if y_prob is not None else 0.0
                )
            except Exception as auc_exc:
                log.warning(f"[Training] ROC-AUC skipped for {name}: {auc_exc}")
                auc = 0.0

            # ── Cross-validation ───────────────────────────────────────────
            if safe_cv_folds >= 2:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv      = StratifiedKFold(n_splits=safe_cv_folds, shuffle=True, random_state=42),
                    scoring = "accuracy",
                    n_jobs  = -1,
                )
            else:
                cv_scores = np.array([acc])
                log.warning(f"[Training] Cross-validation skipped for {name} (insufficient samples).")

            print("\nCross Validation Scores:")
            for fold_i, score in enumerate(cv_scores, start=1):
                print(f"Fold{fold_i}: {score:.4f}")
            print(f"\nCV Mean: {cv_scores.mean():.4f}")
            print(f"CV Std: {cv_scores.std():.4f}")

            print("\nTest Metrics:")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1:        {f1:.4f}")
            print(f"ROC-AUC:   {auc:.4f}")

            # ── Confusion matrix plot ──────────────────────────────────────
            
            present_indices = sorted(np.unique(np.concatenate([y_test, y_pred])))
            present_labels  = [class_names[i] for i in present_indices if i < len(class_names)]
            cm      = confusion_matrix(y_test, y_pred, labels=present_indices)
            cm_path = os.path.join(PLOTS_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
            try:
                fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1c1f26")
                ax.set_facecolor("#1c1f26")
                ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=present_labels
                ).plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(name, color="white", fontsize=11, pad=10)
                for text in ax.texts:
                    text.set_color("white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.tick_params(colors="white")
                plt.tight_layout()
                plt.savefig(cm_path, dpi=100, bbox_inches="tight", facecolor="#1c1f26")
                plt.close()
            except Exception as plot_exc:
                log.warning(f"[Training] Confusion matrix plot failed for {name}: {plot_exc}")
                plt.close()

            results[name] = {
                "model":     model,
                "accuracy":  acc,
                "precision": prec,
                "recall":    rec,
                "f1":        f1,
                "roc_auc":   auc,
                "cv_mean":   cv_scores.mean(),
                "cv_std":    cv_scores.std(),
                "report":    classification_report(
                    y_test, y_pred, target_names=class_names, zero_division=0
                ),
                "cm_path":   cm_path,
                "y_pred":    y_pred,
                "y_prob":    y_prob,
            }

            log.info(
                f"[Training] {name}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
                f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            )

        except Exception as exc:
            log.error(
                f"[Training] {name} FAILED and will be excluded from results. "
                f"Error: {type(exc).__name__}: {exc}"
            )
            # Do NOT re-raise — continue with remaining models

    if not results:
        raise RuntimeError(
            "[Training] All models failed to train. "
            "Check preprocessing and feature matrix validation logs."
        )

    log.info(f"[Training] {len(results)}/{len(models)} models trained successfully.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE & PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = [
        {
            "Model":         name,
            "Accuracy":      round(r["accuracy"],  4),
            "Precision":     round(r["precision"], 4),
            "Recall":        round(r["recall"],    4),
            "F1 (weighted)": round(r["f1"],        4),
            "ROC-AUC":       round(r["roc_auc"],   4),
            "CV Accuracy":   f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
        }
        for name, r in results.items()
    ]
    df = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def plot_comparison(results: dict, save_path: str):
    names   = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    colors  = ["#00d4ff", "#7b2fff", "#ff6b6b", "#6bcb77", "#ffd93d"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0e1117")
    fig.suptitle("Model Comparison — All Metrics", color="white", fontsize=14, y=1.02)

    ax = axes[0]
    ax.set_facecolor("#1c1f26")
    x, width = np.arange(len(names)), 0.15
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, [results[n][m] for n in names], width, label=m.upper(), color=c, alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=20, ha="right", color="white", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color="white")
    ax.set_title("All Metrics by Model", color="white")
    ax.legend(fontsize=8, facecolor="#2a2d35", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2a2d35")

    ax2 = axes[1]
    ax2.set_facecolor("#1c1f26")
    sns.heatmap(
        [[results[n][m] for m in metrics] for n in names],
        annot=True, fmt=".3f",
        xticklabels=[m.upper() for m in metrics],
        yticklabels=names,
        cmap="YlOrRd", ax=ax2,
        linewidths=0.5, linecolor="#2a2d35",
        cbar_kws={"shrink": 0.8},
    )
    ax2.set_title("Metric Heatmap", color="white")
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    for text in ax2.texts:
        text.set_color("black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    log.info(f"[Plots] Comparison chart saved → {save_path}")


def plot_feature_importance(model, save_path: str):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    feat_names  = [FEATURE_COLS[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1c1f26")
    ax.set_facecolor("#1c1f26")
    bars = ax.barh(feat_names[::-1], importances[indices][::-1], color="#00d4ff", alpha=0.85, edgecolor="#2a2d35")
    ax.set_xlabel("Importance", color="white")
    ax.set_title("Feature Importances (Best Model)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2a2d35")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", color="#ffd93d", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#1c1f26")
    plt.close()
    log.info(f"[Plots] Feature importance chart saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(
    best_name:      str,
    best_model,
    scaler:         StandardScaler,
    loc_encoder:    LabelEncoder,
    target_encoder: LabelEncoder,
    table:          pd.DataFrame,
    report_txt:     str,
    n_records:      int = 0,
    data_source:    str = "unknown",
):
    joblib.dump(best_model,     os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler,         os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(loc_encoder,    os.path.join(MODELS_DIR, "loc_encoder.pkl"))
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, "target_encoder.pkl"))

    meta = {
        "best_model_name": best_name,
        "features":        FEATURE_COLS,
        "trained_at":      datetime.now().isoformat(),
        "locations":       list(loc_encoder.classes_),
        "classes":         list(target_encoder.classes_),
        "n_records":       n_records,
        "data_source":     data_source,
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    table.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    with open(os.path.join(REPORTS_DIR, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    log.info(f"[Artifacts] ✅ Best model ({best_name}) saved → {MODELS_DIR}/best_model.pkl")

    artifact_paths = [
        os.path.join(MODELS_DIR, "best_model.pkl"),
        os.path.join(MODELS_DIR, "scaler.pkl"),
        os.path.join(MODELS_DIR, "loc_encoder.pkl"),
        os.path.join(MODELS_DIR, "target_encoder.pkl"),
        os.path.join(MODELS_DIR, "meta.json"),
    ]
    print_artifact_sizes(artifact_paths)
    return artifact_paths


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    csv_path:  Optional[str] = None,
    rows:      Optional[int] = None,
    simulate:  bool          = False,
    use_smote: bool          = True,
    cv_folds:  int           = 5,
):
    print("\n" + "=" * 65)
    print("  Egypt Traffic Intelligence — Unified Training Pipeline v4.0")
    print("=" * 65)
    print(FEATURE_AUDIT)

    # ── STEP 0: Data Ingestion ──────────────────────────────────────────────
    if csv_path:
        log.info(f"Using specified CSV: {csv_path}")
        if not os.path.exists(csv_path):
            sys.exit(f"❌ CSV not found: {csv_path}")
    else:
        from ingestion.data_ingestion import collect_traffic_data, get_latest_raw_csv

        if rows:
            log.info("[Pipeline] Step 0 → Fetching fresh traffic data…")
            collect_traffic_data(rows_to_fetch=rows, simulate=simulate)
        else:
            log.info("[Pipeline] Step 0 → No --rows specified; using latest available raw file.")

        csv_path = get_latest_raw_csv()
        if csv_path is None:
            sys.exit(
                "❌ No raw CSV found in data/raw/. "
                "Run with --rows N to fetch data first."
            )
        log.info(f"[Pipeline] Step 0 → Using raw file: {csv_path}")

    # ── STEP 1: Load, Validate, Clean, Engineer ────────────────────────────
    log.info("[Pipeline] Step 1 → Load / validate / clean / engineer…")
    try:
        df, loc_enc, tgt_enc, n_raw_rows = load_and_engineer(csv_path)
    except RuntimeError as exc:
        sys.exit(f"❌ Data pipeline failed:\n{exc}")

    clean_path = PATHS["clean_csv"]
    df.to_csv(clean_path, index=False, encoding="utf-8")
    log.info(f"[Pipeline] Step 1 → Processed data saved → {clean_path}  ({len(df):,} rows)")

    # ── STEP 2: Preprocessing ─────────────────────────────────────────────
    log.info("[Pipeline] Step 2 → Preprocessing (split / scale / SMOTE)…")
    try:
        X_train, X_test, y_train, y_test, scaler = preprocess(df, use_smote=use_smote, target_encoder=tgt_enc)
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"❌ Preprocessing failed:\n{exc}")
    log.info(f"[Pipeline] Step 2 → Train: {X_train.shape}  Test: {X_test.shape}")

    # ── STEP 2.5: Feature quality audit ────────────────────────────────────
    print_feature_audit(df, FEATURE_COLS)

    # ── STEP 3: Train all models ──────────────────────────────────────────
    log.info("[Pipeline] Step 3 → Training classifiers…")
    training_start = datetime.now()
    results = train_and_evaluate(
        get_models(), X_train, X_test, y_train, y_test, tgt_enc, cv_folds
    )
    training_time_sec = (datetime.now() - training_start).total_seconds()

    # ── STEP 4: Compare ───────────────────────────────────────────────────
    log.info("[Pipeline] Step 4 → Building comparison table…")
    table = build_comparison_table(results)
    print("\n📊 Model Comparison Table")
    print(table.to_string())

    # ── STEP 5: Select best model ─────────────────────────────────────────
    metric    = TRAINING.get("selection_metric", "f1")
    print_ranking_table(results, metric=metric)
    best_name = max(results, key=lambda n: results[n][metric])
    best      = results[best_name]
    print(f"\n🏆 Best Model: {best_name}  (by {metric})")
    print(f"   Accuracy  : {best['accuracy']:.4f}")
    print(f"   F1        : {best['f1']:.4f}")
    print(f"   ROC-AUC   : {best['roc_auc']:.4f}")
    print(f"\nClassification Report:\n{best['report']}")

    # ── STEP 6: Plots ─────────────────────────────────────────────────────
    log.info("[Pipeline] Step 6 → Generating plots…")
    plot_comparison(results, os.path.join(PLOTS_DIR, "model_comparison.png"))
    plot_feature_importance(best["model"], os.path.join(PLOTS_DIR, "feature_importances.png"))

    # ── STEP 7: Build text report ─────────────────────────────────────────
    report_lines = [
        "Egypt Traffic Intelligence — Training Report  v4.0",
        f"Generated  : {datetime.now().isoformat()}",
        f"Data source: {'Simulated' if simulate else 'TomTom API / Azure SQL'}",
        f"Raw file   : {csv_path}",
        f"Records after cleaning: {len(df):,}",
        "=" * 60,
        FEATURE_AUDIT,
        "\nModel Comparison Table",
        "-" * 60,
        table.to_string(),
        f"\n\n🏆 Best Model: {best_name}",
        f"Selection criterion: highest weighted {metric}",
        f"\nFull classification report for {best_name}:",
        best["report"],
    ]
    for name, r in results.items():
        report_lines += [f"\n--- {name} ---", r["report"]]

    # ── STEP 8: Save all artifacts ────────────────────────────────────────
    log.info("[Pipeline] Step 8 → Saving artifacts…")
    artifact_paths = save_artifacts(
        best_name      = best_name,
        best_model     = best["model"],
        scaler         = scaler,
        loc_encoder    = loc_enc,
        target_encoder = tgt_enc,
        table          = table,
        report_txt     = "\n".join(report_lines),
        n_records      = len(df),
        data_source    = "simulation" if simulate else "TomTom API",
    )

    print_final_report(
        input_rows        = n_raw_rows,
        rows_removed      = n_raw_rows - len(df),
        rows_used         = len(df),
        feature_cols      = FEATURE_COLS,
        target_col        = TARGET_COL,
        models_trained    = len(results),
        best_name         = best_name,
        best_f1           = best["f1"],
        training_time_sec = training_time_sec,
        artifact_paths    = artifact_paths,
        status            = "SUCCESS",
    )

    print("\n✅ Pipeline complete. Artifacts saved.")
    print(f"   Models  → {MODELS_DIR}")
    print(f"   Reports → {REPORTS_DIR}")
    print(f"   Plots   → {PLOTS_DIR}")
    return results, best_name, table


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Egypt Traffic Intelligence — Automated Training Pipeline v4.0"
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--rows", type=int, default=None,
        help="Fetch this many fresh records from TomTom API (e.g. --rows 5000)"
    )
    src.add_argument(
        "--data", type=str, default=None,
        help="Skip ingestion and use an existing CSV file"
    )

    parser.add_argument(
        "--simulate", action="store_true",
        help="Generate synthetic data instead of calling the API"
    )
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument(
        "--cv", type=int, default=TRAINING["cv_folds"],
        help=f"Cross-validation folds (default: {TRAINING['cv_folds']})"
    )

    args = parser.parse_args()
    run_pipeline(
        csv_path  = args.data,
        rows      = args.rows,
        simulate  = args.simulate,
        use_smote = not args.no_smote,
        cv_folds  = args.cv,
    )
