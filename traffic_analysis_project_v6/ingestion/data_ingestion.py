

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import glob
import time
import logging
import argparse
import random
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import COLLECTION, LOCATIONS, PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingestion")

# ── Timestamped filename pattern ───────────────────────────────────────────────
_RAW_FILENAME_PATTERN = "traffic_{ts}.csv"          
_RAW_GLOB_PATTERN     = "traffic_*.csv"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC HELPERS — dataset discovery
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_raw_csv(raw_dir: str = None) -> Optional[str]:
    
    raw_dir = raw_dir or PATHS["raw_dir"]
    pattern = os.path.join(raw_dir, _RAW_GLOB_PATTERN)
    files   = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_latest_raw(raw_dir: str = None) -> pd.DataFrame:
    
    path = get_latest_raw_csv(raw_dir)
    if path is None:
        raise FileNotFoundError(
            f"No raw CSV files found in {raw_dir or PATHS['raw_dir']}. "
            "Run data ingestion first."
        )
    log.info(f"Loading latest raw dataset: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"])


def _build_raw_output_path(raw_dir: str = None) -> str:
    
    raw_dir = raw_dir or PATHS["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y_%m_%d_%H_%M")
    filename = _RAW_FILENAME_PATTERN.format(ts=ts)
    return os.path.join(raw_dir, filename)


# ══════════════════════════════════════════════════════════════════════════════
# TOMTOM API — Single Location Poll
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_location(loc: dict, cfg: dict, session: requests.Session) -> Optional[dict]:
    
    params = {
        "point": f"{loc['lat']},{loc['lon']}",
        "unit":  cfg["unit"],
        "key":   cfg["api_key"],
    }
    url     = cfg["base_url"]
    retries = cfg["max_retries"]
    backoff = cfg["retry_backoff_sec"]

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=cfg["request_timeout_sec"])

            if resp.status_code == 200:
                data = resp.json()
                fsd  = data.get("flowSegmentData", {})

                if fsd.get("roadClosure", False):
                    log.debug(f"Road closed: {loc['name']} — skipping.")
                    return None
                if "currentSpeed" not in fsd:
                    log.debug(f"No speed data for {loc['name']} — skipping.")
                    return None

                now = datetime.now(timezone.utc)
                return {
                    "timestamp":       now.isoformat(),
                    "location_name":   loc["name"],
                    "coordinates":     f"{loc['lat']}, {loc['lon']}",
                    "lat":             loc["lat"],
                    "lon":             loc["lon"],
                    "speed":           fsd["currentSpeed"],
                    "free_flow_speed": fsd.get("freeFlowSpeed"),
                    "vehicle_count":   _estimate_vehicle_count(
                        fsd["currentSpeed"], fsd.get("freeFlowSpeed", 80)
                    ),
                    "confidence":      fsd.get("confidence"),
                    "road_closure":    fsd.get("roadClosure", False),
                }

            elif resp.status_code == 403:
                log.error(
                    f"403 Forbidden for {loc['name']}. "
                    "Check your API key on the TomTom Developer Portal."
                )
                return None  # Permanent — do not retry auth errors

            elif resp.status_code == 429:
                wait = backoff * (2 ** attempt)
                log.warning(f"Rate limited (429). Waiting {wait:.1f}s before retry {attempt}/{retries}…")
                time.sleep(wait)

            else:
                log.warning(
                    f"HTTP {resp.status_code} for {loc['name']} "
                    f"(attempt {attempt}/{retries}). Retrying in {backoff}s…"
                )
                time.sleep(backoff * attempt)

        except requests.exceptions.Timeout:
            log.warning(f"Timeout on {loc['name']} (attempt {attempt}/{retries}).")
            time.sleep(backoff * attempt)

        except requests.exceptions.ConnectionError as e:
            log.warning(f"Connection error on {loc['name']}: {e} (attempt {attempt}/{retries}).")
            time.sleep(backoff * attempt)

        except Exception as e:
            log.warning(f"Unexpected error on {loc['name']}: {e} (attempt {attempt}/{retries}).")
            time.sleep(backoff * attempt)

    log.error(f"Permanently failed to fetch {loc['name']} after {retries} retries.")
    return None


def _estimate_vehicle_count(current_speed: float, free_flow_speed: float) -> int:
    
    if free_flow_speed <= 0:
        return 50
    ratio = max(0.0, min(1.0, current_speed / free_flow_speed))
    count = int(200 - ratio * 190)
    return max(5, count + random.randint(-10, 10))


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION MODE
# ══════════════════════════════════════════════════════════════════════════════

# Base free-flow speeds per location — driven by LOCATIONS in config
_BASE_FREE_FLOW = {
    "Ring Road - Maadi":      90,
    "October Bridge":         70,
    "26th July Corridor":     80,
    "Abbas El Akkad":         60,
    "Galaa Square":           50,
    "Salah Salem":            75,
    "Autostrad - Sheraton":   85,
    "Tahrir Square":          45,
    "Corniche El Nil":        65,
    # New locations
    "Mosheer Tantawy Axis":   80,
    "Rod El Farag Axis":      75,
    "Ring Road - Marg":       90,
    "Ring Road - Moneeb":     85,
    "Gamaat El Dowal St":     55,
    "Faisal Street":          50,
    "North 90th Street":      70,
    "South 90th Street":      65,
}
_DEFAULT_FREE_FLOW = 70  # Fallback for any future location not listed above


def _simulate_record(loc: dict, timestamp: datetime) -> dict:
    """Generate a single synthetic traffic record for a location."""
    hour    = timestamp.hour
    weekday = timestamp.weekday()

    base_free_flow = _BASE_FREE_FLOW.get(loc["name"], _DEFAULT_FREE_FLOW)

    is_morning_rush = 7  <= hour <= 9
    is_evening_rush = 16 <= hour <= 19
    is_peak         = is_morning_rush or is_evening_rush
    is_weekend      = weekday >= 5
    is_night        = hour <= 5 or hour >= 23

    if is_peak and not is_weekend:
        factor = random.uniform(0.25, 0.55)
    elif is_night:
        factor = random.uniform(0.85, 1.00)
    elif is_weekend:
        factor = random.uniform(0.60, 0.90)
    else:
        factor = random.uniform(0.45, 0.80)

    current_speed = max(5, int(base_free_flow * factor + random.gauss(0, 4)))
    ratio         = current_speed / base_free_flow
    vehicle_count = max(5, int(200 - ratio * 190 + random.randint(-15, 15)))

    return {
        "timestamp":       timestamp.isoformat(),
        "location_name":   loc["name"],
        "coordinates":     f"{loc['lat']}, {loc['lon']}",
        "lat":             loc["lat"],
        "lon":             loc["lon"],
        "speed":           current_speed,
        "free_flow_speed": base_free_flow,
        "vehicle_count":   vehicle_count,
        "confidence":      round(random.uniform(0.80, 0.99), 2),
        "road_closure":    False,
    }


def _generate_simulated_dataset(rows: int, locations: list) -> pd.DataFrame:
    
    log.info(f"[SIMULATION MODE] Generating {rows:,} synthetic records…")
    records = []
    base_time = datetime.now(timezone.utc)

    # Hour weights that favour rush hours for realism
    hour_weights = [
        1, 1, 1, 1, 1, 2,       # 00-05 night
        3, 8, 10, 6, 4, 4,      # 06-11 morning rush
        4, 4, 5, 5, 9, 11,      # 12-17 evening build
        10, 8, 5, 3, 2, 1,      # 18-23 evening tail
    ]

    for i in range(rows):
        hour   = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        ts     = base_time.replace(
            hour=hour, minute=minute, second=random.randint(0, 59), microsecond=0
        )
        loc = locations[i % len(locations)]
        records.append(_simulate_record(loc, ts))

        if (i + 1) % COLLECTION["log_every_n"] == 0 or (i + 1) == rows:
            log.info(f"  Generating records… {i + 1:,} / {rows:,}")

    df = pd.DataFrame(records)
    log.info(f"[SIMULATION] Dataset generated. Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# RAW FILE PERSISTENCE  (always timestamped, never overwritten)
# ══════════════════════════════════════════════════════════════════════════════

def _save_raw(df: pd.DataFrame, path: str) -> None:
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"  Raw data saved → {path}")
    log.info(f"  Shape: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def collect_traffic_data(
    rows_to_fetch: int  = None,
    simulate:      bool = None,
    output_path:   str  = None,
) -> pd.DataFrame:
    
    cfg      = COLLECTION.copy()
    rows     = rows_to_fetch if rows_to_fetch is not None else cfg["rows_to_fetch"]
    sim_mode = simulate      if simulate      is not None else cfg["simulation_mode"]
    out_path = output_path   if output_path   is not None else _build_raw_output_path()

    log.info("=" * 65)
    log.info("  Egypt Traffic Intelligence — Data Ingestion")
    log.info("=" * 65)
    log.info(f"  Mode           : {'SIMULATION' if sim_mode else 'LIVE API (TomTom)'}")
    log.info(f"  Rows requested : {rows:,}")
    log.info(f"  Locations      : {len(LOCATIONS)}")
    log.info(f"  Output path    : {out_path}")
    log.info("=" * 65)

    # ── Simulation path ────────────────────────────────────────────────────
    if sim_mode:
        df = _generate_simulated_dataset(rows, LOCATIONS)
        _save_raw(df, out_path)
        return df

    # ── Live API path ──────────────────────────────────────────────────────
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    records              = []
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = len(LOCATIONS) * 5

    log.info("Collecting records…")

    while len(records) < rows:
        round_records = []

        for loc in LOCATIONS:
            if len(records) + len(round_records) >= rows:
                break

            record = _fetch_location(loc, cfg, session)
            if record:
                round_records.append(record)
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        if not round_records:
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.error(
                    f"  {consecutive_failures} consecutive API failures.\n"
                    "  → Falling back to SIMULATION MODE for remaining records.\n"
                    "  → Check your API key at: https://developer.tomtom.com"
                )
                remaining   = rows - len(records)
                fallback_df = _generate_simulated_dataset(remaining, LOCATIONS)
                records.extend(fallback_df.to_dict("records"))
                break
            time.sleep(cfg["polling_interval_sec"] * 2)
            continue

        records.extend(round_records)

        n = len(records)
        if n % cfg["log_every_n"] < len(round_records) or n >= rows:
            pct = min(100, n / rows * 100)
            log.info(f"  Collecting records… {min(n, rows):,} / {rows:,}  ({pct:.0f}%)")

        if len(records) < rows:
            time.sleep(cfg["polling_interval_sec"])

    df = pd.DataFrame(records[:rows])
    log.info(f"\n✅ Data collection complete. {len(df):,} records.")
    _save_raw(df, out_path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Egypt Traffic — Data Ingestion from TomTom API"
    )
    parser.add_argument(
        "--rows", type=int, default=COLLECTION["rows_to_fetch"],
        help=f"Number of records to collect (default: {COLLECTION['rows_to_fetch']})"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use simulation mode (no API calls)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output CSV path (default: auto-generated timestamped path in data/raw/)"
    )
    args = parser.parse_args()

    df = collect_traffic_data(
        rows_to_fetch=args.rows,
        simulate=args.simulate,
        output_path=args.output,
    )
    print(f"\nSample output:\n{df.head()}")
    print(f"\nLatest raw file: {get_latest_raw_csv()}")
