import os
import sys
import warnings
import json
import time
import glob
import datetime
import requests
warnings.filterwarnings("ignore")

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sqlalchemy import create_engine
import urllib
from streamlit_option_menu import option_menu
import pydeck as pdk

# ── Central config — single source of truth ───────────────────────────────────
from config.config import BASE_DIR, PATHS, LOCATION_COORDS, LOCATION_NAMES

MODELS_DIR  = PATHS["models_dir"]
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = PATHS["plots_dir"]


# ══════════════════════════════════════════════════════════════════════════════
# AZURE SQL  (optional — graceful fallback if unavailable)
# ══════════════════════════════════════════════════════════════════════════════
from dotenv import load_dotenv, find_dotenv
from os import getenv

load_dotenv(find_dotenv(".env"))
_SQL_SERVER   = getenv("_SQL_SERVER" , "")
_SQL_DATABASE = getenv("_SQL_DATABASE","")
_SQL_USER     = getenv("_SQL_USER","")
_SQL_PASSWORD = getenv("_SQL_PASSWORD","")

_sql_params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER=tcp:{_SQL_SERVER},1433;"
    f"DATABASE={_SQL_DATABASE};"
    f"UID={_SQL_USER};"
    f"PWD={_SQL_PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)
_engine = create_engine("mssql+pyodbc:///?odbc_connect=" + _sql_params)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Egypt Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — only active pages listed
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    import base64
    logo_path = os.path.join(os.path.dirname(__file__), "logo.svg")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as _lf:
            _logo_b64 = base64.b64encode(_lf.read()).decode()
        st.markdown(
            f"""
            <div class="sidebar-logo-wrap">
                <img src="data:image/svg+xml;base64,{_logo_b64}" width="100" />
            </div>
            """,
            unsafe_allow_html=True,
        ) 

    st.markdown("---")


    page = option_menu(
        menu_title=None,
        options=["Overview", "EDA & Viz", "Prediction", "Feature Audit"],
        icons=["house", "bar-chart-line", "robot", "magic", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "10px !important", "background-color": "#121a28 !important"},
            "nav-link": {"font-size": "14px", "text-align": "left", "color": "#ffffff", "font-family": "'Syne', sans-serif"},
            "nav-link-selected": {"background-color": "rgba(0, 255, 231, 0.05)", "color": "#00ffe7"}
        }
    )

    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])
    st.markdown(
        "<div style='font-size:0.68rem; color:#334455; margin-top:-10px;'>"
        "Leave empty to use Azure SQL dataset</div>",
        unsafe_allow_html=True,
    )

    if not uploaded:
        st.markdown("---")
        st.markdown("<h3 style='font-size: 1.1rem; color: #00ffe7;'>🔄 Database Sync</h3>", unsafe_allow_html=True)
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=True, help="Periodically pull fresh streaming data from Azure SQL.")
        refresh_interval = st.slider("Refresh Interval (s)", min_value=5, max_value=120, value=15, step=5)
        
        if auto_refresh:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=refresh_interval * 1000, key="db_autorefresh_timer")
            st.markdown(
                f"<div style='font-size:0.75rem; color:#00ffe7; font-family: monospace; margin-top:-10px; margin-bottom:10px;'>"
                f"● Syncing every {refresh_interval}s</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.75rem; color:#8899aa; font-family: monospace; margin-top:-10px; margin-bottom:10px;'>"
                "○ Sync paused</div>",
                unsafe_allow_html=True,
            )
        
        if st.button("Force Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.session_state["initialized"] = False
            st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_peak_hour(h: int) -> int:
    """Return 1 if hour falls within AM (7–9) or PM (16–19) peak windows."""
    return 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0


def _get_csv_mtime() -> float:
    p = PATHS["clean_csv"]
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


def _get_model_mtime() -> float:
    p = os.path.join(MODELS_DIR, "meta.json")
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply temporal feature engineering — shared between CSV and Azure SQL load paths."""
    # Column aliases for Azure SQL schema (region → location_name, avg_speed → speed)
    if "avg_speed" in df.columns and "speed" not in df.columns:
        df = df.rename(columns={"region": "location_name", "avg_speed": "speed"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "coordinates" in df.columns and "lat" not in df.columns:
        coords    = df["coordinates"].str.extract(r"([\d.]+),\s*([\d.]+)")
        df["lat"] = coords[0].astype(float)
        df["lon"] = coords[1].astype(float)

    df["hour"]         = df["timestamp"].dt.hour
    df["minute"]       = df["timestamp"].dt.minute
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].apply(is_peak_hour)

    # Derive traffic_status from speed only when label is absent in source data
    if "traffic_status" not in df.columns and "speed" in df.columns:
        def _speed_to_status(s):
            if s < 20:   return "Congested"
            elif s < 50: return "Moderate"
            return "Free_Flow"
        df["traffic_status"] = df["speed"].apply(_speed_to_status)

    return df


@st.cache_data(show_spinner=False)
def load_data_cached(file_bytes=None, _csv_mtime: float = 0.0, db_cache_key: int = 0) -> pd.DataFrame:
    """Load data from uploaded CSV or Azure SQL, then apply feature engineering.
    _csv_mtime is used as a cache-busting key so stale data is never shown after retraining.
    """
    if file_bytes:
        import io
        df = pd.read_csv(io.BytesIO(file_bytes), parse_dates=["timestamp"])
    else:
        try:
            df = pd.read_sql("SELECT * FROM dbo.traffic", _engine)
        except Exception:
            #st.error("⚠️ Could not connect to Azure SQL. Upload a CSV or check your connection.")
            #st.info(
            #    "Make sure the Stream Analytics Job has populated the dbo.traffic table, "
            #    "or upload a CSV file from the sidebar."
            #)
            return pd.DataFrame(columns=["timestamp", "location_name", "speed", "traffic_status"])

    return _engineer_features(df)


@st.cache_resource(show_spinner=False)
def load_models_cached(_model_mtime: float = 0.0):
    """Load model artifacts from disk.
    Returns (model, scaler, loc_encoder, target_encoder, meta) or (None×5) if not found.
    _model_mtime is used as a cache-busting key so updated artifacts are picked up automatically.
    """
    paths = {
        "model":  os.path.join(MODELS_DIR, "best_model.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
        "le":     os.path.join(MODELS_DIR, "loc_encoder.pkl"),
        "te":     os.path.join(MODELS_DIR, "target_encoder.pkl"),
        "meta":   os.path.join(MODELS_DIR, "meta.json"),
    }
    if not all(os.path.exists(p) for p in list(paths.values())[:4]):
        return None, None, None, None, None

    model  = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    le     = joblib.load(paths["le"])
    te     = joblib.load(paths["te"])
    meta   = json.load(open(paths["meta"], encoding="utf-8")) if os.path.exists(paths["meta"]) else {}
    return model, scaler, le, te, meta


def load_models():
    """Public wrapper — always passes current mtime so hot-reload works."""
    return load_models_cached(_model_mtime=_get_model_mtime())


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOAD  (runs on every page render; result is cached by mtime key)
# ══════════════════════════════════════════════════════════════════════════════
file_bytes = uploaded.read() if uploaded else None

# Dynamic cache buster key based on user's selected refresh interval
if not uploaded:
    interval_s = refresh_interval if 'refresh_interval' in locals() else 15
    db_cache_key = int(time.time() // interval_s)
else:
    db_cache_key = 0

# State-aware spinner logic for background refreshes
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    show_spinner = True
else:
    show_spinner = False

if show_spinner:
    with st.spinner("Loading data…"):
        df = load_data_cached(file_bytes, _csv_mtime=_get_csv_mtime(), db_cache_key=db_cache_key)
else:
    df = load_data_cached(file_bytes, _csv_mtime=_get_csv_mtime(), db_cache_key=db_cache_key)

# Auto-rerun every 60 s so freshly saved model artifacts are detected without
# the user having to manually refresh the page.
if "last_rerun" not in st.session_state:
    st.session_state["last_rerun"] = time.time()
if time.time() - st.session_state["last_rerun"] > 60:
    st.session_state["last_rerun"] = time.time()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    _now = datetime.datetime.now().strftime("%A, %d %b %Y — %H:%M")
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0e1520 0%, #131c2e 100%);
                border: 1px solid #38bdf8; border-radius: 16px;
                padding: 28px 32px; margin-bottom: 24px;
                box-shadow: 0 0 30px rgba(0,255,231,0.07);'>
        <div style='font-size:0.72rem; color:#00ffe7; font-family:"IBM Plex Mono",monospace;
                    letter-spacing:0.15em; text-transform:uppercase;'>{_now}</div>
        <div style='font-size:2.1rem; font-weight:800; color:white; margin-top:10px; font-family:IBM Plex Mono;'>
            🚦 Egypt Traffic Intelligence
        </div>
        <div style='color:#7c8da0; font-size:0.92rem; margin-top:6px;'>
            AI-powered pre-trip congestion prediction &nbsp;·&nbsp; Cairo Road Network &nbsp;·&nbsp;
            <span style='color:#4ade80;'>● Live</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    total     = len(df)
    locations = df["location_name"].nunique() if "location_name" in df.columns else "—"
    avg_speed = df["speed"].mean() if "speed" in df.columns else 0
    free_pct  = (df["traffic_status"] == "Free_Flow").mean() * 100  if "traffic_status" in df.columns else 0
    cong_pct  = (df["traffic_status"] == "Congested").mean() * 100  if "traffic_status" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)

    metric_defs = [
        (c1, f"{total:,}",       "Total Records",  "📊", "#38bdf8", "rgba(56, 189, 248, 0.45)"),
        (c2, str(locations),     "Locations",       "📍", "#38bdf8", "rgba(56, 189, 248, 0.45)"),
        (c3, f"{avg_speed:.1f}", "Avg Speed km/h",  "⚡", "#00ffe7", "rgba(0, 255, 231, 0.45)"),
        (c4, f"{free_pct:.1f}%", "Free Flow",       "🟢", "#4ade80", "rgba(74, 222, 128, 0.45)"),
        (c5, f"{cong_pct:.1f}%", "Congested",       "🔴", "#f87171", "rgba(248, 113, 113, 0.45)"),
    ]
    for col, val, label, icon, accent, glow in metric_defs:
        col.markdown(f"""
        <div class='metric-card' style='--accent:{accent}; --accent-glow:{glow};'>
            <div class='metric-icon-row'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-dot'></div>
            </div>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="section-title">Traffic Status Distribution</div>', unsafe_allow_html=True)
        if "traffic_status" in df.columns:
            counts = df["traffic_status"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                marker_colors=["#f87171", "#facc15", "#4ade80"],
                hole=0.5,
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white", showlegend=True,
                              margin=dict(t=10, b=10, l=10, r=10), height=360)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Live Traffic Map</div>', unsafe_allow_html=True)

        if "lat" in df.columns and "lon" in df.columns:
            def get_rgb(status):
                if status == "Congested": return [248, 113, 113]  # Red
                elif status == "Moderate": return [250, 204, 21]  # Yellow
                else: return [74, 222, 128]                       # Green

            map_df = df.copy()
            rgb_series = (map_df["traffic_status"].apply(get_rgb)
                          if "traffic_status" in map_df.columns
                          else pd.Series([[0, 255, 231]] * len(map_df), index=map_df.index))

            map_df["color_core"] = rgb_series.apply(lambda c: c + [235])   

            dot_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_fill_color="color_core",
                get_radius=90,
                radius_min_pixels=5,
                radius_max_pixels=11,
                stroked=True,
                get_line_color=[255, 255, 255, 180],
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
            )

            view_state = pdk.ViewState(
                latitude=map_df["lat"].mean() if not map_df.empty else 30.0444,
                longitude=map_df["lon"].mean() if not map_df.empty else 31.2357,
                zoom=10.5,
                pitch=35,
                bearing=-10,
            )

            st.pydeck_chart(pdk.Deck(
                map_provider="carto",     
                map_style="dark",         
                initial_view_state=view_state,
                layers=[dot_layer],
                tooltip={"html": "<b>Location:</b> {location_name} <br/> <b>Speed:</b> {speed} km/h <br/> <b>Status:</b> {traffic_status}"}
            ), use_container_width=True, height=360)
        else:
            st.info("📌 To view the map, ensure your dataset includes 'lat' and 'lon' columns.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)

    preview_cols = [c for c in ["timestamp", "location_name", "speed", "traffic_status"] if c in df.columns]
    preview_df   = df[preview_cols].head(10).copy()

    status_icon_map = {"Free_Flow": "🟢 Free Flow", "Moderate": "🟡 Moderate", "Congested": "🔴 Congested"}
    if "traffic_status" in preview_df.columns:
        preview_df["traffic_status"] = preview_df["traffic_status"].map(status_icon_map).fillna(preview_df["traffic_status"])
    if "timestamp" in preview_df.columns:
        preview_df["timestamp"] = pd.to_datetime(preview_df["timestamp"]).dt.strftime("%Y-%m-%d  %H:%M")

    col_config = {}
    if "speed" in preview_df.columns:
        col_config["speed"] = st.column_config.ProgressColumn(
            "Speed (km/h)", min_value=0, max_value=max(80, int(preview_df["speed"].max() or 80)),
            format="%d km/h",
        )
    if "location_name" in preview_df.columns:
        col_config["location_name"] = st.column_config.TextColumn("Location", width="medium")
    if "timestamp" in preview_df.columns:
        col_config["timestamp"] = st.column_config.TextColumn("Timestamp", width="medium")
    if "traffic_status" in preview_df.columns:
        col_config["traffic_status"] = st.column_config.TextColumn("Status", width="medium")

    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        height=38 * (len(preview_df) + 1),
    )



    # ── Model status panel ─────────────────────────────────────────────────
    model, _, _, _, meta = load_models()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Model Status</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)

    def _mini_card(col, label, value, color="#00ffe7", icon="🤖"):
        glow = color.replace("#", "")
        col.markdown(f"""
        <div class='metric-card' style='--accent:{color}; --accent-glow:rgba({int(glow[0:2],16)},{int(glow[2:4],16)},{int(glow[4:6],16)},0.4);'>
            <div class='metric-icon-row'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-dot'></div>
            </div>
            <div style='font-size:0.95rem;font-weight:700;color:{color};line-height:1.4;word-break:break-word;font-family:"IBM Plex Mono",monospace;'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    _best    = meta.get("best_model_name", "—") if meta else "—"
    _trained = (meta.get("trained_at", "—") or "—")[:16].replace("T", " ") if meta else "—"
    _nrec    = f"{meta.get('n_records', 0):,}" if meta else "—"

    _mini_card(sc1, "Active Model",     _best,    "#00ffe7", "🤖")
    _mini_card(sc2, "Training Records", _nrec,    "#38bdf8", "📊")
    _mini_card(sc3, "Model Updated",    _trained, "#4ade80", "🕒")

    if not model:
        st.markdown("""
        <div class='card' style='border-top:3px solid #f87171; margin-top:12px;'>
            <b style='color:#f87171;'>⚠ No trained model found</b><br>
            <span style='color:#556677;font-size:0.85rem;'>
            Run <code>python models/train_models.py --rows 2000 --simulate</code> to do a first training cycle.
            </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    STATUS_COLORS = {"Congested": "#f87171", "Moderate": "#facc15", "Free_Flow": "#4ade80"}
    ACCENT = "#00ffe7"

    def _style_fig(fig, height=320, legend=True):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d6e5", family="IBM Plex Mono, monospace", size=12),
            title_font=dict(family="Syne, sans-serif", size=15, color="#e8eef5"),
            height=height, showlegend=legend,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=48, b=30, l=10, r=10),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
        return fig

    if df.empty:
        st.info("No data loaded yet — upload a CSV from the sidebar to explore.")
        st.stop()

    # ── Filter bar ────────────────────────────────
    with st.container(border=True):
        fc1, fc2 = st.columns([3, 1])
        with fc1:
            all_locs = sorted(df["location_name"].dropna().unique().tolist()) if "location_name" in df.columns else []
            sel_locs = st.multiselect("📍 Filter by Location (leave empty = all)", all_locs, default=[])
        with fc2:
            st.markdown(f"""
            <div style='padding-top:1.8rem;'>
                <span class='status-pill'><b style='color:{ACCENT};'>{len(df):,}</b>&nbsp;total rows</span>
            </div>
            """, unsafe_allow_html=True)

    fdf = df[df["location_name"].isin(sel_locs)] if sel_locs else df
    st.markdown(f"<div style='color:#556677;font-size:0.78rem;margin:2px 0 10px;font-family:\"IBM Plex Mono\",monospace;'>Showing <b style='color:{ACCENT};'>{len(fdf):,}</b> / {len(df):,} records{' — filtered by ' + str(len(sel_locs)) + ' location(s)' if sel_locs else ''}</div>", unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpis = []
    if "hour" in fdf.columns and "speed" in fdf.columns and not fdf.empty:
        peak_h = int(fdf.groupby("hour")["speed"].mean().idxmin())
        kpis.append((k1, f"{peak_h:02d}:00", "Slowest Hour", "🐢", "#f87171", "rgba(248,113,113,0.45)"))
    if "location_name" in fdf.columns and "traffic_status" in fdf.columns and not fdf.empty:
        cong_by_loc = fdf[fdf["traffic_status"] == "Congested"]["location_name"].value_counts()
        busiest = cong_by_loc.idxmax() if not cong_by_loc.empty else "—"
        kpis.append((k2, busiest, "Most Congested Road", "🚧", "#f87171", "rgba(248,113,113,0.45)"))
    if "day_of_week" in fdf.columns and "traffic_status" in fdf.columns and not fdf.empty:
        days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        cong_by_day = fdf[fdf["traffic_status"] == "Congested"]["day_of_week"].value_counts()
        worst_day = days_map.get(int(cong_by_day.idxmax()), "—") if not cong_by_day.empty else "—"
        kpis.append((k3, worst_day, "Worst Day", "📅", "#facc15", "rgba(250,204,21,0.45)"))
    if "speed" in fdf.columns and not fdf.empty:
        kpis.append((k4, f"±{fdf['speed'].std():.1f}", "Speed Std. Dev.", "📈", ACCENT, "rgba(0,255,231,0.45)"))
    for col, val, label, icon, accent, glow in kpis:
        col.markdown(f"""
        <div class='metric-card' style='--accent:{accent}; --accent-glow:{glow};'>
            <div class='metric-icon-row'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-dot'></div>
            </div>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Distributions", "Temporal Trends", "Location Analysis", "Correlation & Patterns"])

    # ═══════════════════════════════ TAB 1 — DISTRIBUTIONS ═══════════════════
    with t1:
        with st.container(border=True):
            if "speed" in fdf.columns:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(fdf, x="speed", nbins=40, color_discrete_sequence=[ACCENT],
                                       title="Speed Distribution (exploratory only)", marginal="box")
                    med = fdf["speed"].median()
                    fig.add_vline(x=med, line_dash="dash", line_color="#facc15",
                                  annotation_text=f"median {med:.0f}", annotation_font_color="#facc15")
                    st.plotly_chart(_style_fig(fig, legend=False), use_container_width=True)
                with c2:
                    if "traffic_status" in fdf.columns:
                        counts = fdf["traffic_status"].value_counts().reset_index()
                        counts.columns = ["traffic_status", "count"]
                        total_n = counts["count"].sum()
                        fig = go.Figure(go.Pie(
                            labels=counts["traffic_status"], values=counts["count"], hole=0.6,
                            marker_colors=[STATUS_COLORS.get(s, ACCENT) for s in counts["traffic_status"]],
                            textinfo="label+percent",
                        ))
                        fig.add_annotation(text=f"<b>{total_n:,}</b><br><span style='font-size:0.7em;color:#8899aa'>records</span>",
                                            showarrow=False, font=dict(size=16, color="white"))
                        fig.update_layout(title="Traffic Status Share")
                        st.plotly_chart(_style_fig(fig, legend=True), use_container_width=True)
            elif "traffic_status" in fdf.columns:
                counts = fdf["traffic_status"].value_counts().reset_index()
                counts.columns = ["traffic_status", "count"]
                fig = px.bar(counts, x="traffic_status", y="count", color="traffic_status",
                             color_discrete_map=STATUS_COLORS, title="Traffic Status Counts")
                st.plotly_chart(_style_fig(fig, legend=False), use_container_width=True)
            else:
                st.info("No speed or traffic_status column found in the loaded dataset.")

    # ═══════════════════════════════ TAB 2 — TEMPORAL TRENDS ═════════════════
    with t2:
        with st.container(border=True):
            if "hour" in fdf.columns and "speed" in fdf.columns:
                hourly = fdf.groupby("hour")["speed"].mean().reset_index()
                fig = px.line(hourly, x="hour", y="speed", markers=True,
                              title="Average Speed by Hour of Day (exploratory only)",
                              color_discrete_sequence=[ACCENT])
                fig.update_traces(fill="tozeroy", fillcolor="rgba(0,255,231,0.08)", line_width=3)
                fig.add_vrect(x0=6.5, x1=9.5,  fillcolor="#f87171", opacity=0.12, line_width=0, annotation_text="AM Peak", annotation_font_color="#f87171")
                fig.add_vrect(x0=15.5, x1=19.5, fillcolor="#f87171", opacity=0.12, line_width=0, annotation_text="PM Peak", annotation_font_color="#f87171")
                st.plotly_chart(_style_fig(fig, height=340, legend=False), use_container_width=True)

        if "hour" in fdf.columns and "day_of_week" in fdf.columns and "traffic_status" in fdf.columns:
            with st.container(border=True):
                days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
                sev_map = {"Free_Flow": 0, "Moderate": 1, "Congested": 2}
                hm = fdf.copy()
                hm["sev"] = hm["traffic_status"].map(sev_map)
                pivot = hm.pivot_table(index="day_of_week", columns="hour", values="sev", aggfunc="mean")
                pivot = pivot.reindex(range(7))
                pivot.index = [days_map[d] for d in pivot.index]

                fig = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index,
                    colorscale=[[0, "#4ade80"], [0.5, "#facc15"], [1, "#f87171"]],
                    colorbar=dict(title="Congestion", tickvals=[0, 1, 2], ticktext=["Free", "Moderate", "Congested"],
                                  outlinewidth=0, tickfont=dict(color="#c8d6e5")),
                    hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Severity: %{z:.2f}<extra></extra>",
                ))
                fig.update_layout(title="Congestion Heatmap — Hour × Day of Week")
                st.plotly_chart(_style_fig(fig, height=360, legend=False), use_container_width=True)

        if "day_of_week" in fdf.columns and "traffic_status" in fdf.columns:
            with st.container(border=True):
                days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
                dow  = fdf.groupby(["day_of_week", "traffic_status"]).size().reset_index(name="count")
                dow["day"] = dow["day_of_week"].map(days_map)
                fig = px.bar(dow, x="day", y="count", color="traffic_status",
                             color_discrete_map=STATUS_COLORS,
                             category_orders={"day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
                             title="Traffic Status by Day of Week", barmode="group")
                st.plotly_chart(_style_fig(fig, height=320), use_container_width=True)

        if "hour" not in fdf.columns and "speed" not in fdf.columns:
            st.info("Temporal columns (hour, speed) not found in the loaded dataset.")

    # ═══════════════════════════════ TAB 3 — LOCATION ANALYSIS ═══════════════
    with t3:
        if "location_name" in fdf.columns and "traffic_status" in fdf.columns:
            grp = fdf.groupby(["location_name", "traffic_status"]).size().reset_index(name="count")
            totals = grp.groupby("location_name")["count"].sum().rename("total")
            cong = (grp[grp["traffic_status"] == "Congested"]
                    .set_index("location_name")["count"]
                    .reindex(totals.index, fill_value=0))
            rank = pd.DataFrame({"total": totals, "congested": cong})
            rank["congestion_pct"] = (rank["congested"] / rank["total"] * 100).round(1)
            rank = rank.sort_values("congestion_pct", ascending=False)

            with st.container(border=True):
                fig = px.bar(
                    rank.reset_index(), x="congestion_pct", y="location_name", orientation="h",
                    color="congestion_pct", color_continuous_scale=["#4ade80", "#facc15", "#f87171"],
                    title="Roads Ranked by Congestion Rate (%)",
                    labels={"congestion_pct": "Congested %", "location_name": ""},
                )
                fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
                st.plotly_chart(_style_fig(fig, height=max(320, 32 * len(rank)), legend=False), use_container_width=True)

            with st.container(border=True):
                grp["location_name"] = pd.Categorical(grp["location_name"], categories=rank.index, ordered=True)
                grp = grp.sort_values("location_name")
                fig = px.bar(grp, x="location_name", y="count", color="traffic_status",
                             color_discrete_map=STATUS_COLORS,
                             title="Traffic Status Breakdown per Location", barmode="stack")
                fig.update_xaxes(tickangle=-30)
                st.plotly_chart(_style_fig(fig, height=380), use_container_width=True)

            # ── Ranking table ────────────────────
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("<div class='section-title' style='font-size:1rem;'>🔴 Most Congested Roads</div>", unsafe_allow_html=True)
                for loc, row in rank.head(3).iterrows():
                    st.markdown(f"""
                    <div class='card' style='padding:12px 18px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;border-left:3px solid #f87171;'>
                        <span style='color:#e8eef5;font-weight:700;'>{loc}</span>
                        <span class='badge-cong'>{row['congestion_pct']:.1f}%</span>
                    </div>""", unsafe_allow_html=True)
            with rc2:
                st.markdown("<div class='section-title' style='font-size:1rem;'>🟢 Smoothest Roads</div>", unsafe_allow_html=True)
                for loc, row in rank.tail(3).sort_values("congestion_pct").iterrows():
                    st.markdown(f"""
                    <div class='card' style='padding:12px 18px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;border-left:3px solid #4ade80;'>
                        <span style='color:#e8eef5;font-weight:700;'>{loc}</span>
                        <span class='badge-free'>{row['congestion_pct']:.1f}%</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("Location or traffic_status column not found in the loaded dataset.")

    # ═══════════════════════════════ TAB 4 — CORRELATION & PATTERNS ══════════
    with t4:
        num_cols = [c for c in ["hour", "minute", "day_of_week", "is_weekend", "is_peak_hour", "speed"] if c in fdf.columns]
        if len(num_cols) >= 2:
            with st.container(border=True):
                corr = fdf[num_cols].corr().round(2)
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.columns,
                    colorscale=[[0, "#f87171"], [0.5, "#0e1520"], [1, ACCENT]],
                    zmid=0, zmin=-1, zmax=1,
                    text=corr.values, texttemplate="%{text}",
                    textfont=dict(color="#e8eef5", size=11),
                    colorbar=dict(outlinewidth=0, tickfont=dict(color="#c8d6e5")),
                ))
                fig.update_layout(title="Feature Correlation Matrix")
                st.plotly_chart(_style_fig(fig, height=380, legend=False), use_container_width=True)

        if "speed" in fdf.columns and "traffic_status" in fdf.columns:
            with st.container(border=True):
                fig = px.box(fdf, x="traffic_status", y="speed", color="traffic_status",
                             color_discrete_map=STATUS_COLORS,
                             category_orders={"traffic_status": ["Free_Flow", "Moderate", "Congested"]},
                             title="Speed Distribution per Traffic Status (exploratory only)", points="outliers")
                st.plotly_chart(_style_fig(fig, height=340, legend=False), use_container_width=True)

        if "is_peak_hour" in fdf.columns and "traffic_status" in fdf.columns:
            with st.container(border=True):
                pk = fdf.groupby(["is_peak_hour", "traffic_status"]).size().reset_index(name="count")
                pk["Period"] = pk["is_peak_hour"].map({0: "Off-Peak", 1: "Peak Hour"})
                fig = px.bar(pk, x="Period", y="count", color="traffic_status",
                             color_discrete_map=STATUS_COLORS, barmode="group",
                             title="Peak vs Off-Peak — Traffic Status Comparison")
                st.plotly_chart(_style_fig(fig, height=320), use_container_width=True)

        if len(num_cols) < 2:
            st.info("Not enough numeric columns available to compute correlations.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif "Prediction" in page:
    st.markdown('<div class="section-title">Pre-Trip Traffic Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <b>Pre-Trip Planner</b> — Enter the details you already know <em>before leaving home</em>.
    The model predicts whether your chosen road will be congested at that time.<br><br>
    <span style='font-size:0.82rem;color:#556677;'>
    No real-time data needed &nbsp;|&nbsp; No vehicle counts &nbsp;|&nbsp;
    Just pick your route + planned departure time
    </span>
    </div>
    """, unsafe_allow_html=True)

    STATUS_META = {
        "Free_Flow": {"icon": "✅", "color": "#4ade80", "glow": "rgba(74, 222, 128, 0.45)",
                      "badge": "badge-free", "label": "Free Flow", "pill": "status-pill-good",
                      "advice": "Great time to travel! Roads should be clear."},
        "Moderate":  {"icon": "⚠️", "color": "#facc15", "glow": "rgba(250, 204, 21, 0.45)",
                      "badge": "badge-mod", "label": "Moderate", "pill": "status-pill-warn",
                      "advice": "Expect some traffic. Allow 10–20 extra minutes."},
        "Congested": {"icon": "🔴", "color": "#f87171", "glow": "rgba(248, 113, 113, 0.45)",
                      "badge": "badge-cong", "label": "Congested", "pill": "status-pill-bad",
                      "advice": "Heavy congestion expected. Consider delaying or using alternative routes."},
    }
    STATUS_ORDER = ["Free_Flow", "Moderate", "Congested"]
    STATUS_RANK  = {"Free_Flow": 0, "Moderate": 1, "Congested": 2}

    ROUTE_ICON = "🛣️"

    # ── Mode toggles — Compare / Live (Ideas #1 & #5) ──────────────────────
    mt1, mt2 = st.columns(2)
    with mt1:
        compare_mode = st.toggle("Compare Multiple Routes", value=False, key="compare_mode")
    with mt2:
        live_mode = st.toggle("Live Prediction (auto-update on change)", value=False, key="live_mode")

    with st.container(border=True):
        if compare_mode:
            sel_locations = st.multiselect(
                f"{ROUTE_ICON} Roads / Junctions to Compare (2–3)",
                LOCATION_NAMES,
                default=LOCATION_NAMES[:2],
                max_selections=3,
            )
            location = sel_locations[0] if sel_locations else (LOCATION_NAMES[0] if LOCATION_NAMES else None)
        else:
            location = st.selectbox(f"{ROUTE_ICON} Road / Junction", LOCATION_NAMES)
            sel_locations = [location] if location else []

        ic2, ic3, ic4 = st.columns(3)
        with ic2:
            hour = st.slider("🕐 Departure Hour", 0, 23, 8)
        with ic3:
            minute = st.slider("🕑 Minute", 0, 59, 0)
        with ic4:
            day_name = st.selectbox("📅 Day of Week",
                                     ["Monday", "Tuesday", "Wednesday", "Thursday",
                                      "Friday", "Saturday", "Sunday"])

        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                   "Friday": 4, "Saturday": 5, "Sunday": 6}
        dow     = day_map[day_name]
        is_wknd = 1 if dow >= 5 else 0
        is_pk   = is_peak_hour(hour)

        peak_pill  = "status-pill-bad" if is_pk else "status-pill-good"
        wknd_pill  = "status-pill-warn" if is_wknd else "status-pill-good"
        st.markdown(f"""
        <div style='margin-top:4px;'>
            <span class='status-pill {peak_pill}'>{"🔴" if is_pk else "🟢"} <b style='color:{"#f87171" if is_pk else "#4ade80"};'>{"Peak Hour" if is_pk else "Off-Peak"}</b></span>
            <span class='status-pill {wknd_pill}'>{"🏖" if is_wknd else "💼"} <b>{"Weekend" if is_wknd else "Weekday"}</b></span>
            <span class='status-pill'>🕐 <b>{day_name} {hour:02d}:{minute:02d}</b></span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:14px;' class='predict-btn-wrap'>", unsafe_allow_html=True)
        bcol1, bcol2 = st.columns([2, 1])
        with bcol1:
            predict_clicked = st.button(
                "Predict Traffic Status", use_container_width=True, disabled=live_mode
            )
        with bcol2:
            find_best_clicked = st.button(
                "Find Best Time", use_container_width=True, disabled=compare_mode
            )
        st.markdown("</div>", unsafe_allow_html=True)
        if compare_mode:
            st.caption("ℹ️ 'Find Best Time' works on a single route — turn off Compare mode to use it.")

    should_predict = (live_mode or predict_clicked) and bool(sel_locations)

    model, scaler, le, te, meta = load_models()

    def _predict_one(loc: str, h: int, m: int, d: int, wknd: int, pk: int):
        """Run the model for a single (location, time) combination."""
        lat, lon = LOCATION_COORDS.get(loc, (30.05, 31.23))
        loc_enc  = le.transform([loc])[0] if loc in le.classes_ else 0
        X_input  = np.array([[h, m, d, wknd, pk, lat, lon, loc_enc]])
        X_sc     = scaler.transform(X_input)
        pred_enc = model.predict(X_sc)[0]
        pred_prob = model.predict_proba(X_sc)[0] if hasattr(model, "predict_proba") else None
        pred_lbl  = te.inverse_transform([pred_enc])[0]
        conf      = pred_prob.max() * 100 if pred_prob is not None else 0.0

        return pred_lbl, conf, pred_prob, lat, lon
        

    def _historical_context(loc: str, h: int, d: int, pred_lbl: str):
        """Idea #3 — how often was the *actual* historical data this status
        at similar hours / same day-of-week for this road?"""
        if df is None or df.empty or "traffic_status" not in df.columns or "hour" not in df.columns:
            return None
        sub = df[(df.get("location_name") == loc) &
                  (df["hour"].between(max(h - 1, 0), min(h + 1, 23))) &
                  (df.get("day_of_week") == d)]
        if len(sub) < 5:
            sub = df[(df.get("location_name") == loc) &
                      (df["hour"].between(max(h - 1, 0), min(h + 1, 23)))]
        if sub is None or len(sub) == 0:
            return None
        pct = (sub["traffic_status"] == pred_lbl).mean() * 100
        return pct, len(sub)

    if (should_predict or find_best_clicked) and model is None:
        st.error("⚠️ No trained model found. Run `python models/train_models.py` first.")

    # ══════════════════════════════════════════════════════════════════════
    # COMPARE MODE — Idea #1
    # ══════════════════════════════════════════════════════════════════════
    elif compare_mode and should_predict:
        with st.spinner("Comparing routes…"):
            results = []
            for loc in sel_locations:
                pred_lbl, conf, pred_prob, lat, lon = _predict_one(loc, hour, minute, dow, is_wknd, is_pk)
                results.append({"loc": loc, "label": pred_lbl, "conf": conf, "lat": lat, "lon": lon})

        best_loc = min(results, key=lambda r: (STATUS_RANK.get(r["label"], 9), -r["conf"]))["loc"]

        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(len(results))
        for col, r in zip(cols, results):
            mi = STATUS_META.get(r["label"], STATUS_META["Moderate"])
            is_best = r["loc"] == best_loc
            with col:
                st.markdown(f"""
                <div class='compare-card' style='--accent:{mi["color"]}; --accent-glow:{mi["glow"]};'>
                    {"<div class='compare-best-badge'>⭐ FASTEST</div>" if is_best else ""}<div style='font-size:1.6rem;'>{mi["icon"]}</div>
                    <div style='font-weight:800; font-family:Syne,sans-serif; color:#e2e8f0; margin:6px 0 2px;'>{ROUTE_ICON} {r["loc"]}</div>
                    <span class='{mi["badge"]}'>&nbsp;{mi["label"]}&nbsp;</span>
                    <div class='result-confidence'>Confidence: <b style='color:{mi["color"]};'>{r["conf"]:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top:14px;'>
            <span class='best-hour-badge'>⭐ Best route right now: <b>{best_loc}</b></span>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SINGLE-ROUTE PREDICTION
    # ══════════════════════════════════════════════════════════════════════
    elif not compare_mode and should_predict:
        with st.spinner("Running prediction model…"):
            pred_lbl, confidence, pred_prob, lat, lon = _predict_one(
                location, hour, minute, dow, is_wknd, is_pk
            )
        meta_info = STATUS_META.get(pred_lbl, STATUS_META["Moderate"])

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown(f"""
            <div class='result-hero' style='--accent:{meta_info["color"]}; --accent-glow:{meta_info["glow"]};'>
                <div class='result-icon-badge'>{meta_info["icon"]}</div>
                <span class='{meta_info["badge"]}'>&nbsp;{meta_info["label"]}&nbsp;</span>
                <div class='result-confidence'>Confidence: <b style='color:{meta_info["color"]};'>{confidence:.1f}%</b></div>
                <div class='result-advice'>{meta_info["advice"]}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            if pred_prob is not None:
                prob_df = pd.DataFrame({"Status": te.classes_, "Probability": pred_prob})
                prob_df["order"] = prob_df["Status"].apply(lambda s: STATUS_ORDER.index(s) if s in STATUS_ORDER else 99)
                prob_df = prob_df.sort_values("order", ascending=False)
                bar_colors = [STATUS_META.get(s, {}).get("color", "#00ffe7") for s in prob_df["Status"]]
                bar_labels = [STATUS_META.get(s, {}).get("label", s) for s in prob_df["Status"]]

                fig = go.Figure(go.Bar(
                    x=prob_df["Probability"], y=bar_labels, orientation="h",
                    marker_color=bar_colors,
                    text=[f"{p*100:.1f}%" for p in prob_df["Probability"]],
                    textposition="outside", textfont=dict(color="#c8d6e5"),
                ))
                fig.update_layout(
                    title="Confidence Breakdown",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white", height=300,
                    xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    margin=dict(t=40, b=20, l=20, r=40),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Idea #3 — Historical context under the result
        hist = _historical_context(location, hour, dow, pred_lbl)
        if hist is not None:
            hist_pct, hist_n = hist
            st.markdown(f"""
            <div class='card' style='margin-top:10px; border-left:3px solid {meta_info["color"]};'>
                <b style='color:{meta_info["color"]};'>📊 Historical Context</b><br>
                <span style='color:#aebccb; font-size:0.88rem;'>
                Out of <b>{hist_n}</b> historical records around {hour:02d}:00 on {day_name}s at this road,
                the traffic was actually <b style='color:{meta_info["color"]};'>{meta_info["label"]}</b>
                <b style='color:{meta_info["color"]};'>{hist_pct:.1f}%</b> of the time.
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top:6px;'>
            <span class='status-pill'>📍 <b style='color:#00ffe7;'>{location}</b></span>
            <span class='status-pill'>🕐 <b>{day_name} {hour:02d}:{minute:02d}</b></span>
            <span class='status-pill {peak_pill}'>{"🔴" if is_pk else "🟢"} <b>{"Peak hour" if is_pk else "Off-peak"}</b></span>
            <span class='status-pill {wknd_pill}'>{"🏖" if is_wknd else "💼"} <b>{"Weekend" if is_wknd else "Weekday"}</b></span>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # FIND BEST DEPARTURE TIME — Idea #2
    # ══════════════════════════════════════════════════════════════════════
    if find_best_clicked and not compare_mode and model is not None and location:
        with st.spinner("Scanning all 24 hours…"):
            hour_rows = []
            for h in range(24):
                pk_h = is_peak_hour(h)
                lbl_h, conf_h, _, _, _ = _predict_one(location, h, minute, dow, is_wknd, pk_h)
                hour_rows.append({"hour": h, "status": lbl_h, "confidence": conf_h,
                                   "rank": STATUS_RANK.get(lbl_h, 9)})
            hb_df = pd.DataFrame(hour_rows)

        best_row  = hb_df.sort_values(["rank", "confidence"], ascending=[True, False]).iloc[0]
        worst_row = hb_df.sort_values(["rank", "confidence"], ascending=[False, False]).iloc[0]

        st.markdown('<div class="section-title" style="margin-top:20px;">🔍 Best Departure Time — Next 24 Hours</div>', unsafe_allow_html=True)

        with st.container(border=True):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hb_df["hour"], y=hb_df["confidence"], mode="lines+markers",
                line=dict(color="#334455", width=2),
                marker=dict(size=10, color=[STATUS_META.get(s, {}).get("color", "#00ffe7") for s in hb_df["status"]]),
                text=[STATUS_META.get(s, {}).get("label", s) for s in hb_df["status"]],
                hovertemplate="Hour %{x}:00 — %{text} (%{y:.1f}%%)<extra></extra>",
                showlegend=False,
            ))
            fig2.add_trace(go.Scatter(
                x=[best_row["hour"]], y=[best_row["confidence"]], mode="markers+text",
                marker=dict(size=22, color="#00ffe7", symbol="star", line=dict(color="white", width=1)),
                text=["⭐"], textposition="top center", showlegend=False,
                hovertemplate=f"Best: %{{x}}:00<extra></extra>",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=340,
                xaxis=dict(title="Hour of Day", dtick=1, gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Model Confidence (%)", gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

            bm = STATUS_META.get(best_row["status"], STATUS_META["Moderate"])
            wm = STATUS_META.get(worst_row["status"], STATUS_META["Moderate"])
            st.markdown(f"""
            <div>
                <span class='best-hour-badge'>⭐ Best: <b>{int(best_row["hour"]):02d}:00</b> — {bm["icon"]} {bm["label"]}</span>
                <span class='status-pill status-pill-bad' style='margin-left:8px;'>⛔ Worst: <b>{int(worst_row["hour"]):02d}:00</b> — {wm["icon"]} {wm["label"]}</span>
            </div>
            """, unsafe_allow_html=True)
    elif find_best_clicked and compare_mode:
        st.warning("Turn off '🔀 Compare Multiple Routes' to use 🔍 Find Best Time.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE AUDIT
# ══════════════════════════════════════════════════════════════════════════════
elif "Feature Audit" in page:
    st.markdown('<div class="section-title">Feature Audit — Pre-Trip Realism Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <b>Goal:</b> A user predicts road congestion <em>before leaving home</em>.
    Therefore every feature the model uses must be something the user can know or derive
    without any real-time data.
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("hour",            "✅ KEPT",    "User knows their planned departure hour."),
        ("minute",          "✅ KEPT",    "User knows their planned departure minute."),
        ("day_of_week",     "✅ KEPT",    "User knows what day of the week they are travelling."),
        ("is_weekend",      "✅ KEPT",    "Derived from day_of_week — always available."),
        ("is_peak_hour",    "✅ KEPT",    "Derived from hour — flags 7-9 AM and 4-7 PM rush windows."),
        ("lat / lon",       "✅ KEPT",    "User picks a road from a dropdown; coordinates are fixed per location."),
        ("loc_encoded",     "✅ KEPT",    "Label encoding of location name — same info, different format."),
        ("vehicle_count",   "❌ REMOVED", "User has NO way to know real-time vehicle counts from home."),
        ("speed",           "❌ REMOVED", "Speed is what determines congestion — using it as a feature is data leakage. Also unknown before travel."),
        ("traffic_density", "❌ REMOVED", "Was vehicle_count / speed — derived from both removed features."),
    ]

    for feat, status, reason in features:
        kept         = "KEPT" in status
        border_color = "#1e4d2b" if kept else "#4d1e1e"
        status_color = "#4ade80" if kept else "#f87171"
        st.markdown(f"""
        <div class='card' style='border-left:4px solid {border_color}; padding:14px 20px; margin-bottom:8px;'>
            <span style='color:{status_color}; font-weight:800; font-family:IBM Plex Mono,monospace;'> {feat}</span>
            <span style='color:#334455; font-size:0.75rem; margin-left:10px;'>{status}</span><br>
            <span style='color:#8899aa; font-size:0.88rem;'>{reason}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='margin-top:20px; border-top:3px solid #00ffe7;'>
    <b style='color:#00ffe7;'>💡 Recommendations for Improving Prediction Quality</b><br><br>
    <ol style='color:#8899aa; font-size:0.9rem; line-height:1.9;'>
        <li><b style='color:#c8d6e5;'>Add weather data</b> — rain/fog significantly affects Cairo traffic and can be fetched via a free API before travel.</li>
        <li><b style='color:#c8d6e5;'>Add public holidays / special events</b> — Eid, national days, and football matches cause abnormal congestion patterns.</li>
        <li><b style='color:#c8d6e5;'>Add historical congestion averages</b> — e.g., "average speed on this road at this hour on Thursdays".</li>
        <li><b style='color:#c8d6e5;'>Seasonal / month feature</b> — traffic patterns differ across Ramadan, summer school holidays, and winter.</li>
        <li><b style='color:#c8d6e5;'>Route-level features</b> — number of intersections, road type (highway vs arterial) if available.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#1e2d42; font-size:0.72rem; margin-top:50px; padding:18px;
            border-top: 1px solid #1e2535; font-family: IBM Plex Mono, monospace;'>
    🚦 Egypt Traffic Intelligence &nbsp;|&nbsp; Cairo 2026
</div>
""", unsafe_allow_html=True)