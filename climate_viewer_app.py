#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Metrics Viewer
======================
climate_viewer_app.py
Last Updated: 2025-12-02 12:00 AEST - Fixed toggle behaviour, optimised caching

Architecture:
- ALL data loaded ONCE at startup (raw and BC variants)
- Processing applied on-demand based on toggle state
- Processed data cached in session_state for instant toggle switching
- Tabs are DISPLAY ONLY - no calculations

Toggle Behaviour (when ALL OFF = raw data):
- BC toggle OFF: Use Value column (raw model output)
- BC toggle ON: Use Value_BC column (bias-corrected)
- Smooth toggle: Apply/skip rolling average
- Align toggle: Apply/skip AGCD alignment (only with smoothing)

Caching Strategy:
- Base data (raw/BC) loaded once at startup
- Each toggle combination cached separately
- Year range filtering applied AFTER cache lookup (instant)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from climate_viewer_constants import (
    MODE, TIME_HORIZONS, METRIC_PRIORITIES, SEASONS, BASELINE_PERIOD,
    EMOJI, DEFAULT_START_YEAR, DEFAULT_END_YEAR, YEAR_BUFFER, DEFAULT_SCENARIOS_ON,
    WARMING_THRESHOLDS, AMPLIFICATION_FACTOR, get_preindustrial_baseline
)
from climate_viewer_data_operations import (
    resolve_folder,
    discover_scenarios,
    load_metrics_file,
    load_minimal_metadata,
    apply_smoothing,
    align_smoothed_to_agcd,
)
from climate_viewer_tab_metrics import render_metrics_tab
from climate_viewer_tab_dashboard import render_dashboard_tab
from climate_viewer_tab_global import render_global_tab
from climate_viewer_tab_guide import render_user_guide

# ============================================================================
# CONSTANTS
# ============================================================================

GLOBAL_THRESHOLDS = [1.5, 2.0, 2.5]
DEFAULT_SMOOTH_WIN = 20


# ============================================================================
# THRESHOLD CROSSING CALCULATION
# ============================================================================

def calculate_threshold_crossings(
        df: pd.DataFrame,
        location: str,
        baseline: float,
        thresholds: List[float] = GLOBAL_THRESHOLDS,
        amplification: float = AMPLIFICATION_FACTOR
) -> Dict[str, Dict[float, Optional[int]]]:
    """
    Calculate when each scenario crosses each global warming threshold.

    Args:
        df: DataFrame with smoothed+aligned data (Value column)
        location: Location to analyse
        baseline: Pre-industrial baseline temperature for this location
        thresholds: List of global warming thresholds (e.g. [1.5, 2.0, 2.5])
        amplification: Regional amplification factor

    Returns:
        Dict[scenario -> Dict[threshold -> crossing_year or None]]
    """
    results = {}

    # Get scenarios (exclude AGCD and historical)
    all_scenarios = df["Scenario"].unique()
    scenarios = [s for s in all_scenarios
                 if "agcd" not in s.lower() and "historical" not in s.lower()]

    for scenario in scenarios:
        results[scenario] = {}

        # Get annual average temperature for this scenario/location
        scen_data = df[
            (df["Scenario"] == scenario) &
            (df["Location"] == location) &
            (df["Type"] == "Temp") &
            (df["Name"] == "Average") &
            (df["Season"] == "Annual")
            ].copy()

        if scen_data.empty:
            for t in thresholds:
                results[scenario][t] = None
            continue

        # Sort by year
        scen_data = scen_data.sort_values("Year")

        # Calculate global warming from regional temperature
        scen_data["Regional_Warming"] = scen_data["Value"] - baseline
        scen_data["Global_Warming"] = scen_data["Regional_Warming"] / amplification

        for threshold in thresholds:
            # Find first year where global warming >= threshold
            crossing = scen_data[scen_data["Global_Warming"] >= threshold]

            if not crossing.empty:
                results[scenario][threshold] = int(crossing.iloc[0]["Year"])
            else:
                results[scenario][threshold] = None

    return results


# ============================================================================
# DATA LOADING AND PRE-PROCESSING
# ============================================================================

@st.cache_data(show_spinner=False)
def load_parquet_cached(path: str, mtime: float) -> pd.DataFrame:
    """Load parquet file with caching based on path and modification time."""
    return pd.read_parquet(path, engine="pyarrow")


def load_and_process_all_data(
        labels: List[str],
        label_to_path: Dict[str, str]
) -> Dict:
    """
    Load all data from parquet files.

    Returns:
        - df_raw: Raw values (Value column)
        - df_bc: Bias-corrected values (Value_BC -> Value)
        - baselines: Pre-industrial baselines from config

    Smoothing and alignment are applied on-demand based on user toggles.
    Uses disk cache for instant reloads.
    """
    import hashlib
    import pickle
    from pathlib import Path

    # Create cache key from file modification times
    cache_parts = []
    for label in sorted(labels):
        path = label_to_path[label]
        mtime = os.path.getmtime(path)
        cache_parts.append(f"{label}:{mtime}")
    cache_parts.append("v5")  # Cache version - incremented for new structure
    cache_key = hashlib.md5("|".join(cache_parts).encode()).hexdigest()[:12]

    # Cache file location
    cache_dir = Path(label_to_path[labels[0]]).parent.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"viewer_cache_{cache_key}.pkl"

    # Try to load from cache
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            required_keys = ["df_raw", "df_bc", "baselines"]
            if all(k in cached for k in required_keys):
                return cached
        except Exception:
            pass

    # Load all files once
    all_dfs = []
    for label in labels:
        path = label_to_path[label]
        mtime = os.path.getmtime(path)
        df = load_parquet_cached(path, mtime)

        if 'Metric' in df.columns and 'Name' not in df.columns:
            df = df.rename(columns={'Metric': 'Name'})

        df["Scenario"] = label
        all_dfs.append(df)

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Variant 1: Raw (use Value column as-is)
    df_raw = df_combined.copy()

    # Variant 2: BC (Value_BC -> Value)
    df_bc = df_combined.copy()
    if "Value_BC" in df_bc.columns:
        df_bc["Value"] = df_bc["Value_BC"]

    # Get all locations
    locations = df_combined["Location"].dropna().unique()

    # Get baselines from config
    baselines = {}
    for loc in locations:
        baseline = get_preindustrial_baseline(loc)
        if baseline is not None:
            baselines[loc] = baseline

    result = {
        "df_raw": df_raw,
        "df_bc": df_bc,
        "baselines": baselines,
    }

    # Save to cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


# ============================================================================
# UI HELPERS
# ============================================================================

def multi_selector(
        container,
        label: str,
        options: List[str],
        default: Optional[List[str]] = None,
        columns: int = 1,
        namespace: str = "",
        key_prefix: str = "sel"
) -> List[str]:
    """Create multi-select checkboxes."""
    if default is None:
        default = []

    selected = []

    if columns > 1:
        cols = container.columns(columns)
        for i, opt in enumerate(options):
            col = cols[i % columns]
            key = f"{namespace}_{key_prefix}_{opt}"
            default_val = opt in default
            if col.checkbox(opt, value=default_val, key=key):
                selected.append(opt)
    else:
        for opt in options:
            key = f"{namespace}_{key_prefix}_{opt}"
            default_val = opt in default
            if container.checkbox(opt, value=default_val, key=key):
                selected.append(opt)

    return selected


def safe_session_get(key: str, default):
    """Safely get value from session state with default."""
    try:
        if key in st.session_state:
            return st.session_state[key]
    except (KeyError, TypeError):
        pass
    return default


# ============================================================================
# DISCLAIMER
# ============================================================================

def run_disclaimer():
    """Display data disclaimer and get user acceptance."""
    if "disclaimer_accepted" not in st.session_state:
        st.session_state.disclaimer_accepted = False

    st.set_page_config(page_title="Climate Metrics Viewer", layout="wide")

    if not st.session_state.disclaimer_accepted:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.title(f"{EMOJI['thermometer']} Climate Metrics Viewer")
            st.markdown("---")

            st.subheader("Data Disclaimer")
            st.markdown("""
By selecting "Accept", you acknowledge that the climate data provided through this viewer has not been independently verified and that you use any information displayed entirely at your own risk.  This tool is intended for general informational and educational purposes only and should not be used as the sole basis for any decision-making.  Climate projections and historical data may contain uncertainties, errors or limitations.  You are solely responsible for verifying any data before use and for determining whether this information is appropriate for your specific needs and requirements.  The creators and providers of this viewer make no representations or warranties of any kind, express or implied, regarding the accuracy, completeness, timeliness or reliability of the data presented.  Under no circumstances shall the creators or providers be liable for any decisions made or actions taken based on information from this viewer.
            """)

            st.markdown("---")

            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("Accept", key="accept_disclaimer", type="primary"):
                    st.session_state.disclaimer_accepted = True
                    st.rerun()

        st.stop()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def run_metrics_viewer():
    """Main application entry point."""
    run_disclaimer()

    # ========================================================================
    # INITIAL DATA LOADING (once per session)
    # ========================================================================

    if "data_loaded" not in st.session_state:
        loading_msg = st.empty()
        loading_msg.info("🌍 Loading climate data...")

        # Discover scenarios
        base_folder = resolve_folder()
        scenarios = discover_scenarios(base_folder)

        if not scenarios:
            st.error(f"{EMOJI['warning']} No scenarios found in: {base_folder}")
            st.stop()

        labels = [lbl for lbl, _, _ in scenarios]
        label_to_path = {lbl: path for lbl, _, path in scenarios}

        # Load metadata for UI
        metadata_tuples = [
            (lbl, label_to_path[lbl], os.path.getmtime(label_to_path[lbl]))
            for lbl in labels
        ]
        metadata = load_minimal_metadata(metadata_tuples)

        # Load and process all data
        processed = load_and_process_all_data(labels, label_to_path)

        # Store everything in session state
        st.session_state["labels"] = labels
        st.session_state["label_to_path"] = label_to_path
        st.session_state["metadata"] = metadata
        st.session_state["processed_data"] = processed
        st.session_state["data_loaded"] = True

        loading_msg.empty()

    # Retrieve from session state
    labels = st.session_state["labels"]
    label_to_path = st.session_state["label_to_path"]
    metadata = st.session_state["metadata"]
    processed = st.session_state["processed_data"]

    BASE_LABEL = "SSP1-26" if "SSP1-26" in labels else labels[0]

    # ========================================================================
    # TABS
    # ========================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        f"{EMOJI['thermometer']} Climate Metrics",
        f"{EMOJI['chart']} Dashboard",
        f"{EMOJI['globe']} Global Milestones",
        f"{EMOJI['book']} User Guide"
    ])

    # ========================================================================
    # TAB 1: METRICS VIEWER
    # ========================================================================
    with tab1:
        ns = "metrics"

        st.sidebar.title(f"{EMOJI['gear']} Filters")

        # Location selector
        with st.sidebar.expander(f"{EMOJI['pin']} Locations", expanded=False):
            all_locations = sorted(metadata["Location"].dropna().unique())
            default_locs = ["Ravenswood"] if "Ravenswood" in all_locations else all_locations[:1]
            loc_sel = multi_selector(st, "Locations", all_locations, default_locs,
                                     namespace=ns, key_prefix="loc")

        if not loc_sel:
            st.sidebar.warning(f"{EMOJI['warning']} Select at least one location")
            st.stop()

        # Scenario selector
        with st.sidebar.expander(f"{EMOJI['globe']} Scenarios", expanded=False):
            default_scen = [BASE_LABEL]
            if "historical" in [lbl.lower() for lbl in labels]:
                hist_label = next(lbl for lbl in labels if lbl.lower() == "historical")
                default_scen.append(hist_label)
            elif len(labels) > 1:
                default_scen.append(labels[1])

            scen_sel = multi_selector(st, "Scenarios", labels, default_scen,
                                      namespace=ns, key_prefix="scen")

        if not scen_sel:
            st.sidebar.warning(f"{EMOJI['warning']} Select at least one scenario")
            st.stop()

        # Display options
        with st.sidebar.expander(f"{EMOJI['display']} Display Options", expanded=False):
            mode = st.radio(
                "Display Mode",
                ["Values", "Baseline (start year)", f"Deltas vs {BASE_LABEL}"],
                index=0,
                key=f"{ns}_mode",
                help="Choose how to display values"
            )

            smooth = st.toggle(
                "Smooth Values",
                value=True,
                key=f"{ns}_smooth",
                help="Apply rolling average smoothing (IPCC AR6 standard is 20 years)"
            )
            if smooth:
                smooth_win = st.select_slider(
                    "Smoothing Window (years)",
                    options=[5, 10, 15, 20, 25],
                    value=20,
                    key=f"{ns}_smooth_win",
                    help="Number of years in rolling average"
                )
            else:
                smooth_win = 1

            show_global_thresholds = st.toggle(
                "Global Thresholds",
                value=True,
                key=f"{ns}_show_thresholds",
                help="Show 1.5°C, 2.0°C and 2.5°C global warming thresholds (temperature metrics only)"
            )

            show_horizon_shading = st.toggle(
                "Time Horizons",
                value=True,
                key=f"{ns}_show_horizons",
                help="Shade time horizons on charts (annual data only)"
            )

            bc_enabled = st.toggle(
                "Bias Correction",
                value=True,
                key=f"{ns}_bc_enabled",
                help="Display bias-corrected values (pre-computed in metrics files)"
            )

            align_to_agcd = st.toggle(
                "Align to Actual",
                value=True,
                key=f"{ns}_align_agcd",
                help="Align smoothed scenario traces to AGCD observations at 2014/2015 transition (requires smoothing)"
            )

            # Status indicators
            status_parts = []
            if bc_enabled:
                status_parts.append("Bias-corrected")
            else:
                status_parts.append("Raw model output")
            if smooth:
                status_parts.append(f"{smooth_win}yr smoothing")
            if smooth and align_to_agcd:
                status_parts.append("AGCD-aligned")
            elif not smooth and align_to_agcd:
                st.caption("⚠️ Alignment requires smoothing")

            st.caption(" | ".join(status_parts))

        # Year Range with Time Horizons
        with st.sidebar.expander(f"{EMOJI['calendar']} Year Range", expanded=False):
            yr_min = int(metadata["Year"].min())
            yr_max = int(metadata["Year"].max())

            default_start_year = max(DEFAULT_START_YEAR, yr_min) if yr_min <= DEFAULT_START_YEAR else yr_min
            default_end_year = min(DEFAULT_END_YEAR, yr_max)

            if f"{ns}_y0" not in st.session_state:
                st.session_state[f"{ns}_y0"] = default_start_year
            if f"{ns}_y1" not in st.session_state:
                st.session_state[f"{ns}_y1"] = default_end_year

            col_start, col_end = st.columns(2)
            with col_start:
                y0 = st.number_input(
                    "Start year",
                    min_value=yr_min,
                    max_value=yr_max,
                    value=st.session_state[f"{ns}_y0"],
                    step=1,
                    key=f"{ns}_year_start"
                )
            with col_end:
                y1 = st.number_input(
                    "End year",
                    min_value=yr_min,
                    max_value=yr_max,
                    value=st.session_state[f"{ns}_y1"],
                    step=1,
                    key=f"{ns}_year_end"
                )

            st.session_state[f"{ns}_y0"] = y0
            st.session_state[f"{ns}_y1"] = y1

            if y0 > y1:
                st.error(f"{EMOJI['warning']} Start year must be before end year")
                y0, y1 = y1, y0

            # Time Horizons with minimum of 2015 (SSP scenarios start 2015)
            HORIZON_MIN_YEAR = 2015
            horizon_min = max(y0, HORIZON_MIN_YEAR)

            st.markdown("**Time horizons:**")

            # Initialise slider session state if not present
            if f"{ns}_short_mid" not in st.session_state:
                st.session_state[f"{ns}_short_mid"] = (
                    max(TIME_HORIZONS.get('short_start_default', 2020), horizon_min),
                    max(TIME_HORIZONS.get('mid_start_default', 2036), horizon_min + 1)
                )
            if f"{ns}_mid_long" not in st.session_state:
                st.session_state[f"{ns}_mid_long"] = (
                    max(TIME_HORIZONS.get('mid_start_default', 2036), horizon_min + 1),
                    max(TIME_HORIZONS.get('long_start_default', 2039), horizon_min + 2)
                )
            if f"{ns}_long_end" not in st.session_state:
                st.session_state[f"{ns}_long_end"] = (
                    max(TIME_HORIZONS.get('long_start_default', 2039), horizon_min + 2),
                    min(TIME_HORIZONS.get('horizon_end_default', 2045), y1)
                )

            # Clamp all values to valid range
            def clamp_slider(val, min_v, max_v):
                start = max(min_v, min(val[0], max_v - 1))
                end = max(start + 1, min(val[1], max_v))
                return (start, end)

            st.session_state[f"{ns}_short_mid"] = clamp_slider(st.session_state[f"{ns}_short_mid"], horizon_min, y1)
            st.session_state[f"{ns}_mid_long"] = clamp_slider(st.session_state[f"{ns}_mid_long"], horizon_min, y1)
            st.session_state[f"{ns}_long_end"] = clamp_slider(st.session_state[f"{ns}_long_end"], horizon_min, y1)

            # Render sliders
            st.markdown('<p style="color: #3498DB; font-weight: bold; margin-bottom: 0;">Short-term</p>',
                        unsafe_allow_html=True)
            short_mid = st.slider(
                "short_slider_label",
                min_value=horizon_min,
                max_value=y1,
                step=1,
                key=f"{ns}_short_mid",
                label_visibility="collapsed"
            )

            st.markdown('<p style="color: #F39C12; font-weight: bold; margin-bottom: 0;">Medium-term</p>',
                        unsafe_allow_html=True)
            mid_long = st.slider(
                "mid_slider_label",
                min_value=horizon_min,
                max_value=y1,
                step=1,
                key=f"{ns}_mid_long",
                label_visibility="collapsed"
            )

            st.markdown('<p style="color: #E74C3C; font-weight: bold; margin-bottom: 0;">Long-term</p>',
                        unsafe_allow_html=True)
            long_end = st.slider(
                "long_slider_label",
                min_value=horizon_min,
                max_value=y1,
                step=1,
                key=f"{ns}_long_end",
                label_visibility="collapsed"
            )

            # Extract final values
            short_start = short_mid[0]
            mid_start = mid_long[0]
            long_start = long_end[0]
            horizon_end = long_end[1]

            st.caption(f"Planning horizon: {horizon_end}")

        # Metric type
        with st.sidebar.expander(f"{EMOJI['chart']} Metric Type", expanded=False):
            preferred_types = ["Temp", "Rain", "Wind", "Humidity"]
            all_types = list(metadata["Type"].unique())
            type_options = [t for t in preferred_types if t in all_types] + \
                           [t for t in sorted(all_types) if t not in preferred_types]
            default_type = "Temp" if "Temp" in type_options else type_options[0]

            type_sel = st.radio(
                "Type", type_options,
                index=type_options.index(default_type),
                horizontal=True,
                key=f"{ns}_type",
                help="Select the climate variable category to analyse"
            )

        # Metric names
        with st.sidebar.expander(f"{EMOJI['graph']} Metric Names", expanded=False):
            filtered_meta = metadata[
                (metadata["Location"].isin(loc_sel)) & (metadata["Type"] == type_sel)
                ]
            name_options = sorted(filtered_meta["Name"].dropna().unique())

            # Apply priority ordering if available
            if type_sel in METRIC_PRIORITIES:
                priority = METRIC_PRIORITIES[type_sel]
                name_options = sorted(
                    name_options,
                    key=lambda n: (priority.index(n) if n in priority else 99, n)
                )
                default_names = [priority[0]] if priority[0] in name_options else name_options[:1]
            else:
                default_names = name_options[:1]

            name_sel = multi_selector(
                st, "Metrics", name_options,
                default=default_names,
                columns=1, namespace=ns, key_prefix="name"
            )

        if not name_sel:
            st.sidebar.warning(f"{EMOJI['warning']} Select at least one metric")
            st.stop()

        # Seasons
        with st.sidebar.expander(f"{EMOJI['calendar']} Seasons", expanded=False):
            available_seasons = [s for s in SEASONS.get('all', ["Annual", "DJF", "MAM", "JJA", "SON"])
                                 if s in metadata["Season"].unique()]
            default_seasons = ["Annual"] if "Annual" in available_seasons else available_seasons[:1]

            season_sel = multi_selector(
                st, "Seasons", available_seasons,
                default=default_seasons, columns=1,
                namespace=ns, key_prefix="season"
            )

        if not season_sel:
            st.sidebar.warning(f"{EMOJI['warning']} Select at least one season")
            st.stop()

        # ================================================================
        # PROCESS DATA based on toggles (on-demand with caching)
        # ================================================================
        # Cache key includes all processing options
        # Year range is NOT in cache key - filtering happens after

        cache_key = f"processed_{bc_enabled}_{smooth}_{smooth_win}_{align_to_agcd}_{tuple(sorted(loc_sel))}"

        if cache_key not in st.session_state:
            # Start with raw or BC data based on toggle
            if bc_enabled:
                df_work = processed["df_bc"].copy()
            else:
                df_work = processed["df_raw"].copy()

            # Filter to selected locations (full year range for smoothing)
            df_work = df_work[df_work["Location"].isin(loc_sel)].copy()

            # Apply smoothing if enabled
            if smooth and smooth_win > 1:
                df_work = apply_smoothing(df_work, smooth_win)

                # Apply alignment if enabled (only makes sense with smoothing)
                if align_to_agcd:
                    df_work = align_smoothed_to_agcd(df_work, alignment_year=2014)

            # Cache the processed data
            st.session_state[cache_key] = df_work

        df_all_full = st.session_state[cache_key]

        # Filter to selected scenarios and year range (instant - no processing)
        df_all = df_all_full[
            (df_all_full["Scenario"].isin(scen_sel)) &
            (df_all_full["Year"] >= y0) &
            (df_all_full["Year"] <= y1)
            ].copy()

        # Get base scenario data
        base_df = df_all[df_all["Scenario"] == BASE_LABEL].copy()

        # Get baselines
        preindustrial_baseline = processed["baselines"]

        # Render metrics tab (display only - data already processed)
        render_metrics_tab(
            st, ns, loc_sel, scen_sel, type_sel, name_sel, season_sel, y0, y1,
            mode, smooth, smooth_win, show_global_thresholds, show_horizon_shading,
            short_start, mid_start, long_start, horizon_end,
            df_all, base_df, preindustrial_baseline, BASE_LABEL
        )

    # ========================================================================
    # TAB 2: DASHBOARD
    # ========================================================================
    with tab2:
        # Dashboard uses same cache as metrics tab (already processed)
        cache_key = f"processed_{bc_enabled}_{smooth}_{smooth_win}_{align_to_agcd}_{tuple(sorted(loc_sel))}"

        if cache_key in st.session_state:
            df_dash_full = st.session_state[cache_key]
        else:
            # Fallback - should not happen if metrics tab ran first
            df_dash_full = processed["df_bc"] if bc_enabled else processed["df_raw"]
            df_dash_full = df_dash_full[df_dash_full["Location"].isin(loc_sel)]

        # Filter to scenarios and year range (instant)
        df_dash = df_dash_full[
            (df_dash_full["Scenario"].isin(scen_sel)) &
            (df_dash_full["Year"] >= y0) &
            (df_dash_full["Year"] <= y1)
            ]

        render_dashboard_tab(
            st=st,
            scen_sel=scen_sel,
            loc_sel=loc_sel,
            df_all=df_dash,
            labels=labels,
            label_to_path=label_to_path,
            short_start=short_start,
            mid_start=mid_start,
            long_start=long_start,
            horizon_end=horizon_end
        )

    # ========================================================================
    # TAB 3: GLOBAL MILESTONES
    # ========================================================================
    with tab3:
        # Global tab uses same cache as metrics (already has full year range)
        cache_key = f"processed_{bc_enabled}_{smooth}_{smooth_win}_{align_to_agcd}_{tuple(sorted(loc_sel))}"

        if cache_key in st.session_state:
            df_global_full = st.session_state[cache_key]
        else:
            # Fallback - should not happen if metrics tab ran first
            df_global_full = processed["df_bc"] if bc_enabled else processed["df_raw"]
            df_global_full = df_global_full[df_global_full["Location"].isin(loc_sel)]

        # Filter to scenarios (year filtering done by render_global_tab)
        df_global = df_global_full[df_global_full["Scenario"].isin(scen_sel)]

        # Calculate crossing years based on current smoothing settings
        crossing_cache_key = f"crossings_{bc_enabled}_{smooth}_{smooth_win}_{align_to_agcd}_{tuple(sorted(loc_sel))}"

        if crossing_cache_key not in st.session_state:
            crossing_years = {}
            baselines = processed["baselines"]
            for loc in loc_sel:
                if loc in baselines:
                    crossing_years[loc] = calculate_threshold_crossings(
                        df_global_full, loc, baselines[loc]
                    )
            st.session_state[crossing_cache_key] = crossing_years

        crossing_years = st.session_state[crossing_cache_key]
        baselines = processed["baselines"]

        render_global_tab(
            st=st,
            scen_sel=scen_sel,
            loc_sel=loc_sel,
            df_all=df_global,
            crossing_years=crossing_years,
            baselines=baselines,
            y0=y0,
            y1=y1,
            short_start=short_start,
            mid_start=mid_start,
            long_start=long_start,
            horizon_end=horizon_end,
            smooth_win=smooth_win,
            show_horizon_shading=show_horizon_shading
        )

    # ========================================================================
    # TAB 4: USER GUIDE
    # ========================================================================
    with tab4:
        st.title(f"{EMOJI['book']} Scientific User Guide")
        st.markdown("*Complete documentation for Climate Metrics Viewer*")
        st.markdown("---")
        render_user_guide()


if __name__ == "__main__":
    if MODE.lower() == "metrics":
        run_metrics_viewer()
    else:
        sys.exit('Set MODE = "metrics"')