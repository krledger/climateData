#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Metrics Viewer
======================
Last Updated: 2025-11-26 21:25 AEST
Previous Update: 2025-11-26 15:50 AEST

Streamlined multi-tab application for browsing climate metrics.
NO CACHING - Fresh data loading each time to avoid stale data bugs.

Bias correction and scenario alignment are pre-computed in the metrics files
(Value_BC column).  The viewer just selects which column to display based
on the BC toggle.

Time Horizons: Slider minimum is 2015 (SSP scenarios start 2015).
Slider cascading: changing end of one slider updates start of next slider.

Global tab passes smoothing parameters (smooth, smooth_win) to ensure
consistency with Metrics tab display.

Consolidated structure with minimal modules.
"""

import os
import sys
import json
import re
from typing import Literal

try:
    import streamlit as st
    import pandas as pd
except ImportError:
    sys.exit("Missing dependencies.  Run: pip install streamlit pandas pyarrow altair")

from climate_viewer_constants import MODE, TIME_HORIZONS, METRIC_PRIORITIES, SEASONS, BASELINE_PERIOD, REFERENCE_YEAR, \
    EMOJI
from climate_viewer_data_operations import (
    resolve_folder,
    discover_scenarios,
    ensure_schema,
    load_metrics_file,
    load_minimal_metadata,
    calculate_preindustrial_baselines_by_location,
    parse_data_type,
    dedupe_preserve_order,
    slugify,
)
from climate_viewer_tab_metrics import render_metrics_tab
from climate_viewer_tab_dashboard import render_dashboard_tab
from climate_viewer_tab_global import render_global_tab
from climate_viewer_tab_guide import render_user_guide


# ============================================================================
# UI HELPER FUNCTIONS (formerly climate_viewer_ui_components.py)
# ============================================================================

def multi_selector(
        container,
        label: str,
        options,
        default=None,
        widget: Literal["checkbox", "toggle"] = "checkbox",
        columns: int = 1,
        key_prefix: str = "sel",
        namespace: str = None
):
    """Multi-select widget with checkbox or toggle options."""
    opts = dedupe_preserve_order(options or [])
    default = set(default or [])
    selected = []

    container.markdown(f"**{label}**")
    cols = container.columns(columns)
    widget_fn = container.checkbox if widget == "checkbox" else container.toggle

    import hashlib
    opts_hash = hashlib.md5(str(sorted(opts)).encode()).hexdigest()[:8]

    for i, opt in enumerate(opts):
        key = f"{namespace + '_' if namespace else ''}{key_prefix}_{i}_{opts_hash}_{slugify(opt)}"
        with cols[i % columns]:
            if widget_fn(str(opt), value=(opt in default), key=key):
                selected.append(opt)
    return selected


# ============================================================================
# HELP TEXT FUNCTIONS
# ============================================================================

def load_help_text():
    """Load help text from external JSON file."""
    help_file = os.path.join(os.path.dirname(__file__), "climate_viewer_help_text.json")
    if os.path.exists(help_file):
        with open(help_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}


HELP_TEXT = load_help_text()


def get_help(path: str, default: str = "") -> str:
    """
    Get help text from nested JSON structure using dot notation.
    Example: get_help("sidebar.locations.help")
    """
    try:
        keys = path.split(".")
        value = HELP_TEXT
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


# ============================================================================
# DISCLAIMER
# ============================================================================

def run_disclaimer(st):
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
    run_disclaimer(st)

    # Load data WITHOUT CACHING
    base_folder = resolve_folder()
    scenarios = discover_scenarios(base_folder)

    if not scenarios:
        st.error(f"{EMOJI['warning']} No scenarios found in: {base_folder}")
        st.stop()

    labels = [lbl for lbl, _, _ in scenarios]
    label_to_path = {lbl: path for lbl, _, path in scenarios}
    BASE_LABEL = "SSP1-26" if "SSP1-26" in labels else labels[0]

    # Load minimal metadata for UI setup
    metadata_tuples = [
        (lbl, label_to_path[lbl], os.path.getmtime(label_to_path[lbl]))
        for lbl in labels
    ]
    metadata = load_minimal_metadata(metadata_tuples)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [f"{EMOJI['thermometer']} Climate Metrics", f"{EMOJI['chart']} Dashboard",
         f"{EMOJI['globe']} Global Milestones", f"{EMOJI['book']} User Guide"])

    # ========================================================================
    # TAB 1: METRICS VIEWER
    # ========================================================================
    with tab1:
        ns = "metrics"

        st.sidebar.title(f"{EMOJI['gear']} Filters")

        with st.sidebar.expander(f"{EMOJI['pin']} Locations", expanded=False):
            all_locations = sorted(metadata["Location"].dropna().unique())
            default_locs = ["Ravenswood"] if "Ravenswood" in all_locations else all_locations[:1]

            loc_sel = multi_selector(
                st, "Select locations", all_locations,
                default=default_locs, columns=1, namespace=ns, key_prefix="loc"
            )

        if not loc_sel:
            st.sidebar.warning(f"{EMOJI['warning']} Select at least one location")
            st.stop()

        with st.sidebar.expander(f"{EMOJI['globe']} Scenarios", expanded=False):
            default_scen = [BASE_LABEL]
            if "historical" in [lbl.lower() for lbl in labels]:
                hist_label = next(lbl for lbl in labels if lbl.lower() == "historical")
                default_scen.append(hist_label)
            elif len(labels) > 1:
                default_scen.append(labels[1])

            scen_sel = multi_selector(
                st, "Select scenarios", labels,
                default=default_scen, columns=1, namespace=ns, key_prefix="scen"
            )

        if not scen_sel:
            st.sidebar.warning(f"{EMOJI['warning']} Select at least one scenario")
            st.stop()

        with st.sidebar.expander(f"{EMOJI['display']} Display Options", expanded=False):
            mode = st.radio(
                "Display Mode",
                ["Values", "Baseline (start year)", f"Deltas vs {BASE_LABEL}"],
                index=0,
                key=f"{ns}_mode",
                help=get_help("sidebar.display_options.display_mode.help", "Choose how to display values")
            )

            smooth = st.toggle(
                "Smooth Values",
                value=False,
                key=f"{ns}_smooth",
                help=get_help("sidebar.display_options.smooth_values", "Apply rolling average smoothing")
            )
            if smooth:
                smooth_win = st.slider(
                    "Smoothing Window (years)",
                    3, 21, step=2, value=9,
                    key=f"{ns}_smooth_win",
                    help=get_help("sidebar.display_options.smoothing_window", "Number of years in rolling average")
                )
            else:
                smooth_win = 1

            show_15c_shading = st.toggle(
                "Show 1.5C Target",
                value=False,
                key=f"{ns}_show_15c",
                help="Show red line at 1.5 deg C global warming target (temperature metrics only)"
            )

            show_horizon_shading = st.toggle(
                "Show Time Horizons",
                value=False,
                key=f"{ns}_show_horizons",
                help="Shade time horizons on charts (annual data only)"
            )

        with st.sidebar.expander(f"{EMOJI['calendar']} Year Range", expanded=False):
            yr_min, yr_max = int(metadata["Year"].min()), int(metadata["Year"].max())

            # Default start year is REFERENCE_YEAR (or yr_min if data starts after that)
            default_start_year = max(REFERENCE_YEAR, yr_min) if yr_min <= REFERENCE_YEAR else yr_min

            if f"{ns}_y0" not in st.session_state:
                st.session_state[f"{ns}_y0"] = default_start_year
            if f"{ns}_y1" not in st.session_state:
                st.session_state[f"{ns}_y1"] = yr_max

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
                    max(TIME_HORIZONS['short_start_default'], horizon_min),
                    max(TIME_HORIZONS['mid_start_default'], horizon_min + 1)
                )
            if f"{ns}_mid_long" not in st.session_state:
                st.session_state[f"{ns}_mid_long"] = (
                    max(TIME_HORIZONS['mid_start_default'], horizon_min + 1),
                    max(TIME_HORIZONS['long_start_default'], horizon_min + 2)
                )
            if f"{ns}_long_end" not in st.session_state:
                st.session_state[f"{ns}_long_end"] = (
                    max(TIME_HORIZONS['long_start_default'], horizon_min + 2),
                    min(TIME_HORIZONS['horizon_end_default'], y1)
                )

            # Read current values from session state
            short_mid_val = st.session_state[f"{ns}_short_mid"]
            mid_long_val = st.session_state[f"{ns}_mid_long"]
            long_end_val = st.session_state[f"{ns}_long_end"]

            # CASCADE: If short_mid END changed, update mid_long START
            if short_mid_val[1] != mid_long_val[0]:
                new_mid_start = short_mid_val[1]
                new_mid_end = max(mid_long_val[1], new_mid_start + 1)
                st.session_state[f"{ns}_mid_long"] = (new_mid_start, new_mid_end)
                mid_long_val = st.session_state[f"{ns}_mid_long"]

            # CASCADE: If mid_long END changed, update long_end START
            if mid_long_val[1] != long_end_val[0]:
                new_long_start = mid_long_val[1]
                new_long_end = max(long_end_val[1], new_long_start + 1)
                st.session_state[f"{ns}_long_end"] = (new_long_start, new_long_end)
                long_end_val = st.session_state[f"{ns}_long_end"]

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

            # Bias Correction Toggle
            st.markdown("---")
            bc_enabled = st.toggle(
                "Bias Correction",
                value=False,
                key=f"{ns}_bc_enabled",
                help="Display bias-corrected values (pre-computed in metrics files).  "
                     "Corrections adjust for systematic model biases based on AGCD observations (1971-2014).  "
                     "SSP scenarios are aligned to AGCD at the 2014-2015 transition."
            )

            if bc_enabled:
                st.caption("Displaying bias-corrected values (Value_BC column)")

        with st.sidebar.expander(f"{EMOJI['graph']} Metric Names", expanded=False):
            filtered_meta = metadata[
                (metadata["Location"].isin(loc_sel)) &
                (metadata["Type"] == type_sel)
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
                st, "Select metrics", name_options,
                default=default_names, widget="toggle",
                columns=1, namespace=ns, key_prefix="name"
            )

        if not name_sel:
            st.sidebar.warning(f"{EMOJI['warning']} Select at least one metric")
            st.stop()

        with st.sidebar.expander(f"{EMOJI['calendar']} Seasons", expanded=False):
            available_seasons = [s for s in SEASONS['all'] if s in metadata["Season"].unique()]
            default_seasons = ["Annual"] if "Annual" in available_seasons else available_seasons

            season_sel = multi_selector(
                st, "Select seasons", available_seasons,
                default=default_seasons, columns=1,
                namespace=ns, key_prefix="season"
            )

        if not season_sel:
            st.sidebar.warning(f"{EMOJI['warning']} Select at least one season")
            st.stop()

        # Load data WITHOUT CACHING
        with st.spinner("Loading data..."):
            dfs = []

            for label in scen_sel:
                path = label_to_path[label]
                df = load_metrics_file(path, use_bc=bc_enabled)
                df["Scenario"] = label
                dfs.append(df)

            df_all = pd.concat(dfs, ignore_index=True)

            # Load base scenario for delta calculations
            base_path = label_to_path[BASE_LABEL]
            base_df = load_metrics_file(base_path, use_bc=bc_enabled)
            base_df["Scenario"] = BASE_LABEL

            preindustrial_baseline = calculate_preindustrial_baselines_by_location(
                df_all, BASELINE_PERIOD, loc_sel
            )

        # Create wrapper function for tabs that need to load additional data
        def load_metrics_func_with_bc(path):
            df = load_metrics_file(path, use_bc=bc_enabled)
            # Infer scenario from path
            scenario_name = os.path.basename(os.path.dirname(path))
            df["Scenario"] = scenario_name
            return df

        render_metrics_tab(
            st, ns, loc_sel, scen_sel, type_sel, name_sel, season_sel, y0, y1,
            mode, smooth, smooth_win, show_15c_shading, show_horizon_shading,
            short_start, mid_start, long_start, horizon_end,
            df_all, base_df, preindustrial_baseline, BASE_LABEL
        )

    # ========================================================================
    # TAB 2: DASHBOARD
    # ========================================================================
    with tab2:
        render_dashboard_tab(
            st, scen_sel, loc_sel, None, df_all, labels, label_to_path,
            load_metrics_func_with_bc, y0, mid_start, long_start, horizon_end
        )

        # ========================================================================
        # TAB 3: GLOBAL WARMING THRESHOLDS
        # ========================================================================
        with tab3:
            render_global_tab(
                st, scen_sel, loc_sel, df_all, labels, label_to_path, load_metrics_func_with_bc,
                y0, y1, short_start, mid_start, long_start, horizon_end,
                smooth=smooth, smooth_win=smooth_win
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