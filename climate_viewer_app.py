#!/usr/bin/env python3
"""
Climate Metrics Viewer (Refactored)
Streamlined multi-tab application for browsing climate metrics.
NO CACHING - Fresh data loading each time to avoid stale data bugs.
"""

import os
import sys

try:
    import streamlit as st
    import pandas as pd
except ImportError:
    sys.exit("Missing dependencies. Run: pip install streamlit pandas pyarrow altair")

from climate_viewer_config import MODE
from climate_viewer_helpers import resolve_folder
from climate_viewer_data_operations import (
    discover_scenarios,
    ensure_schema,
    load_metrics_file,
    load_minimal_metadata,
    calculate_preindustrial_baseline,
)
from climate_viewer_helpers import parse_data_type
from climate_viewer_ui_components import multi_selector
from climate_viewer_tab_metrics import render_metrics_tab
from climate_viewer_tab_dashboard import render_dashboard_tab
from climate_viewer_tab_global import render_global_tab
from climate_viewer_constants import (
    TIME_HORIZONS,
    METRIC_PRIORITIES,
    SEASONS,
)


def run_disclaimer(st):
    """Display data disclaimer and get user acceptance."""
    if "disclaimer_accepted" not in st.session_state:
        st.session_state.disclaimer_accepted = False
    
    st.set_page_config(page_title="Climate Metrics Viewer", layout="wide")
    
    if not st.session_state.disclaimer_accepted:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🌡️ Climate Metrics Viewer")
            st.markdown("---")
            
            st.subheader("Data Disclaimer")
            st.markdown("""
By selecting "Accept," you acknowledge that the climate data provided through this viewer has not been independently verified and that you use any information displayed entirely at your own risk. This tool is intended for general informational and educational purposes only and should not be used as the sole basis for any decision-making. Climate projections and historical data may contain uncertainties, errors or limitations. You are solely responsible for verifying any data before use and for determining whether this information is appropriate for your specific needs and requirements. The creators and providers of this viewer make no representations or warranties of any kind, express or implied, regarding the accuracy, completeness, timeliness or reliability of the data presented. Under no circumstances shall the creators or providers be liable for any decisions made or actions taken based on information from this viewer.
            """)
            
            st.markdown("---")
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("Accept", key="accept_disclaimer", type="primary", width="stretch"):
                    st.session_state.disclaimer_accepted = True
                    st.rerun()
        
        st.stop()


def run_metrics_viewer():
    """Main application entry point."""
    run_disclaimer(st)
    
    # Load data WITHOUT CACHING
    base_folder = resolve_folder()
    scenarios = discover_scenarios(base_folder)
    
    if not scenarios:
        st.error(f"Ã¢ÂÅ’ No scenarios found in: {base_folder}")
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
    tab1, tab2, tab3 = st.tabs(["🌡️ Climate Metrics", "📊 Dashboard", "🌍 Global 1.5°C Impact"])
    
    # ============================= TAB 1: METRICS VIEWER =====================
    with tab1:
        ns = "metrics"
        
        st.sidebar.title("⚙️ Filters")
        
        with st.sidebar.expander("📍 Locations", expanded=True):
            all_locations = sorted(metadata["Location"].dropna().unique())
            default_locs = ["Ravenswood"] if "Ravenswood" in all_locations else all_locations[:1]
            
            loc_sel = multi_selector(
                st, "Select locations", all_locations,
                default=default_locs, columns=1, namespace=ns, key_prefix="loc"
            )
        
        if not loc_sel:
            st.sidebar.warning("⚠️ Select at least one location")
            st.stop()
        
        with st.sidebar.expander("🌍 Scenarios", expanded=True):
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
            st.sidebar.warning("⚠️ Select at least one scenario")
            st.stop()
        
        with st.sidebar.expander("🖨 Display Options", expanded=True):
            mode = st.radio(
                "Display Mode",
                ["Values", "Baseline (start year)", f"Deltas vs {BASE_LABEL}"],
                index=0, key=f"{ns}_mode"
            )
            
            smooth = st.toggle("Smooth Values", value=False, key=f"{ns}_smooth")
            if smooth:
                smooth_win = st.slider(
                    "Smoothing Window (years)",
                    3, 21, step=2, value=9, key=f"{ns}_smooth_win"
                )
            else:
                smooth_win = 1
            
            table_interval = st.radio(
                "Table Interval (years)",
                [1, 2, 5, 10],
                index=1, horizontal=True, key=f"{ns}_table_int"
            )
            
            show_15c_shading = st.toggle(
                "1.5°C above preindustrial",
                value=False,
                key=f"{ns}_show_15c"
            )
            show_horizon_shading = st.toggle(
                "Time horizon shading",
                value=False,
                key=f"{ns}_show_horizons"
            )
        
        with st.sidebar.expander("📅 Year Range", expanded=True):
            yr_min, yr_max = int(metadata["Year"].min()), int(metadata["Year"].max())
            
            non_historical_scenarios = [s for s in labels if "historical" not in s.lower()]
            if non_historical_scenarios:
                non_hist_data = metadata[metadata["Scenario"].isin(non_historical_scenarios)]
                default_start_year = int(non_hist_data["Year"].min()) if not non_hist_data.empty else yr_min
            else:
                default_start_year = yr_min
            
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
                st.error("⚠️ Start year must be before end year")
                y0, y1 = y1, y0
            
            st.markdown("**Time horizons:**")
            
            short_mid = st.slider(
                "Short",
                min_value=y0,
                max_value=y1,
                value=(min(max(TIME_HORIZONS['short_start_default'], y0), y1),
                       min(max(TIME_HORIZONS['mid_start_default'], y0), y1)),
                step=1,
                key=f"{ns}_short_mid",
                help="Short-term period (blue shading on chart)"
            )
            short_start = short_mid[0]
            mid_start = short_mid[1]
            
            mid_long = st.slider(
                "Medium",
                min_value=y0,
                max_value=y1,
                value=(mid_start, min(max(TIME_HORIZONS['long_start_default'], y0), y1)),
                step=1,
                key=f"{ns}_mid_long",
                help="Medium-term period (orange shading on chart)"
            )
            mid_start = mid_long[0]
            long_start = mid_long[1]
            
            long_end = st.slider(
                "Long",
                min_value=y0,
                max_value=y1,
                value=(long_start, min(TIME_HORIZONS['horizon_end_default'], y1)),
                step=1,
                key=f"{ns}_long_end",
                help="Long-term period (red shading on chart)"
            )
            long_start = long_end[0]
            horizon_end = long_end[1]
        
        with st.sidebar.expander("📊 Metric Type", expanded=False):
            preferred_types = ["Temp", "Rain", "Wind", "Humidity"]
            all_types = list(metadata["Type"].unique())
            type_options = [t for t in preferred_types if t in all_types] + \
                           [t for t in sorted(all_types) if t not in preferred_types]
            default_type = "Temp" if "Temp" in type_options else type_options[0]
            
            type_sel = st.radio(
                "Type", type_options,
                index=type_options.index(default_type),
                horizontal=True, key=f"{ns}_type"
            )
        
        with st.sidebar.expander("📈 Metric Names", expanded=False):
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
            st.sidebar.warning("⚠️ Select at least one metric")
            st.stop()
        
        with st.sidebar.expander("📅 Seasons", expanded=False):
            available_seasons = [s for s in SEASONS['all'] if s in metadata["Season"].unique()]
            default_seasons = ["Annual"] if "Annual" in available_seasons else available_seasons
            
            season_sel = multi_selector(
                st, "Select seasons", available_seasons,
                default=default_seasons, columns=1,
                namespace=ns, key_prefix="season"
            )
        
        if not season_sel:
            st.sidebar.warning("⚠️ Select at least one season")
            st.stop()
        
        with st.sidebar.expander("ℹ️ Help"):
            st.markdown("""
            **Smoothing**: Applies a rolling average to reduce noise in the data.
            
            **Baseline**: Shows change from the first year in the selected range.
            
            **Deltas**: Shows difference from the reference scenario (SSP1-26).
            
            **Table Interval**: Controls row spacing in the values table.
            
            **Time Horizon Shading**: Shades chart regions matching sidebar colours (°Å¸Å¸Â¦ blue, °Å¸Å¸Â§ orange, °Å¸Å¸Â¥ red) for annual data only.
            
            **1.5°C above preindustrial**: Shows horizontal red dashed line indicating regional temperature when **global** warming reaches 1.5°C above 1850-1900 baseline (~2.1°C regional warming due to land amplification). Hover over line for details.
            
            **Historical Gap**: Visual gaps between historical and future scenarios reflect actual temporal discontinuity in the data source.
            """)
        
        # Load data WITHOUT CACHING
        with st.spinner("Loading data..."):
            dfs = []
            for label in scen_sel:
                path = label_to_path[label]
                df = load_metrics_file(path)
                df["Scenario"] = label
                dfs.append(df)
            
            df_all = pd.concat(dfs, ignore_index=True)
            
            base_path = label_to_path[BASE_LABEL]
            base_df = load_metrics_file(base_path)
            base_df["Scenario"] = BASE_LABEL
            
            preindustrial_baseline = calculate_preindustrial_baseline(df_all, loc_sel)
        
        render_metrics_tab(
            st, ns, loc_sel, scen_sel, type_sel, name_sel, season_sel, y0, y1,
            mode, smooth, smooth_win, table_interval, show_15c_shading, show_horizon_shading,
            short_start, mid_start, long_start, horizon_end,
            df_all, base_df, preindustrial_baseline, BASE_LABEL
        )
    
    with tab2:
        render_dashboard_tab(
            st, scen_sel, loc_sel, None, df_all, labels, label_to_path,
            load_metrics_file, y0, mid_start, long_start, horizon_end
        )
    
    with tab3:
        render_global_tab(
            st, scen_sel, loc_sel, df_all, labels, label_to_path, load_metrics_file
        )


if __name__ == "__main__":
    if MODE.lower() == "metrics":
        run_metrics_viewer()
    else:
        sys.exit('Set MODE = "metrics"')
