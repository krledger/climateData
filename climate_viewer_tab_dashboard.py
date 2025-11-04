"""
Climate Viewer Dashboard Tab
Displays climate metrics across time horizons with pre-industrial baseline comparison.
"""

import os
import numpy as np

from climate_viewer_config import FOLIUM_AVAILABLE, BASE_DIR
from climate_viewer_constants import (
    BASELINE_PERIOD,
    REFERENCE_YEAR,
    DASHBOARD_METRICS,
    should_use_inverse_delta,
)
from climate_viewer_utils import (
    find_metric_name,
    calculate_metric_change,
    calculate_preindustrial_baselines_by_metric,
    format_metric_card_html,
)
from climate_viewer_maps import create_region_map

if FOLIUM_AVAILABLE:
    from streamlit_folium import st_folium


def render_dashboard_tab(
    st,
    scen_sel,
    loc_sel,
    dashboard_location,
    df_all,
    labels,
    label_to_path,
    load_metrics_func,
    baseline_year,
    mid_start,
    long_start,
    horizon_end
):
    """
    Render the climate dashboard tab.
    
    Args:
        st: Streamlit module
        scen_sel: Selected scenarios
        loc_sel: Selected locations
        dashboard_location: Selected dashboard location
        df_all: All loaded data
        labels: All scenario labels
        label_to_path: Mapping of label to file path
        load_metrics_func: Function to load metrics (NOT CACHED)
        baseline_year: Baseline year for comparisons
        mid_start: Mid-term horizon start year
        long_start: Long-term horizon start year
        horizon_end: End of long-term horizon
    """
    st.title("📊 Climate Change Dashboard")
    
    dashboard_scenarios = [s for s in scen_sel if "historical" not in s.lower()]
    
    if not dashboard_scenarios:
        st.warning("⚠️ Please select at least one future scenario (excluding historical) to view dashboard.")
        st.stop()
    
    dashboard_location = st.selectbox(
        "Select location for dashboard",
        loc_sel,
        key="dashboard_location"
    )
    
    # Show region map if available
    kml_path = os.path.join(BASE_DIR, "australia_grid_coverage.kml")
    ravenswood_kml_path = os.path.join(BASE_DIR, "ravenswood_grid_cell.kml")
    
    # Diagnostic information (remove after debugging)
    with st.expander("🔍 Map Diagnostic Info", expanded=False):
        st.write(f"**FOLIUM_AVAILABLE:** {FOLIUM_AVAILABLE}")
        st.write(f"**BASE_DIR:** {BASE_DIR}")
        st.write(f"**KML path:** {kml_path}")
        st.write(f"**KML exists:** {os.path.exists(kml_path)}")
        st.write(f"**Ravenswood path:** {ravenswood_kml_path}")
        st.write(f"**Ravenswood exists:** {os.path.exists(ravenswood_kml_path)}")
        
        # List files in BASE_DIR
        try:
            files = os.listdir(BASE_DIR)
            kml_files = [f for f in files if f.endswith('.kml')]
            st.write(f"**KML files in BASE_DIR:** {kml_files if kml_files else 'None found'}")
        except Exception as e:
            st.write(f"**Error listing files:** {str(e)}")
    
    if FOLIUM_AVAILABLE and os.path.exists(kml_path):
        with st.expander("🗺️ Region Map", expanded=False):
            st.markdown("**Grid coverage area for climate metrics**")
            region_map, status_msg, placemark_count = create_region_map(
                kml_path,
                ravenswood_kml_path if os.path.exists(ravenswood_kml_path) else None
            )
            if region_map:
                if placemark_count > 0:
                    st.caption(f"✓ Displaying {placemark_count} grid cells from KML file")
                elif "Partial" in status_msg:
                    st.warning(f"⚠️ {status_msg}")
                st_folium(region_map, width=1100, height=600)
            else:
                st.info(f"Map unavailable: {status_msg}")
    elif not FOLIUM_AVAILABLE:
        st.info("📦 Install folium and streamlit-folium to view region map: `pip install folium streamlit-folium`")
    
    # Prepare dashboard data
    with st.spinner("Calculating dashboard metrics..."):
        dashboard_data = df_all[
            (df_all["Location"] == dashboard_location) &
            (df_all["Season"] == "Annual") &
            (df_all["Scenario"].isin(dashboard_scenarios))
        ].copy()
        
        if dashboard_data.empty:
            st.error("❌ No data available for selected location and scenarios.")
            st.stop()
        
        # Calculate pre-industrial baselines
        preindustrial_baselines = {}
        
        historical_scenario = None
        for label in labels:
            if "historical" in label.lower():
                historical_scenario = label
                break
        
        if historical_scenario:
            historical_path = label_to_path[historical_scenario]
            historical_df = load_metrics_func(historical_path)
            
            preindustrial_baselines = calculate_preindustrial_baselines_by_metric(
                historical_df, BASELINE_PERIOD, dashboard_location
            )
            
            if preindustrial_baselines:
                baseline_source = f"Historical scenario ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]})"
            else:
                baseline_source = "Historical scenario (no 1850-1900 data)"
        else:
            earliest_year = int(df_all["Year"].min())
            earliest_data = df_all[
                (df_all["Location"] == dashboard_location) &
                (df_all["Season"] == "Annual") &
                (df_all["Year"] == earliest_year)
            ]
            
            for metric_type in earliest_data["Type"].unique():
                for metric_name in earliest_data[earliest_data["Type"] == metric_type]["Name"].unique():
                    key = (metric_type, metric_name)
                    subset = earliest_data[
                        (earliest_data["Type"] == metric_type) &
                        (earliest_data["Name"] == metric_name)
                    ]
                    if not subset.empty:
                        preindustrial_baselines[key] = subset["Value"].mean()
            
            baseline_source = f"Earliest available ({earliest_year})"
    
    # Display metrics for each scenario
    for scenario in dashboard_scenarios:
        st.markdown("---")
        st.subheader(f"🌐 {scenario}")
        
        st.markdown("### 📅 Time Horizons")
        col_ref, col_short, col_mid, col_long = st.columns(4)
        
        with col_ref:
            st.markdown(f"<h2 style='text-align: center; color: #808080;'>{REFERENCE_YEAR}</h2>",
                        unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: bold; margin-top: -10px;'>REFERENCE</p>",
                        unsafe_allow_html=True)
        with col_short:
            st.markdown(f"<h2 style='text-align: center; color: #3498DB;'>{mid_start}</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: bold; margin-top: -10px;'>SHORT TERM</p>",
                        unsafe_allow_html=True)
        with col_mid:
            st.markdown(f"<h2 style='text-align: center; color: #F39C12;'>{long_start}</h2>",
                        unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: bold; margin-top: -10px;'>MEDIUM TERM</p>",
                        unsafe_allow_html=True)
        with col_long:
            st.markdown(f"<h2 style='text-align: center; color: #E74C3C;'>{horizon_end}</h2>",
                        unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: bold; margin-top: -10px;'>LONG TERM</p>",
                        unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
        
        # Display metrics by category
        for category, metrics in DASHBOARD_METRICS.items():
            st.markdown(f"<h3 style='margin: 12px 0 8px 0;'>{category}</h3>", unsafe_allow_html=True)
            
            for metric_type, metric_name, unit, icon, key in metrics:
                actual_metric_name = find_metric_name(dashboard_data, metric_type, metric_name)
                display_name = actual_metric_name
                
                col_ref, col_short, col_mid, col_long = st.columns(4)
                
                # Calculate changes for each time horizon
                change_ref, pi_change_ref, value_ref = calculate_metric_change(
                    dashboard_data, metric_type, actual_metric_name, scenario,
                    baseline_year, REFERENCE_YEAR, dashboard_location, preindustrial_baselines
                )
                change_short, pi_change_short, value_short = calculate_metric_change(
                    dashboard_data, metric_type, actual_metric_name, scenario,
                    baseline_year, mid_start, dashboard_location, preindustrial_baselines
                )
                change_mid, pi_change_mid, value_mid = calculate_metric_change(
                    dashboard_data, metric_type, actual_metric_name, scenario,
                    baseline_year, long_start, dashboard_location, preindustrial_baselines
                )
                change_long, pi_change_long, value_long = calculate_metric_change(
                    dashboard_data, metric_type, actual_metric_name, scenario,
                    baseline_year, horizon_end, dashboard_location, preindustrial_baselines
                )
                
                is_inverse = should_use_inverse_delta(metric_type, metric_name)
                
                # Display cards using centralized HTML formatter
                with col_ref:
                    html = format_metric_card_html(
                        display_name, change_ref, unit, baseline_year,
                        value_ref, pi_change_ref, is_inverse
                    )
                    st.markdown(html, unsafe_allow_html=True)
                
                with col_short:
                    html = format_metric_card_html(
                        display_name, change_short, unit, baseline_year,
                        value_short, pi_change_short, is_inverse
                    )
                    st.markdown(html, unsafe_allow_html=True)
                
                with col_mid:
                    html = format_metric_card_html(
                        display_name, change_mid, unit, baseline_year,
                        value_mid, pi_change_mid, is_inverse
                    )
                    st.markdown(html, unsafe_allow_html=True)
                
                with col_long:
                    html = format_metric_card_html(
                        display_name, change_long, unit, baseline_year,
                        value_long, pi_change_long, is_inverse
                    )
                    st.markdown(html, unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
