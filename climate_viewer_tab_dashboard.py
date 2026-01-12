# -*- coding: utf-8 -*-
"""
Climate Viewer Dashboard Tab
=============================
climate_viewer_tab_dashboard.py
Last Updated: 2025-12-02 14:30 AEST - Simplified, uses climate_viewer_maps module

Features:
- Interactive map via climate_viewer_maps.create_multi_location_map()
- Climate metrics displayed in simple tables at time horizon years
- Uses pre-processed data from main app (display only)
"""

import pandas as pd
from pathlib import Path

from climate_viewer_constants import EMOJI, DASHBOARD_METRICS, should_use_inverse_delta, FOLIUM_AVAILABLE
from climate_viewer_maps import create_multi_location_map, get_location_colour, get_map_height


def get_value_at_year(df: pd.DataFrame, metric_type: str, metric_name: str,
                      scenario: str, year: int, location: str) -> float:
    """Get the value for a metric at a specific year.  Simple lookup."""
    mask = (
            (df["Location"] == location) &
            (df["Type"] == metric_type) &
            (df["Name"] == metric_name) &
            (df["Scenario"] == scenario) &
            (df["Year"] == year) &
            (df["Season"] == "Annual")
    )
    data = df[mask]
    if data.empty:
        return None
    return data["Value"].iloc[0]


def find_metric_name(df: pd.DataFrame, metric_type: str, preferred_name: str) -> str:
    """Find actual metric name in data, preferring BC version for Rain."""
    available = df[df["Type"] == metric_type]["Name"].unique()

    if metric_type == "Rain":
        bc_name = f"{preferred_name} (BC)"
        if bc_name in available:
            return bc_name

    if preferred_name in available:
        return preferred_name

    for name in available:
        if preferred_name.lower() in name.lower():
            return name

    return preferred_name


def format_metric_card(name: str, value: float, change: float, unit: str,
                       is_inverse: bool = False) -> str:
    """Format a compact metric card as HTML."""
    if value is None:
        return f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:8px;margin:2px 0;text-align:center;">
            <div style="font-size:11px;color:#666;margin-bottom:2px;">{name}</div>
            <div style="font-size:16px;font-weight:bold;color:#999;">N/A</div>
        </div>
        """

    if change is None:
        colour = "#666"
        change_str = "N/A"
    elif change == 0:
        colour = "#666"
        change_str = "+0.0"
    else:
        if is_inverse:
            colour = "#E74C3C" if change < 0 else "#27AE60"
        else:
            colour = "#E74C3C" if change > 0 else "#27AE60"
        change_str = f"{change:+.1f}"

    return f"""
    <div style="background:#f8f9fa;border-radius:8px;padding:8px;margin:2px 0;text-align:center;">
        <div style="font-size:11px;color:#666;margin-bottom:2px;">{name}</div>
        <div style="font-size:16px;font-weight:bold;color:{colour};">{change_str} {unit}</div>
        <div style="font-size:10px;color:#888;">({value:.1f})</div>
    </div>
    """


def render_location_map(st, loc_sel):
    """
    Render the location map using the maps module.

    Returns dict of loaded location info for legend display.
    """
    if not FOLIUM_AVAILABLE:
        st.info(f"{EMOJI['map']} Map view requires folium and streamlit-folium packages.")
        st.code("pip install folium streamlit-folium")
        return {}

    from streamlit_folium import st_folium

    # Create map via maps module
    m, loaded_locations = create_multi_location_map(loc_sel)

    if m is None:
        st.warning(f"{EMOJI['warning']} Could not create map: {loaded_locations.get('error', 'Unknown error')}")
        return {}

    if not loaded_locations:
        st.warning(
            f"{EMOJI['warning']} No KML files found for selected locations.  Run generator_kml.py to create them.")
        return {}

    # Display map
    st_folium(m, width=None, height=get_map_height(), returned_objects=[])

    return loaded_locations


def render_dashboard_tab(
        st,
        scen_sel,
        loc_sel,
        df_all,
        labels,
        label_to_path,
        short_start,
        mid_start,
        long_start,
        horizon_end
):
    """
    Render the climate dashboard tab.

    Features:
    - Interactive map showing grid cells for selected locations
    - Climate metrics displayed in simple tables at time horizon years
    - Uses pre-processed data from main app (display only)
    """
    st.title(f"{EMOJI['chart']} Climate Change Dashboard")

    # Filter to SSP scenarios only
    dashboard_scenarios = [s for s in scen_sel
                           if "historical" not in s.lower() and "agcd" not in s.lower()]

    if not dashboard_scenarios:
        st.warning(f"{EMOJI['warning']} Select at least one SSP scenario to view dashboard.")
        st.stop()

    if df_all.empty:
        st.warning(f"{EMOJI['warning']} No data available.")
        st.stop()

    # ========================================================================
    # MAP VIEW
    # ========================================================================
    st.subheader(f"{EMOJI['map']} Location Map")

    with st.expander("View Grid Coverage Map", expanded=True):
        loaded_locations = render_location_map(st, loc_sel)

        # Dynamic legend - Streamlit native
        if loaded_locations:
            st.divider()
            cols = st.columns(min(len(loaded_locations), 4))
            for idx, (loc, info) in enumerate(loaded_locations.items()):
                with cols[idx % len(cols)]:
                    cell_text = f"{info['cells']} cells" if info['cells'] != 1 else "1 cell"
                    st.caption(f"● {loc} — {cell_text}")

    st.divider()

    # ========================================================================
    # TIME HORIZON METRICS
    # ========================================================================

    horizons = [
        ("Short", short_start),
        ("Medium", mid_start),
        ("Long", long_start),
        ("End", horizon_end),
    ]

    for loc_idx, location in enumerate(loc_sel):
        if loc_idx > 0:
            st.divider()

        st.header(f"{EMOJI['pin']} {location}")

        loc_data = df_all[df_all["Location"] == location]

        if loc_data.empty:
            st.warning(f"{EMOJI['warning']} No data for {location}")
            continue

        for scenario in dashboard_scenarios:
            st.subheader(f"{EMOJI['globe']} {scenario}")

            # Build table data for all metrics
            table_rows = []

            for category, metrics in DASHBOARD_METRICS.items():
                for metric_type, metric_name, unit, icon, key in metrics:
                    actual_name = find_metric_name(loc_data, metric_type, metric_name)
                    display_name = actual_name.replace(" (BC)", "")

                    short_term_val = get_value_at_year(
                        loc_data, metric_type, actual_name, scenario, short_start, location
                    )

                    row = {"Type": metric_type, "Metric": display_name}

                    for label, year in horizons:
                        value = get_value_at_year(
                            loc_data, metric_type, actual_name, scenario, year, location
                        )

                        if value is not None and short_term_val is not None:
                            change = value - short_term_val
                            change_str = f"{change:+.1f}" if change != 0 else "+0.0"
                            row[f"{year}"] = f"{change_str}{unit} ({value:.1f})"
                        elif value is not None:
                            row[f"{year}"] = f"{value:.1f}{unit}"
                        else:
                            row[f"{year}"] = "---"

                    table_rows.append(row)

            # Display as single DataFrame table
            if table_rows:
                df_table = pd.DataFrame(table_rows)
                st.dataframe(df_table, hide_index=True, width="stretch")

    st.divider()
    st.caption(f"Data from {len(dashboard_scenarios)} scenario(s) across {len(loc_sel)} location(s). "
               f"Horizons: Short ({short_start}), Medium ({mid_start}), Long ({long_start}), End ({horizon_end}).")