# -*- coding: utf-8 -*-
"""
Climate Viewer Global Milestones Tab
=====================================
climate_viewer_tab_global.py
Last Updated: 2025-11-28 16:45 AEST - Simplified to display-only

DISPLAY ONLY - all calculations done in main app.
Uses pre-calculated crossing years and baselines from session state.
"""

import pandas as pd
import altair as alt
from typing import Dict, List, Optional

from climate_viewer_constants import (
    BASELINE_PERIOD,
    AMPLIFICATION_FACTOR,
    EMOJI,
)

# Global warming thresholds for analysis
WARMING_THRESHOLDS = [1.5, 2.0, 2.5]

# SSP scenario data range
SSP_START_YEAR = 2015
SSP_END_YEAR = 2100


def render_global_tab(
        st,
        scen_sel: List[str],
        loc_sel: List[str],
        df_all: pd.DataFrame,
        crossing_years: Dict[str, Dict[str, Dict[float, Optional[int]]]],
        baselines: Dict[str, float],
        y0: int,
        y1: int,
        short_start: int,
        mid_start: int,
        long_start: int,
        horizon_end: int,
        smooth_win: int = 20,
        show_horizon_shading: bool = True
):
    """
    Render the Global Milestones tab.

    DISPLAY ONLY - uses pre-calculated data from main app.

    Args:
        st: Streamlit module
        scen_sel: Selected scenarios
        loc_sel: Selected locations
        df_all: Smoothed+aligned data (for impact extraction)
        crossing_years: Pre-calculated Dict[location -> Dict[scenario -> Dict[threshold -> year]]]
        baselines: Pre-calculated Dict[location -> baseline_temp]
        y0, y1: Year range for display
        short_start, mid_start, long_start, horizon_end: Time horizons
        smooth_win: Smoothing window used
    """
    st.title(f"{EMOJI['globe']} Global Milestones")

    # Clamp display range to SSP data
    chart_y0 = max(y0, SSP_START_YEAR)
    chart_y1 = min(y1, SSP_END_YEAR)

    st.markdown(f"""
    This analysis shows when each scenario crosses key global warming thresholds 
    (1.5°C, 2.0°C, 2.5°C above pre-industrial levels).

    **Regional amplification:** Australia warms approximately {AMPLIFICATION_FACTOR:.2f}× the global average.  
    When global temperature rises 1.5°C, regional warming is approximately {1.5 * AMPLIFICATION_FACTOR:.2f}°C.

    **Data:** {smooth_win}-year smoothed values, bias-corrected, aligned to AGCD at 2014.
    """)

    # Filter to SSP scenarios only
    global_scenarios = [s for s in scen_sel
                        if "historical" not in s.lower() and "agcd" not in s.lower()]

    if not global_scenarios:
        st.warning(f"{EMOJI['warning']} Select at least one SSP scenario to view threshold analysis.")
        return

    if not loc_sel:
        st.info("Select at least one location in the sidebar.")
        return

    # ========================================================================
    # DISPLAY FOR EACH LOCATION
    # ========================================================================

    for location in loc_sel:
        st.subheader(f"📍 {location}")

        if location not in crossing_years:
            st.info(f"No threshold data available for {location}")
            continue

        loc_crossings = crossing_years[location]
        baseline = baselines.get(location, 0)

        # Filter to selected scenarios
        filtered_crossings = {s: loc_crossings.get(s, {}) for s in global_scenarios if s in loc_crossings}

        if not filtered_crossings:
            st.info(f"No crossing data for selected scenarios at {location}")
            continue

        # Show baseline info
        st.caption(f"Pre-industrial baseline: {baseline:.2f}°C ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]})")

        # ================================================================
        # THRESHOLD CROSSING TIMELINE
        # ================================================================

        st.markdown(f"#### 📅 Global Average Temperatures")

        # Build timeline data
        timeline_data = []
        for scenario, thresholds in filtered_crossings.items():
            for threshold, year in thresholds.items():
                if year is not None:
                    timeline_data.append({
                        "Scenario": scenario,
                        "Threshold": f"{threshold}°C",
                        "Year": year,
                        "Year_label": str(year)
                    })

        if not timeline_data:
            st.info("No threshold crossings found for selected scenarios.")
            continue

        timeline_df = pd.DataFrame(timeline_data)

        # Toggle between chart and table
        display_mode = st.radio(
            "Display",
            ["Chart", "Table"],
            horizontal=True,
            key=f"timeline_display_{location}"
        )

        if display_mode == "Chart":
            # Filter to display range
            timeline_df_filtered = timeline_df[
                (timeline_df["Year"] >= chart_y0) &
                (timeline_df["Year"] <= chart_y1)
                ]

            if timeline_df_filtered.empty:
                st.info(f"No crossings within {chart_y0}-{chart_y1}")
                continue

            # Build chart layers
            chart_layers = []

            # Time horizon shading (if enabled)
            if show_horizon_shading:
                if short_start < mid_start and short_start >= chart_y0 and mid_start <= chart_y1:
                    short_rect = alt.Chart(pd.DataFrame({
                        'start': [max(short_start, chart_y0)],
                        'end': [min(mid_start, chart_y1)]
                    })).mark_rect(opacity=0.15, color='#3498DB').encode(
                        x=alt.X('start:Q', scale=alt.Scale(domain=[chart_y0, chart_y1])),
                        x2='end:Q'
                    )
                    chart_layers.append(short_rect)

                if mid_start < long_start and mid_start >= chart_y0 and long_start <= chart_y1:
                    mid_rect = alt.Chart(pd.DataFrame({
                        'start': [max(mid_start, chart_y0)],
                        'end': [min(long_start, chart_y1)]
                    })).mark_rect(opacity=0.15, color='#F39C12').encode(
                        x=alt.X('start:Q', scale=alt.Scale(domain=[chart_y0, chart_y1])),
                        x2='end:Q'
                    )
                    chart_layers.append(mid_rect)

                if long_start < horizon_end and long_start >= chart_y0 and horizon_end <= chart_y1:
                    long_rect = alt.Chart(pd.DataFrame({
                        'start': [max(long_start, chart_y0)],
                        'end': [min(horizon_end, chart_y1)]
                    })).mark_rect(opacity=0.15, color='#E74C3C').encode(
                        x=alt.X('start:Q', scale=alt.Scale(domain=[chart_y0, chart_y1])),
                        x2='end:Q'
                    )
                    chart_layers.append(long_rect)

            # Points
            points = alt.Chart(timeline_df_filtered).mark_circle(size=200, opacity=0.85).encode(
                x=alt.X('Year:Q',
                        scale=alt.Scale(domain=[chart_y0, chart_y1]),
                        axis=alt.Axis(title='Year', format='d')),
                y=alt.Y('Scenario:N', title='Scenario'),
                color=alt.Color('Threshold:N',
                                title='Global Warming Threshold',
                                scale=alt.Scale(
                                    domain=['1.5°C', '2.0°C', '2.5°C'],
                                    range=['#2ECC71', '#F39C12', '#E74C3C']
                                )),
                tooltip=['Scenario', 'Threshold', 'Year']
            )

            # Year labels
            text = alt.Chart(timeline_df_filtered).mark_text(
                align='center',
                baseline='bottom',
                dy=-12,
                fontSize=12
            ).encode(
                x='Year:Q',
                y='Scenario:N',
                text='Year_label:N'
            )

            chart_layers.extend([points, text])

            chart = alt.layer(*chart_layers).properties(
                height=250,
                width=800
            ).configure_axis(
                grid=False
            ).configure_view(
                strokeWidth=0
            )

            st.altair_chart(chart, use_container_width=True)

        else:
            # Table view
            table_data = []
            for scenario in sorted(filtered_crossings.keys()):
                row = {"Scenario": scenario}
                for threshold in WARMING_THRESHOLDS:
                    year = filtered_crossings[scenario].get(threshold)
                    row[f"{threshold}°C"] = str(year) if year else "---"
                table_data.append(row)

            st.dataframe(
                pd.DataFrame(table_data),
                hide_index=True,
                use_container_width=True
            )

        # ================================================================
        # IMPACTS AT THRESHOLDS
        # ================================================================

        st.markdown(f"#### 📊 Impacts at Global Warming Thresholds — {location}")

        # Build impacts table from df_all
        impacts_data = build_impacts_table(
            df_all, location, filtered_crossings, baseline, WARMING_THRESHOLDS
        )

        if impacts_data:
            st.dataframe(
                pd.DataFrame(impacts_data),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No impact data available.")

        st.markdown("---")

    # ========================================================================
    # METHODOLOGY
    # ========================================================================

    with st.expander("📖 Methodology"):
        st.markdown(f"""
        ### Threshold Crossing Detection

        - Uses {smooth_win}-year smoothed, bias-corrected temperature data
        - Scenarios aligned to AGCD observations at 2014
        - Crossing year is when smoothed value first exceeds threshold
        - Thresholds are GLOBAL warming levels (regional warming is higher)

        ### Pre-industrial Baseline

        - Period: {BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]} (IPCC AR6 standard)
        - Regional amplification: {AMPLIFICATION_FACTOR:.3f}× (Australia continental average)

        ### Why Scenarios Are Independent

        The Paris Agreement requires assessing resilience under DIFFERENT futures:
        - SSP1-26: Low emissions pathway
        - SSP3-70: High emissions pathway

        These are separate stress tests, not a range of outcomes.
        """)


def build_impacts_table(
        df: pd.DataFrame,
        location: str,
        crossings: Dict[str, Dict[float, Optional[int]]],
        baseline: float,
        thresholds: List[float]
) -> List[Dict]:
    """
    Build impacts table showing values at threshold crossing years.

    Returns list of dicts for DataFrame.
    """
    # Metrics to extract
    metrics = [
        ("Temp", "Average", "Mean Temp", "°C"),
        ("Temp", "Max Day", "Max Temp", "°C"),
        ("Rain", "Total", "Annual Rain", "mm"),
    ]

    results = []

    for scenario, thresh_years in crossings.items():
        row = {"Scenario": scenario}

        for threshold in thresholds:
            year = thresh_years.get(threshold)
            if year is None:
                row[f"{threshold}°C"] = "---"
                continue

            # Get temperature at crossing year
            temp_data = df[
                (df["Scenario"] == scenario) &
                (df["Location"] == location) &
                (df["Type"] == "Temp") &
                (df["Name"] == "Average") &
                (df["Season"] == "Annual") &
                (df["Year"] == year)
                ]

            if not temp_data.empty:
                temp = temp_data["Value"].iloc[0]
                warming = temp - baseline
                row[f"{threshold}°C"] = f"{temp:.1f}°C (+{warming:.1f}°C)"
            else:
                row[f"{threshold}°C"] = "---"

        results.append(row)

    return results