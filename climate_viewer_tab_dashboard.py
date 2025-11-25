# -*- coding: utf-8 -*-
"""
Climate Viewer Dashboard Tab - Compact Version
Displays climate metrics across time horizons with pre-industrial baseline comparison.
Enhanced with bias-corrected rain metrics and PDF export.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from climate_viewer_constants import (
    FOLIUM_AVAILABLE,
    BASE_DIR,
    BASELINE_PERIOD,
    REFERENCE_YEAR,
    DASHBOARD_METRICS,
    should_use_inverse_delta,
    EMOJI,
)
from climate_viewer_data_operations import (
    find_metric_name,
    calculate_metric_change,
    calculate_preindustrial_baselines_by_metric,
    get_5year_average,
    format_compact_metric_card_html,
)
from climate_viewer_maps import create_region_map

if FOLIUM_AVAILABLE:
    from streamlit_folium import st_folium


def find_best_metric_name(data, metric_type, metric_name):
    """
    Find the best metric name, preferring bias-corrected versions for rain metrics.
    """
    available = data[data["Type"] == metric_type]["Name"].unique()

    if metric_type == "Rain":
        bc_version = f"{metric_name} (BC)"
        if bc_version in available:
            return bc_version

    return find_metric_name(data, metric_type, metric_name)


def generate_pdf_report(
        scenarios_data: dict,
        locations: list,
        baseline_year: int,
        mid_start: int,
        long_start: int,
        horizon_end: int,
        preindustrial_baselines: dict
) -> bytes:
    """
    Generate PDF report of dashboard metrics.
    Uses fpdf2 library for pure Python PDF generation.

    Returns:
        PDF as bytes for download
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 16)
            self.cell(0, 10, 'Climate Change Dashboard Report', align='C', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C', new_x='LMARGIN',
                      new_y='NEXT')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title and parameters
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Analysis Parameters', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Baseline Year: {baseline_year}', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Time Horizons: {mid_start} (Short), {long_start} (Medium), {horizon_end} (Long)', new_x='LMARGIN',
             new_y='NEXT')
    pdf.cell(0, 6, f'Locations: {", ".join(locations)}', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    # Process each scenario
    for scenario, data in scenarios_data.items():
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'Scenario: {scenario}', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(3)

        for location in locations:
            if location not in data:
                continue

            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, f'Location: {location}', new_x='LMARGIN', new_y='NEXT')
            pdf.ln(2)

            # Table header
            pdf.set_font('Helvetica', 'B', 9)
            col_widths = [50, 35, 35, 35, 35]
            headers = ['Metric', f'{baseline_year}', f'{mid_start}', f'{long_start}', f'{horizon_end}']

            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 7, header, border=1, align='C')
            pdf.ln()

            # Table data
            pdf.set_font('Helvetica', '', 8)

            for category, metrics in data[location].items():
                # Category row
                pdf.set_font('Helvetica', 'B', 8)
                pdf.set_fill_color(240, 240, 240)
                pdf.cell(sum(col_widths), 6, category, border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
                pdf.set_font('Helvetica', '', 8)

                for metric_info in metrics:
                    metric_name = metric_info['name']
                    unit = metric_info['unit']
                    values = metric_info['values']

                    # Format values
                    def fmt_val(v, is_baseline=False):
                        if v is None:
                            return 'N/A'
                        if is_baseline:
                            return f'{v:.1f}'
                        return f'{v:+.1f}'

                    pdf.cell(col_widths[0], 6, f'{metric_name} ({unit})', border=1)
                    pdf.cell(col_widths[1], 6, fmt_val(values.get('baseline'), True), border=1, align='C')
                    pdf.cell(col_widths[2], 6, fmt_val(values.get('short')), border=1, align='C')
                    pdf.cell(col_widths[3], 6, fmt_val(values.get('mid')), border=1, align='C')
                    pdf.cell(col_widths[4], 6, fmt_val(values.get('long')), border=1, align='C')
                    pdf.ln()

            pdf.ln(5)

    return bytes(pdf.output())


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
    Render the climate dashboard tab with compact layout and PDF export.
    """
    st.title(f"{EMOJI['chart']} Climate Change Dashboard")

    st.info(f"{EMOJI['lightbulb']} **Rainfall metrics use bias-corrected (BC) data** for improved accuracy.")

    dashboard_scenarios = [s for s in scen_sel if "historical" not in s.lower() and "agcd" not in s.lower()]

    if not dashboard_scenarios:
        st.warning(f"{EMOJI['warning']} Please select at least one future scenario (excluding historical) to view dashboard.")
        st.stop()

    # Show region map if available
    kml_path = os.path.join(BASE_DIR, "australia_grid_coverage.kml")
    ravenswood_kml_path = os.path.join(BASE_DIR, "ravenswood_grid_cell.kml")

    if FOLIUM_AVAILABLE and os.path.exists(kml_path):
        with st.expander(f"{EMOJI['map']} Region Map", expanded=False):
            st.markdown("**Grid coverage area for climate metrics**")
            region_map, status_msg, placemark_count = create_region_map(
                kml_path,
                ravenswood_kml_path if os.path.exists(ravenswood_kml_path) else None
            )
            if region_map:
                if placemark_count > 0:
                    st.caption(f"Displaying {placemark_count} grid cells from KML file")
                elif "Partial" in status_msg:
                    st.warning(f"{EMOJI['warning']} {status_msg}")
                st_folium(region_map, width=1100, height=600)
            else:
                st.info(f"Map unavailable: {status_msg}")
    elif not FOLIUM_AVAILABLE:
        st.info(f"{EMOJI['map']} Install folium and streamlit-folium to view region map")

    # Collect data for PDF export
    pdf_data = {}
    preindustrial_baselines = {}

    # Loop through all selected locations
    for location_idx, dashboard_location in enumerate(loc_sel):
        if location_idx > 0:
            st.markdown("---")

        st.markdown(f"## {EMOJI['pin']} {dashboard_location}")

        # Prepare dashboard data
        with st.spinner(f"Calculating metrics for {dashboard_location}..."):
            dashboard_data = df_all[
                (df_all["Location"] == dashboard_location) &
                (df_all["Season"] == "Annual") &
                (df_all["Scenario"].isin(dashboard_scenarios))
                ].copy()

            if dashboard_data.empty:
                st.warning(f"{EMOJI['warning']} No data available for {dashboard_location} and selected scenarios.")
                continue

            # Include Historical data for baseline if needed
            if baseline_year < REFERENCE_YEAR:
                hist_scenario = None
                for lbl in labels:
                    if "historical" in lbl.lower():
                        hist_scenario = lbl
                        break

                if hist_scenario and hist_scenario not in dashboard_scenarios:
                    hist_path = label_to_path[hist_scenario]
                    hist_df = load_metrics_func(hist_path)
                    hist_subset = hist_df[
                        (hist_df["Location"] == dashboard_location) &
                        (hist_df["Season"] == "Annual")
                        ].copy()
                    dashboard_data = pd.concat([dashboard_data, hist_subset], ignore_index=True)

            # Calculate pre-industrial baselines
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

        # Display metrics for each scenario
        for scenario in dashboard_scenarios:
            st.markdown("---")
            st.subheader(f"{EMOJI['globe']} {scenario}")

            # Initialise PDF data structure
            if scenario not in pdf_data:
                pdf_data[scenario] = {}
            if dashboard_location not in pdf_data[scenario]:
                pdf_data[scenario][dashboard_location] = {}

            scenario_baseline_year = baseline_year
            baseline_scenario = scenario
            if baseline_year < REFERENCE_YEAR:
                for lbl in labels:
                    if "historical" in lbl.lower():
                        baseline_scenario = lbl
                        break

            # Time horizon header - original sizes
            st.markdown(f"### {EMOJI['calendar']} Time Horizons")
            col_ref, col_short, col_mid, col_long = st.columns(4)

            with col_ref:
                baseline_label = 'BASELINE' if baseline_year != REFERENCE_YEAR else 'REFERENCE'
                st.markdown(f"<h2 style='text-align:center;color:#808080;margin:0;'>{scenario_baseline_year}</h2>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center;font-weight:bold;margin-top:-10px;'>{baseline_label}</p>",
                            unsafe_allow_html=True)
            with col_short:
                st.markdown(f"<h2 style='text-align:center;color:#3498DB;margin:0;'>{mid_start}</h2>",
                            unsafe_allow_html=True)
                st.markdown("<p style='text-align:center;font-weight:bold;margin-top:-10px;'>SHORT TERM</p>",
                            unsafe_allow_html=True)
            with col_mid:
                st.markdown(f"<h2 style='text-align:center;color:#F39C12;margin:0;'>{long_start}</h2>",
                            unsafe_allow_html=True)
                st.markdown("<p style='text-align:center;font-weight:bold;margin-top:-10px;'>MEDIUM TERM</p>",
                            unsafe_allow_html=True)
            with col_long:
                st.markdown(f"<h2 style='text-align:center;color:#E74C3C;margin:0;'>{horizon_end}</h2>",
                            unsafe_allow_html=True)
                st.markdown("<p style='text-align:center;font-weight:bold;margin-top:-10px;'>LONG TERM</p>",
                            unsafe_allow_html=True)

            # Display metrics by category
            for category, metrics in DASHBOARD_METRICS.items():
                st.markdown(f"<h3 style='margin:12px 0 8px 0;'>{category}</h3>", unsafe_allow_html=True)

                # Initialise PDF category
                if category not in pdf_data[scenario][dashboard_location]:
                    pdf_data[scenario][dashboard_location][category] = []

                for metric_type, metric_name, unit, icon, key in metrics:
                    actual_metric_name = find_best_metric_name(dashboard_data, metric_type, metric_name)
                    display_name = actual_metric_name.replace(" (BC)",
                                                              "") if "(BC)" in actual_metric_name else actual_metric_name

                    col_ref, col_short, col_mid, col_long = st.columns(4)

                    # Get baseline value
                    baseline_val = get_5year_average(
                        dashboard_data, metric_type, actual_metric_name,
                        baseline_scenario, scenario_baseline_year, dashboard_location
                    )

                    # Calculate values for each horizon
                    value_ref = baseline_val
                    change_ref = 0.0 if value_ref is not None else None
                    pi_change_ref = (value_ref - preindustrial_baselines.get((metric_type, actual_metric_name))) if (
                            value_ref is not None and preindustrial_baselines.get(
                        (metric_type, actual_metric_name)) is not None
                    ) else None

                    value_short = get_5year_average(
                        dashboard_data, metric_type, actual_metric_name,
                        scenario, mid_start, dashboard_location
                    )
                    change_short = (value_short - baseline_val) if (
                                value_short is not None and baseline_val is not None) else None
                    pi_change_short = (
                                value_short - preindustrial_baselines.get((metric_type, actual_metric_name))) if (
                            value_short is not None and preindustrial_baselines.get(
                        (metric_type, actual_metric_name)) is not None
                    ) else None

                    value_mid = get_5year_average(
                        dashboard_data, metric_type, actual_metric_name,
                        scenario, long_start, dashboard_location
                    )
                    change_mid = (value_mid - baseline_val) if (
                                value_mid is not None and baseline_val is not None) else None
                    pi_change_mid = (value_mid - preindustrial_baselines.get((metric_type, actual_metric_name))) if (
                            value_mid is not None and preindustrial_baselines.get(
                        (metric_type, actual_metric_name)) is not None
                    ) else None

                    value_long = get_5year_average(
                        dashboard_data, metric_type, actual_metric_name,
                        scenario, horizon_end, dashboard_location
                    )
                    change_long = (value_long - baseline_val) if (
                                value_long is not None and baseline_val is not None) else None
                    pi_change_long = (value_long - preindustrial_baselines.get((metric_type, actual_metric_name))) if (
                            value_long is not None and preindustrial_baselines.get(
                        (metric_type, actual_metric_name)) is not None
                    ) else None

                    is_inverse = should_use_inverse_delta(metric_type, metric_name)

                    # Store for PDF
                    pdf_data[scenario][dashboard_location][category].append({
                        'name': display_name,
                        'unit': unit,
                        'values': {
                            'baseline': value_ref,
                            'short': change_short,
                            'mid': change_mid,
                            'long': change_long
                        }
                    })

                    # Display compact cards
                    with col_ref:
                        html = format_compact_metric_card_html(
                            display_name, change_ref, unit, value_ref, pi_change_ref, is_inverse
                        )
                        st.markdown(html, unsafe_allow_html=True)

                    with col_short:
                        html = format_compact_metric_card_html(
                            display_name, change_short, unit, value_short, pi_change_short, is_inverse
                        )
                        st.markdown(html, unsafe_allow_html=True)

                    with col_mid:
                        html = format_compact_metric_card_html(
                            display_name, change_mid, unit, value_mid, pi_change_mid, is_inverse
                        )
                        st.markdown(html, unsafe_allow_html=True)

                    with col_long:
                        html = format_compact_metric_card_html(
                            display_name, change_long, unit, value_long, pi_change_long, is_inverse
                        )
                        st.markdown(html, unsafe_allow_html=True)

                st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)

    # PDF Export - single click download at bottom of dashboard
    if pdf_data:  # Only show if we have data
        pdf_bytes = generate_pdf_report(
            pdf_data,
            loc_sel,
            baseline_year,
            mid_start,
            long_start,
            horizon_end,
            preindustrial_baselines
        )

        if pdf_bytes:
            st.markdown("---")
            st.download_button(
                label=f"{EMOJI['page']} Download PDF Report",
                data=pdf_bytes,
                file_name=f"climate_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                help="Download complete dashboard as PDF to your Downloads folder"
            )
        else:
            st.error(f"{EMOJI['x_mark']} PDF generation failed. Install fpdf2: `pip install fpdf2`")