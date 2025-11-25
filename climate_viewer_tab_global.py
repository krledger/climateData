# -*- coding: utf-8 -*-
"""
Climate Viewer Global 1.5°C Impact Tab
Shows regional climate impacts when GLOBAL warming reaches 1.5°C above pre-industrial levels.
Uses proper pre-industrial baselines and regional amplification factors.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from climate_viewer_constants import (
    BASELINE_PERIOD,
    WARMING_TARGET,
    HARDCODED_GLOBAL_BASELINE,
    HARDCODED_AUSTRALIA_BASELINE,
    AMPLIFICATION_FACTOR,
    DASHBOARD_METRICS,
    SCENARIO_DESCRIPTIONS,
    get_scenario_description,
    should_use_inverse_delta,
)
from climate_viewer_data_operations import (
    calculate_preindustrial_baseline,
    find_year_at_global_warming_target,
    extract_conditions_at_year,
)


def get_best_metric_name_for_extraction(metric_type: str, metric_name: str) -> str:
    """
    Get best metric name, preferring bias-corrected versions for rain.

    Args:
        metric_type: Type of metric (e.g. 'Temp', 'Rain')
        metric_name: Base metric name

    Returns:
        Metric name to use (with BC suffix for rain if available)
    """
    if metric_type == "Rain":
        return f"{metric_name} (BC)"
    return metric_name


def get_compliance_status(scenario):
    """
    Determine compliance status for scenario.

    Returns:
        tuple: (status_text, color_emoji)
    """
    if "SSP1-26" in scenario or "1-26" in scenario:
        return "✅ Compliant (Best case)", "🟢"
    elif "SSP1-1.9" in scenario or "1-1.9" in scenario:
        return "✅ Compliant (Best case)", "🟢"
    elif "SSP2-45" in scenario or "2-45" in scenario:
        return "⚠️ Warning (Middle pathway)", "🟡"
    elif "SSP3-70" in scenario or "3-70" in scenario:
        return "✅ Compliant (Worst case)", "🟢"
    elif "SSP5-85" in scenario or "5-85" in scenario:
        return "❌ Non-Compliant", "🔴"
    else:
        return "⚠️ Unknown", "🟡"


def generate_global_pdf_report(
        all_location_results: dict,
        selected_locations: list,
        global_scenarios: list,
        warming_target: float,
        amplification_factor: float,
        baseline_period: tuple
) -> bytes:
    """
    Generate PDF report of global 1.5°C impact analysis.

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
            self.cell(0, 10, f'Global {warming_target}C Impact Report', align='C', new_x='LMARGIN', new_y='NEXT')
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

    # Summary section
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Analysis Summary', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6,
             f'Warming Target: {warming_target}C above pre-industrial ({baseline_period[0]}-{baseline_period[1]})',
             new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Amplification Factor: {amplification_factor}x (regional vs global)', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Locations: {", ".join(selected_locations)}', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    # Year comparison table
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, f'Year When {warming_target}C Global Warming Is Reached', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)

    # Table header
    pdf.set_font('Helvetica', 'B', 9)
    col_widths = [40, 50] + [25] * len(selected_locations)
    headers = ['Scenario', 'Compliance'] + selected_locations

    for i, header in enumerate(headers):
        w = col_widths[i] if i < len(col_widths) else 25
        pdf.cell(w, 7, header[:15], border=1, align='C')
    pdf.ln()

    # Table data
    pdf.set_font('Helvetica', '', 8)
    for scenario in global_scenarios:
        compliance_status, _ = get_compliance_status(scenario)
        # Strip emoji for PDF
        compliance_text = compliance_status.replace('✅ ', '').replace('❌ ', '').replace('⚠️ ', '')

        pdf.cell(col_widths[0], 6, scenario, border=1)
        pdf.cell(col_widths[1], 6, compliance_text[:20], border=1)

        for i, location in enumerate(selected_locations):
            w = col_widths[2 + i] if 2 + i < len(col_widths) else 25
            if location in all_location_results and scenario in all_location_results[location]:
                year = str(all_location_results[location][scenario]["year"])
            else:
                year = "N/A"
            pdf.cell(w, 6, year, border=1, align='C')
        pdf.ln()

    pdf.ln(5)

    # Detailed results for each location
    for location in selected_locations:
        if location not in all_location_results or not all_location_results[location]:
            continue

        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'Location: {location}', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(3)

        location_results = all_location_results[location]

        for scenario, data in sorted(location_results.items(), key=lambda x: x[1]["year"]):
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 8, f'Scenario: {scenario}', new_x='LMARGIN', new_y='NEXT')

            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 6, f'Year: {data["year"]} | Regional warming: {data["regional_warming"]:.2f}C', new_x='LMARGIN',
                     new_y='NEXT')
            pdf.ln(2)

            # Conditions table
            conditions = data.get("conditions", {})
            if conditions:
                pdf.set_font('Helvetica', 'B', 9)
                pdf.cell(60, 6, 'Metric', border=1)
                pdf.cell(30, 6, 'Change', border=1, align='C')
                pdf.cell(30, 6, 'Value', border=1, align='C')
                pdf.cell(20, 6, 'Unit', border=1, align='C')
                pdf.ln()

                pdf.set_font('Helvetica', '', 8)
                for key, c in conditions.items():
                    if not np.isnan(c.get("change", np.nan)):
                        pdf.cell(60, 5, c.get("label", key)[:25], border=1)
                        pdf.cell(30, 5, f'{c["change"]:+.1f}', border=1, align='C')
                        pdf.cell(30, 5, f'{c["value"]:.1f}', border=1, align='C')
                        pdf.cell(20, 5, c.get("unit", ""), border=1, align='C')
                        pdf.ln()

            pdf.ln(5)

    return bytes(pdf.output())


def render_global_tab(
        st,
        scen_sel,
        loc_sel,
        df_all,
        labels,
        label_to_path,
        load_metrics_func
):
    """
    Render the global 1.5°C impact tab.

    Args:
        st: Streamlit module
        scen_sel: Selected scenarios
        loc_sel: Selected locations
        df_all: All loaded data
        labels: All scenario labels
        label_to_path: Mapping of label to file path
        load_metrics_func: Function to load metrics (NOT CACHED)
    """
    st.title("🌍 Global 1.5°C Impact Display")

    st.markdown(f"""
    This analysis shows regional climate impacts when **global warming reaches {WARMING_TARGET}°C** above pre-industrial levels ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]} baseline).

    **Key Concept**: Regional warming is typically {AMPLIFICATION_FACTOR}× higher than global average due to land-ocean differences.
    When global temperature rises {WARMING_TARGET}°C, most land regions warm by ~{WARMING_TARGET * AMPLIFICATION_FACTOR:.1f}°C.
    """)

    st.info(
        "💡 **Rainfall metrics use bias-corrected (BC) data** to account for systematic model errors and improve accuracy.")

    # Get non-historical scenarios
    global_scenarios = [s for s in scen_sel if "historical" not in s.lower()]

    if not global_scenarios:
        st.warning("⚠️ Please select at least one future scenario (excluding historical) to view global impacts.")
        st.stop()

    # Use locations from sidebar
    selected_locations = loc_sel

    if not selected_locations:
        st.info("Please select at least one location in the sidebar to view impacts.")
        st.stop()

    # Calculate results for all locations
    st.markdown("---")
    st.subheader("⏰ When Does 1.5°C Global Warming Occur?")

    with st.spinner("Analysing scenarios across locations..."):
        all_location_results = {}

        # Prepare metrics list for extract_conditions_at_year
        metrics_list = []
        for category, metrics in DASHBOARD_METRICS.items():
            for metric_type, name, unit, icon, key in metrics:
                best_name = get_best_metric_name_for_extraction(metric_type, name)
                metrics_list.append((metric_type, best_name, key, unit))

        for location in selected_locations:
            # Calculate regional pre-industrial baseline
            historical_data = df_all[df_all["Scenario"].str.lower().str.contains("historical", na=False)]

            if not historical_data.empty:
                regional_baseline = calculate_preindustrial_baseline(
                    historical_data, BASELINE_PERIOD, location
                )
            else:
                regional_baseline = np.nan

            # Fallback to hardcoded value
            if np.isnan(regional_baseline):
                if location == "Australia" or "Australia" in location:
                    regional_baseline = HARDCODED_AUSTRALIA_BASELINE
                else:
                    regional_baseline = HARDCODED_GLOBAL_BASELINE + 7.8

            location_results = {}

            for scenario in global_scenarios:
                scenario_data = df_all[df_all["Scenario"] == scenario]

                year, regional_warming, global_warming = find_year_at_global_warming_target(
                    scenario_data, scenario, location, regional_baseline,
                    WARMING_TARGET, AMPLIFICATION_FACTOR
                )

                if year and regional_warming and global_warming:
                    conditions = extract_conditions_at_year(
                        scenario_data, scenario, location, year, 2020, metrics_list
                    )

                    location_results[scenario] = {
                        "year": year,
                        "regional_warming": regional_warming,
                        "global_warming": global_warming,
                        "conditions": conditions,
                        "baseline": regional_baseline
                    }

            all_location_results[location] = location_results

    # Display comparison table
    st.markdown(f"""
    **Regulatory Context (Australian Corporations Act):**

    Climate-related financial disclosures must include scenario analysis using:
    - **SSP1-26**: Paris Agreement pathway (limited to 1.5°C) - *Best case*
    - **SSP3-70** or higher: Well exceeds 2°C pathway - *Worst case*
    """)

    # Create comparison table
    comparison_data = []
    for scenario in global_scenarios:
        row = {"Scenario": scenario}
        compliance_status, compliance_color = get_compliance_status(scenario)
        row["Compliance"] = f"{compliance_color} {compliance_status}"
        row["Description"] = get_scenario_description(scenario)

        for location in selected_locations:
            if location in all_location_results and scenario in all_location_results[location]:
                year = all_location_results[location][scenario]["year"]
                row[location] = year
            else:
                row[location] = None

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by first location's year
    first_loc = selected_locations[0]
    if first_loc in comparison_df.columns:
        comparison_df = comparison_df.sort_values(
            by=first_loc,
            key=lambda x: x.fillna(9999)
        )

    st.markdown("### 📅 Year When 1.5°C Global Warming Is Reached")
    st.dataframe(
        comparison_df,
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
            "Compliance": st.column_config.TextColumn("Compliance", width="large"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            **{loc: st.column_config.NumberColumn(loc, format="%d", width="small")
               for loc in selected_locations}
        },
        hide_index=True,
        width='stretch'
    )

    st.caption(f"""
    **Methodology:** Regional warming calculated for each location using pre-industrial baseline (1850-1900).
    Global warming estimated by dividing regional warming by {AMPLIFICATION_FACTOR}× amplification factor.
    Year shown is when 5-year rolling average first exceeds 1.5°C global warming threshold.
    Empty cells indicate threshold not reached in available data range.
    """)

    # Display baseline information
    if selected_locations:
        baseline_info = []
        for loc in selected_locations:
            if loc in all_location_results and all_location_results[loc]:
                first_scenario = list(all_location_results[loc].keys())[0]
                baseline_info.append(f"{loc}: {all_location_results[loc][first_scenario]['baseline']:.1f}°C")
            else:
                baseline_info.append(f"{loc}: N/A")

        st.info(f"""
        **Baselines ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]} average)**:
        - Global: {HARDCODED_GLOBAL_BASELINE:.1f}°C
        - Selected locations: {', '.join(baseline_info)}
        - Amplification factor: {AMPLIFICATION_FACTOR}×
        """)

    # Display impacts for each location and scenario
    st.markdown("---")
    st.subheader("📊 Regional Impacts at 1.5°C Global Warming")
    st.caption("Changes shown are relative to 2020 baseline (5-year averages)")

    for location in selected_locations:
        if location not in all_location_results or not all_location_results[location]:
            st.warning(f"No scenarios reach 1.5°C global warming for {location} in the available data range.")
            continue

        st.markdown(f"## 📍 {location}")

        location_results = all_location_results[location]

        for scenario, data in sorted(location_results.items(), key=lambda x: x[1]["year"]):
            st.markdown(f"### 🌍 {scenario}")

            regional_temp = data["regional_warming"]
            global_temp = data["global_warming"]
            year = data["year"]

            compliance_status, compliance_color = get_compliance_status(scenario)

            # Header box with compliance indicator
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; border-radius: 10px; color: white; margin-bottom: 15px;'>
                    <h2 style='margin: 0; color: white;'>IMPACTS AT {WARMING_TARGET}°C GLOBAL WARMING - {location}</h2>
                    <p style='font-size: 16px; margin: 8px 0 0 0;'>
                        <strong>Year:</strong> {year} | 
                        <strong>Global warming:</strong> {global_temp:.2f}°C | 
                        <strong>Regional warming:</strong> {regional_temp:.2f}°C above pre-industrial
                    </p>
                    <p style='font-size: 14px; margin: 8px 0 0 0;'>
                        <strong>Compliance Status:</strong> {compliance_status}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            conditions = data["conditions"]

            # Display metrics by category
            for category, metrics in DASHBOARD_METRICS.items():
                st.markdown(f"<h3 style='margin: 10px 0 6px 0;'>{category}</h3>", unsafe_allow_html=True)

                cols = st.columns(len(metrics))

                for col_idx, (metric_type, metric_name, unit, icon, key) in enumerate(metrics):
                    with cols[col_idx]:
                        if key in conditions and not np.isnan(conditions[key]["change"]):
                            c = conditions[key]

                            is_inverse = should_use_inverse_delta(metric_type, metric_name)
                            delta_color = "inverse" if is_inverse else "normal"

                            st.metric(
                                f"{icon} {metric_name}",
                                f"{c['change']:+.1f} {c['unit']}",
                                f"At {c['value']:.1f} {c['unit']}",
                                delta_color=delta_color
                            )
                        else:
                            st.metric(
                                f"{icon} {metric_name}",
                                "N/A",
                                None
                            )

            st.markdown("---")

    # PDF Export
    if all_location_results:
        pdf_bytes = generate_global_pdf_report(
            all_location_results,
            selected_locations,
            global_scenarios,
            WARMING_TARGET,
            AMPLIFICATION_FACTOR,
            BASELINE_PERIOD
        )

        if pdf_bytes:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"global_impact_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                help="Download complete analysis as PDF"
            )
        else:
            st.info("Install fpdf2 for PDF export: `pip install fpdf2`")

    # Explanatory note
    with st.expander("ℹ️ About This Analysis"):
        st.markdown(f"""
        ### Understanding Global vs Regional Warming

        - **Global warming** = Earth's average surface temperature (71% ocean, 29% land)
        - **Regional warming** = Temperature over land areas
        - **Land areas warm faster** because oceans absorb heat slowly (high heat capacity)
        - **Amplification factor** = Regional warming / Global warming ≈ {AMPLIFICATION_FACTOR}× for mid-latitude land

        ### Calculation Method

        1. Calculate regional warming from pre-industrial baseline ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]})
        2. Estimate global warming = Regional warming ÷ {AMPLIFICATION_FACTOR}
        3. Find year when estimated global warming reaches {WARMING_TARGET}°C
        4. Show climate conditions at that year

        ### Display Format

        - **Large number**: Change from 2020 (matches Dashboard tab format)
        - **Small number**: Absolute value at the {WARMING_TARGET}°C year (5-year average)
        - **Delta colours**: Green = improvement, Red = worsening (inverse for drought metrics)

        ### Compliance Assessment

        - 🟢 **Compliant**: SSP1-26, SSP1-1.9 (best case), SSP3-70 (worst case)
        - 🟡 **Warning**: SSP2-45 (middle pathway, not explicitly required)
        - 🔴 **Non-Compliant**: SSP5-85 and other high-emission scenarios

        ### Why These Years?

        - **High emissions** (SSP5-85): Reaches {WARMING_TARGET}°C earliest (~2025-2030)
        - **Medium emissions** (SSP2-45, SSP3-70): Middle timeframe (~2028-2033)
        - **Low emissions** (SSP1-26): Reaches latest (~2030-2035) or may not reach at all

        ### Scientific Basis

        - Pre-industrial baseline: {BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]} (IPCC standard)
        - Global baseline: {HARDCODED_GLOBAL_BASELINE:.1f}°C (IPCC AR6)
        - Amplification: Well-established in climate science (IPCC AR6 WG1 Chapter 4)
        - Paris Agreement: {WARMING_TARGET}°C target refers to GLOBAL warming

        ### Data Quality Note

        These estimates are based on regional temperature data and standard amplification factors.
        Actual global temperatures may vary. Results are most reliable when historical data ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]}) is available.
        """)