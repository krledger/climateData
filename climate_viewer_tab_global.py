"""
Climate Viewer Global 1.5°C Impact Tab
Shows regional climate impacts when GLOBAL warming reaches 1.5°C above pre-industrial levels.
Uses proper pre-industrial baselines and regional amplification factors.
"""

import pandas as pd
import numpy as np

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
from climate_viewer_utils import (
    calculate_preindustrial_baseline,
    find_year_at_global_warming_target,
    extract_conditions_at_year,
)


def assess_scenario_compliance(scenario: str, crossing_year: int = None) -> tuple:
    """
    Assess Paris Agreement compliance for a scenario.
    
    Paris Agreement Article 2.1.a aims to hold "the increase in the global 
    average temperature to well below 2°C above pre-industrial levels and 
    pursuing efforts to limit the temperature increase to 1.5°C".
    
    Args:
        scenario: Scenario name (e.g., "SSP1-26", "SSP5-85")
        crossing_year: Year when 1.5°C is sustainably exceeded (or None)
        
    Returns:
        Tuple of (status_with_emoji, reasoning):
        - status_with_emoji: "🟢 Compliant" or similar with emoji
        - reasoning: Explanation of compliance assessment
    """
    scenario_upper = scenario.upper()
    
    # SSP1-1.9: Limits warming to 1.5°C with no or limited overshoot
    if "SSP1-1.9" in scenario_upper or "1-1.9" in scenario_upper:
        return (
            "🟢 Compliant (Best Case)",
            "Limits warming to 1.5°C with no or limited overshoot. "
            "Represents ambitious mitigation aligned with Paris Agreement's "
            "aspirational 1.5°C target."
        )
    
    # SSP1-2.6: Limits warming to well below 2°C
    if "SSP1-26" in scenario_upper or "SSP1-2.6" in scenario_upper or "1-26" in scenario_upper:
        year_text = f"Reaches 1.5°C around {crossing_year}" if crossing_year else "Crosses 1.5°C in early 2030s"
        return (
            "🟢 Compliant",
            f"Peak warming stays well below 2°C through strong and sustained mitigation. "
            f"Consistent with Paris Agreement's core temperature goal. "
            f"{year_text} but stabilises below 2°C."
        )
    
    # SSP2-4.5: Middle pathway, warming around 2-2.5°C
    if "SSP2-45" in scenario_upper or "SSP2-4.5" in scenario_upper or "2-45" in scenario_upper:
        return (
            "🟡 Warning",
            "Peak warming likely exceeds 2°C. Represents intermediate pathway "
            "with moderate climate action. May exceed Paris Agreement's 'well below 2°C' "
            "objective. Useful for assessing risks under delayed mitigation."
        )
    
    # SSP3-7.0: Regional rivalry, warming 2.5-3.5°C
    if "SSP3-70" in scenario_upper or "SSP3-7.0" in scenario_upper or "3-70" in scenario_upper:
        year_text = f"Crosses 1.5°C early (around {crossing_year})" if crossing_year else "Crosses 1.5°C in mid-2020s"
        return (
            "🟢 Compliant (Worst Case)",
            f"Represents upper bound of scenarios for stress testing resilience and "
            f"adaptation capacity. {year_text}. While warming exceeds 2°C, "
            f"used to test outer limits of physical climate risks and evaluate "
            f"adaptation requirements under challenging conditions."
        )
    
    # SSP5-8.5: High emissions, warming >4°C
    if "SSP5-85" in scenario_upper or "SSP5-8.5" in scenario_upper or "5-85" in scenario_upper:
        return (
            "🔴 Non-Compliant",
            "High emissions pathway with warming well exceeding 2°C threshold. "
            "Peak warming >4°C by end of century represents failure to achieve "
            "Paris Agreement goals. Used as high-risk reference scenario for "
            "stress testing extreme physical climate impacts."
        )
    
    # Default for unknown scenarios
    if crossing_year and crossing_year < 2030:
        return (
            "🟡 Warning",
            f"Crosses 1.5°C threshold in {crossing_year}. Compliance depends on "
            "peak warming level and long-term trajectory. Further analysis required."
        )
    elif crossing_year is None:
        return (
            "🟢 Compliant",
            "Does not exceed 1.5°C global warming within projection period. "
            "Represents strong mitigation consistent with Paris Agreement."
        )
    else:
        return (
            "🟡 Warning",
            f"Crosses 1.5°C in {crossing_year}. Compliance assessment requires "
            "additional information on peak warming and long-term pathway."
        )


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
    st.markdown("## 🌍 Global 1.5°C Impact Display")

    st.markdown(f"""
    <div style='font-size: 0.9em; margin-bottom: 10px;'>
    This analysis shows regional climate impacts when <strong>global warming reaches {WARMING_TARGET}°C</strong> above pre-industrial levels ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]} baseline).
    Regional warming is typically {AMPLIFICATION_FACTOR}× higher than global average due to land-ocean differences.
    </div>
    """, unsafe_allow_html=True)

    # Get non-historical scenarios
    global_scenarios = [s for s in scen_sel if "historical" not in s.lower()]

    if not global_scenarios:
        st.warning("⚠️ Please select at least one future scenario (excluding historical) to view global impacts.")
        st.stop()

    # Multi-location selector
    st.markdown("<h4 style='margin:5px 0;'>Select Locations</h4>", unsafe_allow_html=True)
    selected_locations = st.multiselect(
        "Choose one or more locations to analyse",
        loc_sel,
        default=[loc_sel[0]] if loc_sel else [],
        key="global_locations"
    )

    if not selected_locations:
        st.info("Please select at least one location to view impacts.")
        st.stop()

    # Calculate results for all locations
    st.markdown("<hr style='margin:10px 0;'><h4 style='margin:5px 0;'>⏰ When Does 1.5°C Global Warming Occur?</h4>", unsafe_allow_html=True)

    with st.spinner("Analysing scenarios across locations..."):
        all_location_results = {}

        # Prepare metrics list for extract_conditions_at_year
        metrics_list = []
        for category, metrics in DASHBOARD_METRICS.items():
            for metric_type, name, unit, icon, key in metrics:
                metrics_list.append((metric_type, name, key, unit))

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
                    # Extract conditions at that year (change from 2020)
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

    # Display comparison table (always show, even for single location)
    st.markdown(f"""
    <div style='font-size: 0.95em; background: #f8f9fa; padding: 8px; border-radius: 4px; margin: 8px 0;'>
    <strong>Regulatory Context:</strong><br>
    <strong>Low warming (≤1.5°C)</strong>: Strong climate action, high transition risk, lower physical risk.<br> 
    <strong>High warming (≥2.5°C)</strong>: Limited action, lower transition risk, higher physical risk.
    </div>
    """, unsafe_allow_html=True)

    # Create comparison table
    comparison_data = []
    for scenario in global_scenarios:
        # Get crossing year for compliance assessment (use first location as reference)
        crossing_year = None
        if selected_locations and selected_locations[0] in all_location_results:
            if scenario in all_location_results[selected_locations[0]]:
                crossing_year = all_location_results[selected_locations[0]][scenario].get("year")

        # Assess compliance using algorithm
        status_with_emoji, reasoning = assess_scenario_compliance(scenario, crossing_year)

        row = {"Scenario": scenario}
        row["Compliance"] = status_with_emoji
        row["Description"] = reasoning  # Compliance reasoning instead of scenario description

        # Add location columns
        for location in selected_locations:
            if location in all_location_results and scenario in all_location_results[location]:
                year = all_location_results[location][scenario]["year"]
                row[location] = year
            else:
                row[location] = "N/A"

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by first location's year
    first_loc = selected_locations[0]
    if first_loc in comparison_df.columns:
        comparison_df = comparison_df.sort_values(
            by=first_loc,
            key=lambda x: x.replace("N/A", "9999").astype(str).astype(int)
        )

    st.markdown("<h4 style='margin:8px 0 5px 0;'>📅 Year When 1.5°C Global Warming Is Reached</h4>", unsafe_allow_html=True)
    st.dataframe(
        comparison_df,
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
            "Compliance": st.column_config.TextColumn("Compliance", width="medium"),
            "Description": st.column_config.TextColumn(
                "Compliance Reasoning",
                width="large",
                help="Explanation of why scenario is assessed as compliant, warning, or non-compliant based on Paris Agreement temperature goals"
            ),
            **{loc: st.column_config.NumberColumn(loc, format="%d", width="small")
               for loc in selected_locations}
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown(f"""
    <div style='font-size: 0.75em; color: #666; margin: 5px 0;'>
    <strong>Methodology:</strong> Regional warming from pre-industrial baseline ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]}). 
    Global warming = Regional ÷ {AMPLIFICATION_FACTOR}. Year when threshold crossed and sustained (3 of 5 years exceed).
    <strong>Compliance:</strong> Evaluated against Paris Agreement goals (well below 2°C; pursue 1.5°C).
    </div>
    """, unsafe_allow_html=True)

    # Display baseline information (compact)
    if selected_locations:
        baseline_text = ', '.join([f"{loc}: {all_location_results[loc][list(all_location_results[loc].keys())[0]]['baseline']:.1f}°C" if loc in all_location_results and all_location_results[loc] else f"{loc}: N/A" for loc in selected_locations])
        st.markdown(f"""
        <div style='font-size: 0.75em; color: #666; margin: 5px 0;'>
        Baselines ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]}): Global {HARDCODED_GLOBAL_BASELINE:.1f}°C | {baseline_text} | Amplification {AMPLIFICATION_FACTOR}×
        </div>
        """, unsafe_allow_html=True)

    # Display impacts for each location and scenario
    st.markdown("<hr style='margin:10px 0;'><h4 style='margin:5px 0;'>📊 Regional Impacts at 1.5°C Global Warming</h4><div style='font-size:0.75em;color:#666;margin:2px 0;'>Changes relative to 2020 baseline (5-year averages)</div>", unsafe_allow_html=True)

    for location in selected_locations:
        if location not in all_location_results or not all_location_results[location]:
            st.warning(f"No scenarios reach 1.5°C global warming for {location} in the available data range.")
            continue

        st.markdown(f"<h4 style='margin:10px 0 5px 0;'>📍 {location}</h4>", unsafe_allow_html=True)

        location_results = all_location_results[location]

        for scenario, data in sorted(location_results.items(), key=lambda x: x[1]["year"]):
            st.markdown(f"### 🌐 {scenario}")

            regional_temp = data["regional_warming"]
            global_temp = data["global_warming"]
            year = data["year"]

            # Get compliance status
            status_with_emoji, _ = assess_scenario_compliance(scenario, year)

            # Compact header with compliance
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 8px 12px; border-radius: 5px; color: white; margin: 8px 0;'>
                    <div style='font-size: 0.95em; margin: 0;'>
                        <strong>Year {year}</strong> | 
                        Global: {global_temp:.2f}°C | 
                        Regional: {regional_temp:.2f}°C | 
                        {status_with_emoji}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            conditions = data["conditions"]

            # Display metrics by category (Dashboard style) with SWAPPED values
            for category, metrics in DASHBOARD_METRICS.items():
                st.markdown(f"<div style='font-size: 0.9em; font-weight: 600; margin: 8px 0 4px 0;'>{category}</div>", unsafe_allow_html=True)

                # Create columns for metrics in this category
                cols = st.columns(len(metrics))

                for col_idx, (metric_type, metric_name, unit, icon, key) in enumerate(metrics):
                    with cols[col_idx]:
                        if key in conditions and not np.isnan(conditions[key]["change"]):
                            c = conditions[key]

                            # Determine delta color
                            is_inverse = should_use_inverse_delta(metric_type, metric_name)
                            delta_color = "inverse" if is_inverse else "normal"

                            # SWAPPED: Large print = change, small print = value
                            st.metric(
                                f"{icon} {metric_name}",
                                f"{c['change']:+.1f} {c['unit']}",  # Change as main value
                                f"At {c['value']:.1f} {c['unit']}",  # Absolute value as delta
                                delta_color=delta_color
                            )
                        else:
                            st.metric(
                                f"{icon} {metric_name}",
                                "N/A",
                                None
                            )



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
        Actual global temperatures may vary.  Results are most reliable when historical data ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]}) is available.
        """)