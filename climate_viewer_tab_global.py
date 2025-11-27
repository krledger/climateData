# -*- coding: utf-8 -*-
"""
Climate Viewer Global Milestones Tab
=====================================
Last Updated: 2025-11-27 03:45 AEST - Fixed style_horizons_column() to use apply(axis=0) instead of applymap() to avoid TypeError on column styling.
Previous Update: 2025-11-27 02:35 AEST

Shows regional climate impacts at global warming milestones (1.5C, 2.0C, 2.5C).
Structured for compliance with Corporations Act s296D(2B) scenario analysis requirements.

Methodology Notes:
- Uses 20-year centred rolling average per IPCC AR6 methodology for threshold detection
- Column headings are GLOBAL warming thresholds (1.5, 2.0, 2.5 degrees C)
- Regional warming = Global warming x AMPLIFICATION_FACTOR (1.15)
- ALWAYS uses bias-corrected (BC) data for improved accuracy
- Applies post-rolling-average alignment to AGCD at 2014-2015 transition

REVISED APPROACH (2025-11-26):
- Each scenario stands alone (no min/max ranges across scenarios)
- Change calculation is SCENARIO-INTERNAL:
  Î” = scenario's smoothed value at threshold year - scenario's smoothed value at start year
- Uses same smoothing window as Metrics tab for consistency
- User can verify values by checking same year/scenario in Metrics tab

Performance Note:
- Uses df_all from main app with Value_BC column (avoids duplicate loading)
- Only loads AGCD/Historical if not already selected by user
- Timeline visualization reuses pre-computed crossing_years_by_location (no recalculation)

Interactive Features:
- Timeline visualization offers Chart (interactive Altair) and Table (data grid) views
- Users can toggle between views without recalculation
- Both views show crossing year for each scenario at each warming threshold
"""

import os
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from climate_viewer_constants import (
    BASELINE_PERIOD,
    HARDCODED_GLOBAL_BASELINE,
    HARDCODED_AUSTRALIA_BASELINE,
    AMPLIFICATION_FACTOR,
    DASHBOARD_METRICS,
    SCENARIO_DESCRIPTIONS,
    get_scenario_description,
    EMOJI,
)
from climate_viewer_data_operations import (
    calculate_preindustrial_baseline,
    find_metric_name,
    load_metrics_file,
    apply_smoothing,
)

# Global warming thresholds for analysis (degrees C above pre-industrial)
WARMING_THRESHOLDS = [1.5, 2.0, 2.5]


def prepare_bc_data_for_global_tab(
        df_all: pd.DataFrame,
        scen_sel: List[str],
        labels: List[str],
        label_to_path: Dict[str, str]
) -> pd.DataFrame:
    """
    Prepare data with bias correction for global tab analysis.

    Uses Value_BC column from df_all (already loaded by main app) to avoid
    duplicate data loading.  Only loads AGCD/Historical if not already present.

    Optimised to avoid expensive full dataframe copy.

    Args:
        df_all: DataFrame already loaded by main app (contains Value_BC column)
        scen_sel: Selected scenarios
        labels: All available scenario labels
        label_to_path: Mapping of scenario label to file path

    Returns:
        DataFrame with BC values for all required scenarios
    """
    # Start with df_all directly - avoid expensive full copy
    bc_data = df_all

    # If Value_BC exists, use it as the Value column for this tab
    if "Value_BC" in bc_data.columns:
        # Create a small working copy with Value mapped to Value_BC
        bc_data = bc_data.assign(Value=bc_data["Value_BC"])

    scenarios_present = set(bc_data["Scenario"].unique())
    additional_dfs = []

    # Only load AGCD if actually needed (check if in scen_sel first)
    needs_agcd = "AGCD" not in scenarios_present and "AGCD" in label_to_path
    if needs_agcd:
        agcd_df = load_metrics_file(label_to_path["AGCD"], use_bc=True)
        agcd_df["Scenario"] = "AGCD"
        additional_dfs.append(agcd_df)

    # Only load Historical if actually needed
    hist_label = None
    for lbl in labels:
        if "historical" in lbl.lower():
            hist_label = lbl
            break

    needs_hist = hist_label and hist_label not in scenarios_present and hist_label in label_to_path
    if needs_hist:
        hist_df = load_metrics_file(label_to_path[hist_label], use_bc=True)
        hist_df["Scenario"] = hist_label
        additional_dfs.append(hist_df)

    # Combine if we loaded additional data
    if additional_dfs:
        bc_data = pd.concat([bc_data] + additional_dfs, ignore_index=True)

    return bc_data


def get_best_metric_name_for_extraction(data: pd.DataFrame, metric_type: str, metric_name: str) -> str:
    """
    Get best metric name, preferring bias-corrected versions for rain.

    Args:
        data: DataFrame containing metrics
        metric_type: Type of metric (e.g. 'Temp', 'Rain')
        metric_name: Base metric name

    Returns:
        Metric name to use (with BC suffix for rain if available)
    """
    available = data[data["Type"] == metric_type]["Name"].unique()

    if metric_type == "Rain":
        bc_version = f"{metric_name} (BC)"
        if bc_version in available:
            return bc_version

    return find_metric_name(data, metric_type, metric_name)


def find_threshold_crossing_years(
        df: pd.DataFrame,
        scenarios: List[str],
        location: str,
        regional_baseline: float,
        thresholds: List[float],
        amplification_factor: float,
        align_to_agcd: bool = True
) -> Dict[str, Dict[float, Optional[int]]]:
    """
    Find the year each scenario crosses each global warming threshold.

    Thresholds are GLOBAL warming levels (e.g. 1.5C, 2.0C above pre-industrial).
    Regional warming = Global warming x amplification_factor.

    Uses 20-year centred rolling average consistent with IPCC AR6 methodology.
    Validates that crossing is sustained by checking end-state warming.

    Args:
        df: DataFrame with climate data (should include AGCD for alignment)
        scenarios: List of scenario names to analyse (excluding AGCD)
        location: Location name
        regional_baseline: Pre-industrial regional temperature baseline (degrees C)
        thresholds: List of GLOBAL warming thresholds (e.g. [1.5, 2.0, 2.5])
        amplification_factor: Regional amplification factor (e.g. 1.15)
        align_to_agcd: Whether to align rolling averages to AGCD (default True)

    Returns:
        Dict mapping scenario -> {threshold -> year or None}
        Year is the CENTRE of the 20-year period that first exceeds threshold
    """
    results = {}

    # First pass: Calculate rolling averages for ALL scenarios (including AGCD for alignment)
    all_scenarios = list(scenarios)
    if align_to_agcd and "AGCD" not in all_scenarios:
        if "AGCD" in df["Scenario"].unique():
            all_scenarios = ["AGCD"] + all_scenarios

    rolling_data = {}

    for scenario in all_scenarios:
        scenario_data = df[df["Scenario"] == scenario]

        annual_temps = scenario_data[
            (scenario_data["Location"] == location) &
            (scenario_data["Type"] == "Temp") &
            (scenario_data["Name"] == "Average") &
            (scenario_data["Season"] == "Annual")
            ].copy()

        if annual_temps.empty:
            continue

        annual_temps = annual_temps.groupby("Year")["Value"].mean().reset_index()
        annual_temps = annual_temps.sort_values("Year")
        annual_temps["Scenario"] = scenario

        # Calculate regional warming relative to pre-industrial baseline
        annual_temps["Regional_Warming"] = annual_temps["Value"] - regional_baseline

        # Convert to estimated global warming (inverse of amplification)
        annual_temps["Global_Warming"] = annual_temps["Regional_Warming"] / amplification_factor

        # Apply 20-year CENTRED rolling average (IPCC AR6 methodology)
        annual_temps["Rolling_Global_Warming"] = annual_temps["Global_Warming"].rolling(
            window=20, min_periods=10, center=True
        ).mean()

        rolling_data[scenario] = annual_temps

    # Apply alignment to AGCD if requested and AGCD is available
    if align_to_agcd and "AGCD" in rolling_data:
        agcd_temps = rolling_data["AGCD"]
        agcd_at_2014 = agcd_temps[agcd_temps["Year"] == 2014]

        if not agcd_at_2014.empty:
            agcd_rolling_value = agcd_at_2014["Rolling_Global_Warming"].mean()

            if not pd.isna(agcd_rolling_value):
                for scenario in rolling_data:
                    if scenario == "AGCD":
                        continue

                    scen_temps = rolling_data[scenario]
                    is_ssp = scenario.upper().startswith("SSP")
                    scen_align_year = 2015 if is_ssp else 2014

                    scen_at_year = scen_temps[scen_temps["Year"] == scen_align_year]
                    if scen_at_year.empty:
                        continue

                    scen_rolling_value = scen_at_year["Rolling_Global_Warming"].mean()
                    if pd.isna(scen_rolling_value):
                        continue

                    offset = agcd_rolling_value - scen_rolling_value

                    if abs(offset) > 0.001:
                        rolling_data[scenario]["Rolling_Global_Warming"] = \
                            rolling_data[scenario]["Rolling_Global_Warming"] + offset

    # Second pass: Detect threshold crossings using aligned rolling averages
    for scenario in scenarios:
        results[scenario] = {}

        if scenario not in rolling_data:
            for threshold in thresholds:
                results[scenario][threshold] = None
            continue

        annual_temps = rolling_data[scenario]

        # Calculate end-state global warming (last 20 years) for validation
        last_20_years = annual_temps.tail(20)
        end_state_global_warming = None
        if len(last_20_years) >= 10:
            end_state_global_warming = last_20_years["Global_Warming"].mean()

        for threshold in thresholds:
            # Validate: does scenario's end-state actually sustain this global warming level?
            if end_state_global_warming is not None and end_state_global_warming < (threshold - 0.2):
                results[scenario][threshold] = None
                continue

            # Find first year where 20-year rolling average exceeds threshold
            warming_years = annual_temps[annual_temps["Rolling_Global_Warming"] >= threshold]

            if not warming_years.empty:
                results[scenario][threshold] = int(warming_years.iloc[0]["Year"])
            else:
                results[scenario][threshold] = None

    return results


def apply_smoothing_for_global_tab(
        df: pd.DataFrame,
        smooth_window: int
) -> pd.DataFrame:
    """
    Apply smoothing to data for global tab, consistent with Metrics tab.

    Args:
        df: DataFrame with climate data
        smooth_window: Smoothing window size (odd number)

    Returns:
        DataFrame with smoothed values
    """
    if smooth_window <= 1:
        return df

    return apply_smoothing(df, smooth_window)


def get_smoothed_value(
        df: pd.DataFrame,
        metric_type: str,
        metric_name: str,
        scenario: str,
        year: int,
        location: str
) -> Optional[float]:
    """
    Get value at specified year from smoothed data.

    Args:
        df: Smoothed DataFrame
        metric_type: Metric type
        metric_name: Metric name
        scenario: Scenario name
        year: Target year
        location: Location name

    Returns:
        Value at year or None if not found
    """
    subset = df[
        (df["Type"] == metric_type) &
        (df["Name"] == metric_name) &
        (df["Scenario"] == scenario) &
        (df["Location"] == location) &
        (df["Season"] == "Annual") &
        (df["Year"] == year)
        ]

    if not subset.empty:
        return subset["Value"].mean()

    return None


def extract_scenario_conditions_at_threshold(
        df: pd.DataFrame,
        scenario: str,
        location: str,
        threshold_year: int,
        start_year: int,
        metrics_list: List[Tuple[str, str, str, str]]
) -> Dict[str, Dict[str, float]]:
    """
    Extract climate conditions for a single scenario at threshold year.

    Change is calculated WITHIN the scenario:
    Î” = smoothed value at threshold_year - smoothed value at start_year

    Args:
        df: Smoothed DataFrame with climate data
        scenario: Scenario name
        location: Location name
        threshold_year: Year when threshold is crossed
        start_year: Reference/start year for change calculation
        metrics_list: List of (metric_type, metric_name, key, unit) tuples

    Returns:
        Dict mapping metric key -> {value, change, unit, label}
    """
    conditions = {}

    scenario_data = df[
        (df["Scenario"] == scenario) &
        (df["Location"] == location) &
        (df["Season"] == "Annual")
        ]

    if scenario_data.empty:
        return conditions

    for metric_type, metric_name, key, unit in metrics_list:
        actual_metric_name = get_best_metric_name_for_extraction(scenario_data, metric_type, metric_name)

        # Get smoothed value at threshold year
        threshold_value = get_smoothed_value(
            scenario_data, metric_type, actual_metric_name, scenario, threshold_year, location
        )

        # Get smoothed value at start year (same scenario)
        start_value = get_smoothed_value(
            scenario_data, metric_type, actual_metric_name, scenario, start_year, location
        )

        if threshold_value is not None:
            change = threshold_value - start_value if start_value is not None else np.nan

            conditions[key] = {
                'value': threshold_value,
                'change': change,
                'unit': unit,
                'label': actual_metric_name.replace(" (BC)", ""),
                'threshold_year': threshold_year
            }

    return conditions


def build_scenario_impacts(
        df: pd.DataFrame,
        scenarios: List[str],
        location: str,
        crossing_years: Dict[str, Dict[float, Optional[int]]],
        start_year: int,
        thresholds: List[float],
        metrics_list: List[Tuple[str, str, str, str]]
) -> Dict[str, Dict[float, Dict[str, Dict]]]:
    """
    Build per-scenario impact data for each threshold.

    Each scenario is independent - no ranges or cross-scenario comparisons.

    Args:
        df: Smoothed DataFrame with climate data
        scenarios: List of scenario names
        location: Location name
        crossing_years: Dict from find_threshold_crossing_years
        start_year: Reference year for calculating changes
        thresholds: List of warming thresholds
        metrics_list: List of (metric_type, metric_name, key, unit) tuples

    Returns:
        Dict mapping scenario -> {threshold -> {metric_key -> {value, change, unit, label, threshold_year}}}
    """
    results = {}

    for scenario in scenarios:
        results[scenario] = {}

        for threshold in thresholds:
            year = crossing_years.get(scenario, {}).get(threshold)

            if year is not None:
                conditions = extract_scenario_conditions_at_threshold(
                    df, scenario, location, year, start_year, metrics_list
                )
                results[scenario][threshold] = conditions
            else:
                results[scenario][threshold] = None

    return results


def get_horizon_indicator(year: Optional[int], short_start: int, mid_start: int, long_start: int,
                          horizon_end: int) -> str:
    """
    Get visual horizon indicator (emoji) for a year.

    Returns emoji based on which horizon the year falls into.
    """
    if year is None:
        return "⚪"  # Grey circle for N/A

    if short_start <= year < mid_start:
        return "🔵"  # Blue circle
    elif mid_start <= year < long_start:
        return "🟠"  # Orange circle
    elif long_start <= year <= horizon_end:
        return "🔴"  # Red circle
    else:
        return "⚪"  # Grey circle for outside horizons


def get_horizon_cell_color(year: Optional[int], short_start: int, mid_start: int, long_start: int,
                           horizon_end: int) -> str:
    """
    Get background color for table cell based on horizon.

    Returns CSS color string for pandas Styler.
    """
    if year is None:
        return "background-color: #e8e8e8"  # Medium grey for N/A

    if short_start <= year < mid_start:
        return "background-color: rgba(52, 152, 219, 0.2)"  # Blue (#3498DB)
    elif mid_start <= year < long_start:
        return "background-color: rgba(243, 156, 18, 0.2)"  # Orange (#F39C12)
    elif long_start <= year <= horizon_end:
        return "background-color: rgba(231, 76, 60, 0.2)"  # Red (#E74C3C)
    else:
        return "background-color: #e8e8e8"  # Grey for outside horizons


def style_impact_table(df: pd.DataFrame, active_scenarios: List[str],
                       crossing_years: Dict[str, Dict[float, Optional[int]]], threshold: float, short_start: int,
                       mid_start: int, long_start: int, horizon_end: int) -> 'pd.io.formats.style.Styler':
    """
    Apply horizon-based cell background styling to impact dataframe.

    Scenario value columns are shaded based on which time horizon the threshold crossing year falls into.
    Non-scenario columns (Type, Metric, Unit) are not styled.
    Column headers include emoji horizon indicators.

    Args:
        df: Impact dataframe
        active_scenarios: List of active scenario names
        crossing_years: Dict mapping scenario -> {threshold -> year}
        threshold: Current threshold being displayed
        short_start, mid_start, long_start, horizon_end: Horizon boundaries

    Returns:
        Styled dataframe (pd.io.formats.style.Styler)
    """
    # First, don't add emoji indicators - just keep original columns
    df_renamed = df.copy()

    # Build mapping of column names to years for styling
    col_scenario_year = {}
    for col in df_renamed.columns:
        if col in ["Type", "Metric", "Unit"]:
            continue
        # Find matching scenario and its year
        for scenario in active_scenarios:
            if col.startswith(scenario):
                year = crossing_years.get(scenario, {}).get(threshold)
                col_scenario_year[col] = year
                break

    # Function to style a column (receives full column + name)
    def style_column(col: pd.Series) -> List[str]:
        col_name = col.name

        # Non-scenario columns get no styling
        if col_name in ["Type", "Metric", "Unit"]:
            return ["" for _ in col]

        # Get year for this column
        year = col_scenario_year.get(col_name)

        # Return CSS styles for all cells in column
        return [get_horizon_cell_color(year, short_start, mid_start, long_start, horizon_end)
                for _ in col]

    # Apply styling to all columns
    styled = df_renamed.style.apply(style_column, axis=0).hide(axis="index")
    return styled


def format_value(value: Optional[float], is_change: bool = False, decimal_places: int = 1) -> str:
    """
    Format a single value for display.

    Args:
        value: Value to format
        is_change: If True, show sign (+/-)
        decimal_places: Number of decimal places

    Returns:
        Formatted string
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"

    if is_change:
        return f"{value:+.{decimal_places}f}"
    else:
        return f"{value:.{decimal_places}f}"


def generate_global_pdf_report(
        crossing_years_by_location: Dict[str, Dict[str, Dict[float, Optional[int]]]],
        scenario_impacts_by_location: Dict[str, Dict[str, Dict[float, Dict]]],
        selected_locations: List[str],
        global_scenarios: List[str],
        thresholds: List[float],
        amplification_factor: float,
        baseline_period: Tuple[int, int],
        start_year: int,
        short_start: int,
        mid_start: int,
        long_start: int,
        horizon_end: int,
        smooth_window: int
) -> bytes:
    """
    Generate PDF report of global warming threshold analysis.

    Per-scenario layout (not ranges).

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
            self.cell(0, 10, 'Global Warming Threshold Analysis', align='C', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C', new_x='LMARGIN',
                      new_y='NEXT')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

    pdf = PDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Summary section
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Analysis Parameters', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Pre-industrial baseline: {baseline_period[0]}-{baseline_period[1]}', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Start year for changes: {start_year}', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Smoothing window: {smooth_window} years', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Regional amplification factor: {amplification_factor}x', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Locations: {", ".join(selected_locations)}', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    # Planning Horizons (color-coded as visual key)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Planning Horizons - Color Key', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)

    # Header row with colors
    pdf.set_font('Helvetica', 'B', 9)
    horizon_col_widths = [70, 50, 50, 50]

    # Label column (grey background)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(horizon_col_widths[0], 7, '', border=1, align='C', fill=True)

    # Short Term (blue background: RGB 52, 152, 219)
    pdf.set_fill_color(52, 152, 219)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(horizon_col_widths[1], 7, 'Short Term', border=1, align='C', fill=True)

    # Medium Term (orange background: RGB 243, 156, 18)
    pdf.set_fill_color(243, 156, 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(horizon_col_widths[2], 7, 'Medium Term', border=1, align='C', fill=True)

    # Long Term (red background: RGB 231, 76, 60)
    pdf.set_fill_color(231, 76, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(horizon_col_widths[3], 7, 'Long Term', border=1, align='C', fill=True)

    pdf.set_text_color(0, 0, 0)  # Reset to black
    pdf.ln()

    # Data row with colors
    pdf.set_font('Helvetica', '', 9)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(horizon_col_widths[0], 6, 'Dates', border=1, fill=True)

    pdf.set_fill_color(52, 152, 219)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(horizon_col_widths[1], 6, f'{short_start} to {mid_start}', border=1, align='C', fill=True)

    pdf.set_fill_color(243, 156, 18)
    pdf.cell(horizon_col_widths[2], 6, f'{mid_start} to {long_start}', border=1, align='C', fill=True)

    pdf.set_fill_color(231, 76, 60)
    pdf.cell(horizon_col_widths[3], 6, f'{long_start} to {horizon_end}', border=1, align='C', fill=True)

    pdf.set_text_color(0, 0, 0)  # Reset to black
    pdf.ln()

    # Add legend explanation
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, 'This color scheme is used throughout the report to show when threshold crossings occur.',
             new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(0, 0, 0)

    pdf.ln(5)

    # Threshold Crossing Years
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Threshold Crossing Years', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)

    first_location = selected_locations[0]
    crossing_years = crossing_years_by_location[first_location]
    sorted_scenarios = sorted(global_scenarios, key=lambda s: s.upper())

    # Table header
    pdf.set_font('Helvetica', 'B', 9)
    n_thresholds = len(thresholds)
    crossing_col_widths = [70] + [50] * n_thresholds

    pdf.cell(crossing_col_widths[0], 7, 'Scenario', border=1, align='C')
    for i, t in enumerate(thresholds):
        pdf.cell(crossing_col_widths[1 + i], 7, f'{t}\u00b0C', border=1, align='C')
    pdf.ln()

    # Table data
    pdf.set_font('Helvetica', '', 9)
    for scenario in sorted_scenarios:
        pdf.cell(crossing_col_widths[0], 6, scenario, border=1)
        for i, threshold in enumerate(thresholds):
            year = crossing_years.get(scenario, {}).get(threshold)
            year_str = str(year) if year else "---"
            pdf.cell(crossing_col_widths[1 + i], 6, year_str, border=1, align='C')
        pdf.ln()

    pdf.ln(5)

    # Impacts per location and threshold
    for location in selected_locations:
        scenario_impacts = scenario_impacts_by_location[location]
        crossing_years = crossing_years_by_location[location]

        for threshold in thresholds:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, f'{location} - Impacts at {threshold}\u00b0C Global Warming', new_x='LMARGIN', new_y='NEXT')
            pdf.ln(2)

            # Determine which scenarios have data for this threshold
            active_scenarios = [s for s in sorted_scenarios
                                if scenario_impacts.get(s, {}).get(threshold) is not None]

            if not active_scenarios:
                pdf.set_font('Helvetica', 'I', 10)
                pdf.cell(0, 6, 'No scenarios reach this threshold.', new_x='LMARGIN', new_y='NEXT')
                continue

            # Build header: Type | Metric | Unit | Scenario1 Chg | Scenario1 Val | ...
            n_scenarios = len(active_scenarios)
            type_w, metric_w, unit_w = 18, 35, 12
            scen_chg_w, scen_val_w = 22, 22

            # Helper to get horizon color for a scenario/threshold
            def get_pdf_horizon_color(year):
                """Returns (R, G, B) tuple for scenario's threshold crossing year."""
                if year is None:
                    return (200, 200, 200)  # Grey
                if short_start <= year < mid_start:
                    return (52, 152, 219)  # Blue
                elif mid_start <= year < long_start:
                    return (243, 156, 18)  # Orange
                elif long_start <= year <= horizon_end:
                    return (231, 76, 60)  # Red
                else:
                    return (200, 200, 200)  # Grey

            # Header row (metric labels - no fill)
            pdf.set_font('Helvetica', 'B', 7)
            pdf.cell(type_w, 7, 'Type', border=1, align='C')
            pdf.cell(metric_w, 7, 'Metric', border=1, align='C')
            pdf.cell(unit_w, 7, 'Unit', border=1, align='C')

            # Scenario column headers with horizon colors
            for scenario in active_scenarios:
                year = crossing_years.get(scenario, {}).get(threshold)
                r, g, b = get_pdf_horizon_color(year)
                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(255, 255, 255) if (r, g, b) != (200, 200, 200) else pdf.set_text_color(0, 0, 0)
                # Two columns per scenario: Change and Value
                pdf.cell(scen_chg_w, 7, f'{scenario[:7]} Ch', border=1, align='C', fill=True)
                pdf.cell(scen_val_w, 7, f'{scenario[:7]} Val', border=1, align='C', fill=True)

            pdf.set_text_color(0, 0, 0)  # Reset text color
            pdf.ln()

            # Metric rows
            pdf.set_font('Helvetica', '', 7)

            metric_display = [
                ('temp_avg', 'Temp', 'Average'),
                ('temp_max', 'Temp', '5-Day Avg Max'),
                ('temp_heat_days', 'Temp', 'Days >37\u00b0C 3hr'),
                ('rain_total', 'Rain', 'Total'),
                ('rain_max', 'Rain', 'Max 5-Day'),
                ('rain_cdd', 'Rain', 'CDD'),
                ('wind_avg', 'Wind', 'Average'),
                ('wind_max', 'Wind', '95th Percentile'),
                ('humidity_avg', 'Humidity', 'Average'),
                ('vpd_avg', 'VPD', 'Average'),
            ]

            def get_light_horizon_color(year):
                """Returns light (R, G, B) tuple for metric cell fills."""
                if year is None:
                    return (240, 240, 240)  # Light grey
                if short_start <= year < mid_start:
                    return (173, 216, 230)  # Light blue
                elif mid_start <= year < long_start:
                    return (255, 218, 125)  # Light orange
                elif long_start <= year <= horizon_end:
                    return (239, 154, 154)  # Light red
                else:
                    return (240, 240, 240)  # Light grey

            for key, metric_type, display_name in metric_display:
                # Find unit from first scenario that has this metric
                unit = ""
                for scenario in active_scenarios:
                    conditions = scenario_impacts.get(scenario, {}).get(threshold, {})
                    if key in conditions:
                        unit = conditions[key].get('unit', '')
                        break

                # Label cells (no fill)
                pdf.cell(type_w, 5, metric_type, border=1)
                pdf.cell(metric_w, 5, display_name[:20], border=1)
                pdf.cell(unit_w, 5, unit, border=1, align='C')

                # Data cells with light horizon colors
                for scenario in active_scenarios:
                    year = crossing_years.get(scenario, {}).get(threshold)
                    r, g, b = get_light_horizon_color(year)
                    pdf.set_fill_color(r, g, b)

                    conditions = scenario_impacts.get(scenario, {}).get(threshold, {})
                    if key in conditions:
                        change = conditions[key].get('change')
                        value = conditions[key].get('value')

                        pdf.cell(scen_chg_w, 5, format_value(change, is_change=True), border=1, align='C', fill=True)
                        pdf.cell(scen_val_w, 5, format_value(value), border=1, align='C', fill=True)
                    else:
                        pdf.cell(scen_chg_w, 5, '---', border=1, align='C', fill=True)
                        pdf.cell(scen_val_w, 5, '---', border=1, align='C', fill=True)
                pdf.ln()

            pdf.ln(3)
            pdf.set_font('Helvetica', 'I', 8)
            pdf.cell(0, 5,
                     f'Changes calculated from each scenario\'s own {start_year} value (smoothed with {smooth_window}-year window).',
                     new_x='LMARGIN', new_y='NEXT')

    # =========================================================================
    # Legend Page — Color Scheme Explanation
    # =========================================================================
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 12, 'Color Scheme Legend', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Time Horizons', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(2)

    # Blue horizon
    pdf.set_fill_color(52, 152, 219)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(20, 8, '', border=1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f'  Short-term ({short_start}-{mid_start}): Near-future projections',
             new_x='LMARGIN', new_y='NEXT')

    # Orange horizon
    pdf.set_fill_color(243, 156, 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(20, 8, '', border=1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f'  Medium-term ({mid_start}-{long_start}): Mid-century projections',
             new_x='LMARGIN', new_y='NEXT')

    # Red horizon
    pdf.set_fill_color(231, 76, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(20, 8, '', border=1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f'  Long-term ({long_start}-{horizon_end}): End-of-century projections',
             new_x='LMARGIN', new_y='NEXT')

    # Grey
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(20, 8, '', border=1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, '  Outside Planning Horizons: Crossing not within defined periods',
             new_x='LMARGIN', new_y='NEXT')

    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'How to Use This Report', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(2)

    legend_text = [
        "1.  Planning Horizons Table: Shows the date ranges for each time horizon using this color scheme.",
        "",
        "2.  Impact Tables: Table headers and cells are color-coded by planning horizon to indicate when",
        "    each scenario crosses the global warming threshold.",
        "",
        "3.  Column Colors: Scenario column headers indicate which time horizon that scenario reaches the",
        "    threshold.  Cell colors use lighter shades of the same colors for easier reading.",
        "",
        "4.  Interpretation: Thresholds crossed in short-term (blue) require immediate adaptation.",
        "    Medium-term (orange) and long-term (red) thresholds allow for more gradual planning.",
        "",
        "5.  Scenario Comparison: Compare colors across columns to see if different scenarios reach",
        "    the same threshold in different planning horizons."
    ]

    for line in legend_text:
        pdf.set_font('Helvetica', '', 10)
        if line:
            pdf.multi_cell(0, 5, line, new_x='LMARGIN', new_y='NEXT')
        else:
            pdf.ln(3)

    return bytes(pdf.output())


def build_timeline_data(
        crossing_years: Dict[str, Dict[float, Optional[int]]],
        thresholds: List[float]
) -> pd.DataFrame:
    """
    Build timeline data from threshold crossing years for Altair visualization.

    Format: Y-axis = Scenario, X-axis = Year, points colored by threshold.
    Reuses pre-computed crossing_years - no recalculation.

    Args:
        crossing_years: Dict mapping scenario -> {threshold -> year}
        thresholds: List of warming thresholds

    Returns:
        DataFrame with columns: Scenario, Year, Threshold, Threshold_num, Year_label
    """
    timeline_records = []

    for scenario, years_dict in crossing_years.items():
        for threshold in thresholds:
            year = years_dict.get(threshold)

            if year is not None:
                timeline_records.append({
                    'Scenario': scenario,
                    'Year': int(year),
                    'Threshold': f"{threshold}°C",
                    'Threshold_num': threshold,
                    'Year_label': str(year)
                })

    if not timeline_records:
        return pd.DataFrame()

    return pd.DataFrame(timeline_records)


def build_timeline_table_data(
        crossing_years: Dict[str, Dict[float, Optional[int]]],
        thresholds: List[float]
) -> pd.DataFrame:
    """
    Build a table DataFrame from crossing years data.

    Format matches the Threshold Crossing Years table above.

    Args:
        crossing_years: Dict mapping scenario -> {threshold -> year}
        thresholds: List of warming thresholds

    Returns:
        DataFrame with Scenario column and one column per threshold
    """
    rows = []

    sorted_scenarios = sorted(crossing_years.keys(), key=lambda s: s.upper())

    for scenario in sorted_scenarios:
        row = {"Scenario": scenario}
        for threshold in thresholds:
            year = crossing_years.get(scenario, {}).get(threshold)
            row[f"{threshold}°C"] = str(year) if year is not None else "---"
        rows.append(row)

    return pd.DataFrame(rows)


def render_timeline_visualization(
        st,
        crossing_years_by_location: Dict[str, Dict[str, Dict[float, Optional[int]]]],
        thresholds: List[float],
        location: str,
        y0: int,
        y1: int,
        short_start: int,
        mid_start: int,
        long_start: int,
        horizon_end: int
):
    """
    Render an Altair timeline showing threshold crossing years with chart/table toggle.

    Years on X-axis, Scenarios on Y-axis.  Points colored by warming threshold.
    Background shading shows time horizons (like Metrics tab).

    Reuses pre-computed crossing_years_by_location - no recalculation.

    Args:
        st: Streamlit module
        crossing_years_by_location: Pre-computed crossing years by location
        thresholds: List of warming thresholds
        location: Current location
        y0: Start year from sidebar
        y1: End year from sidebar
        short_start: Short-term horizon start year
        mid_start: Medium-term horizon start year
        long_start: Long-term horizon start year
        horizon_end: Planning horizon end year
    """
    crossing_years = crossing_years_by_location.get(location, {})

    if not crossing_years:
        st.info(f"No threshold crossing data available for {location}")
        return

    # Build both datasets once (chart will use timeline_df, table will use table_data)
    timeline_df = build_timeline_data(crossing_years, thresholds)
    table_data = build_timeline_table_data(crossing_years, thresholds)

    if timeline_df.empty:
        st.info(f"No scenarios cross thresholds at {location}")
        return

    # Toggle between chart and table views
    show_chart = st.radio(
        "Display",
        options=["Chart", "Table"],
        horizontal=True,
        key=f"timeline_view_{location}"
    )

    if show_chart == "Chart":
        # CHART VIEW - Years on X-axis, Scenarios on Y-axis
        chart_layers = []

        # Add horizon shading bands (like Metrics tab)
        if short_start < mid_start and short_start >= y0 and mid_start <= y1:
            short_rect = alt.Chart(pd.DataFrame({
                'start': [max(short_start, y0)],
                'end': [min(mid_start, y1)]
            })).mark_rect(opacity=0.15, color='#3498DB').encode(
                x=alt.X('start:Q', scale=alt.Scale(domain=[y0, y1])),
                x2=alt.X2('end:Q')
            )
            chart_layers.append(short_rect)

        if mid_start < long_start and mid_start >= y0 and long_start <= y1:
            mid_rect = alt.Chart(pd.DataFrame({
                'start': [max(mid_start, y0)],
                'end': [min(long_start, y1)]
            })).mark_rect(opacity=0.15, color='#F39C12').encode(
                x=alt.X('start:Q', scale=alt.Scale(domain=[y0, y1])),
                x2=alt.X2('end:Q')
            )
            chart_layers.append(mid_rect)

        if long_start < horizon_end and long_start >= y0 and horizon_end <= y1:
            long_rect = alt.Chart(pd.DataFrame({
                'start': [max(long_start, y0)],
                'end': [min(horizon_end, y1)]
            })).mark_rect(opacity=0.15, color='#E74C3C').encode(
                x=alt.X('start:Q', scale=alt.Scale(domain=[y0, y1])),
                x2=alt.X2('end:Q')
            )
            chart_layers.append(long_rect)

        # Point markers colored by threshold
        points = alt.Chart(timeline_df).mark_circle(size=200, opacity=0.85).encode(
            x=alt.X(
                'Year:Q',
                scale=alt.Scale(domain=[y0, y1]),
                axis=alt.Axis(format='d', labelFontSize=11, titleFontSize=12)
            ),
            y=alt.Y(
                'Scenario:N',
                title='Scenario',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)
            ),
            color=alt.Color(
                'Threshold:N',
                title='Global Warming Threshold',
                scale=alt.Scale(
                    domain=[f"{t}°C" for t in sorted(thresholds)],
                    range=['#2ECC71', '#F39C12', '#E74C3C']  # Green, Orange, Red
                ),
                legend=alt.Legend(labelFontSize=10, titleFontSize=11)
            ),
            tooltip=[
                alt.Tooltip('Scenario:N', title='Scenario'),
                alt.Tooltip('Threshold:N', title='Threshold'),
                alt.Tooltip('Year:Q', title='Year', format='d'),
            ]
        )

        # Year labels above dots
        text_labels = alt.Chart(timeline_df).mark_text(
            align='center',
            baseline='bottom',
            dy=-10,
            fontSize=13,
            color='black'
        ).encode(
            x='Year:Q',
            y='Scenario:N',
            text='Year_label:N'
        )

        # Combine all layers: shading (back) + points + text (front)
        chart_layers.append(points)
        chart_layers.append(text_labels)

        chart = alt.layer(*chart_layers).properties(
            height=300,
            width=900
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    else:
        # TABLE VIEW - use pre-built table_data from above
        st.dataframe(
            table_data,
            column_config={
                "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
                **{f"{t}°C": st.column_config.TextColumn(f"{t}°C", width="small")
                   for t in thresholds}
            },
            hide_index=True,
            width='stretch'
        )


def render_global_tab(
        st,
        scen_sel: List[str],
        loc_sel: List[str],
        df_all: pd.DataFrame,
        labels: List[str],
        label_to_path: Dict[str, str],
        load_metrics_func,
        y0: int,
        y1: int,
        short_start: int,
        mid_start: int,
        long_start: int,
        horizon_end: int,
        smooth: bool = False,
        smooth_win: int = 9
):
    """
    Render the global milestones tab.

    Args:
        st: Streamlit module
        scen_sel: Selected scenarios
        loc_sel: Selected locations
        df_all: All loaded data (used directly with Value_BC column)
        labels: All scenario labels
        label_to_path: Mapping of label to file path
        load_metrics_func: Function to load metrics (used only if AGCD/Historical missing)
        y0: Start year from sidebar
        y1: End year from sidebar
        short_start: Short-term horizon start year
        mid_start: Medium-term horizon start year
        long_start: Long-term horizon start year
        horizon_end: Planning horizon end year
        smooth: Whether smoothing is enabled
        smooth_win: Smoothing window size
    """
    st.title(f"{EMOJI['globe']} Global Milestones")

    # Ensure minimum smoothing window of 5 for meaningful results
    effective_smooth_win = smooth_win if smooth else 9
    if effective_smooth_win < 5:
        effective_smooth_win = 9

    st.markdown(f"""
    This analysis shows regional climate impacts at key global warming thresholds 
    (1.5\u00b0C, 2.0\u00b0C, 2.5\u00b0C above pre-industrial levels).

    **Regional amplification:** Land areas warm approximately {AMPLIFICATION_FACTOR} times as much as the global average.  
    When global temperature rises 1.5\u00b0C, regional warming is approximately {1.5 * AMPLIFICATION_FACTOR:.1f}\u00b0C.

    **Each scenario stands alone:** Changes are calculated from each scenario's own start year value.  
    Values use {effective_smooth_win}-year smoothing to match the Metrics tab.
    """)

    # Filter to non-historical scenarios
    global_scenarios = [s for s in scen_sel if "historical" not in s.lower() and "agcd" not in s.lower()]

    if not global_scenarios:
        st.warning(
            f"{EMOJI['warning']} Please select at least one future scenario (excluding historical) to view threshold analysis.")
        st.stop()

    selected_locations = loc_sel

    if not selected_locations:
        st.info("Please select at least one location in the sidebar to view impacts.")
        st.stop()

    # Start year for calculating changes
    # Minimum of 2020 ensures SSP data availability
    start_year = max(short_start, 2020)

    # Pre-compute sorted scenarios (used multiple times below)
    sorted_scenarios = sorted(global_scenarios, key=lambda s: s.upper())

    # Prepare metrics list (used for building impacts tables)
    metrics_list = []
    for category, metrics in DASHBOARD_METRICS.items():
        for metric_type, name, unit, icon, key in metrics:
            if metric_type == "Temp" and name == "Max Day":
                metrics_list.append((metric_type, "5-Day Avg Max", key, unit))
            elif metric_type == "Rain" and name == "Max Day":
                metrics_list.append((metric_type, "Max 5-Day", key, unit))
            elif metric_type == "Wind" and name == "Max Day":
                metrics_list.append((metric_type, "95th Percentile", key, unit))
            else:
                metrics_list.append((metric_type, name, key, unit))

    # Pre-compute metric_display list for impact tables (used multiple times)
    metric_display = [
        ('temp_avg', 'Temp', 'Average'),
        ('temp_max', 'Temp', '5-Day Avg Max'),
        ('temp_heat_days', 'Temp', 'Days >37\u00b0C 3hr'),
        ('rain_total', 'Rain', 'Total'),
        ('rain_max', 'Rain', 'Max 5-Day'),
        ('rain_cdd', 'Rain', 'CDD'),
        ('wind_avg', 'Wind', 'Average'),
        ('wind_max', 'Wind', '95th Percentile'),
        ('humidity_avg', 'Humidity', 'Average'),
        ('vpd_avg', 'VPD', 'Average'),
    ]

    # Prepare BC data (cache to avoid recalculation on radio toggle)
    bc_cache_key = f"bc_data_{tuple(sorted(scen_sel))}"
    if bc_cache_key not in st.session_state:
        with st.spinner("Preparing bias-corrected data..."):
            bc_data = prepare_bc_data_for_global_tab(df_all, scen_sel, labels, label_to_path)
        st.session_state[bc_cache_key] = bc_data
    else:
        bc_data = st.session_state[bc_cache_key]

    # Apply smoothing (cache to avoid recalculation on radio toggle)
    smooth_cache_key = f"smoothed_data_{tuple(sorted(scen_sel))}_{effective_smooth_win}"
    if smooth_cache_key not in st.session_state:
        with st.spinner("Applying smoothing..."):
            smoothed_data = apply_smoothing_for_global_tab(bc_data, effective_smooth_win)
        st.session_state[smooth_cache_key] = smoothed_data
    else:
        smoothed_data = st.session_state[smooth_cache_key]

    # Calculate results for all locations (cache in session state to avoid recalc on radio toggle)
    cache_key = f"global_tab_results_{tuple(sorted(scen_sel))}_{tuple(sorted(loc_sel))}_{effective_smooth_win}"

    if cache_key not in st.session_state:
        with st.spinner("Analysing warming thresholds..."):
            crossing_years_by_location = {}
            scenario_impacts_by_location = {}

            for location in selected_locations:
                # Calculate regional pre-industrial baseline
                historical_data = bc_data[bc_data["Scenario"].str.lower().str.contains("historical", na=False)]

                if not historical_data.empty:
                    regional_baseline = calculate_preindustrial_baseline(
                        historical_data, BASELINE_PERIOD, location
                    )
                else:
                    regional_baseline = np.nan

                if np.isnan(regional_baseline):
                    if location == "Australia" or "Australia" in location:
                        regional_baseline = HARDCODED_AUSTRALIA_BASELINE
                    else:
                        regional_baseline = HARDCODED_GLOBAL_BASELINE + 7.8

                # Find crossing years (uses 20-year rolling for threshold detection - IPCC standard)
                crossing_years = find_threshold_crossing_years(
                    bc_data, global_scenarios, location, regional_baseline,
                    WARMING_THRESHOLDS, AMPLIFICATION_FACTOR, align_to_agcd=True
                )
                crossing_years_by_location[location] = crossing_years

                # Build per-scenario impacts using smoothed data
                scenario_impacts = build_scenario_impacts(
                    smoothed_data, global_scenarios, location, crossing_years,
                    start_year, WARMING_THRESHOLDS, metrics_list
                )
                scenario_impacts_by_location[location] = scenario_impacts

        # Store results in session state
        st.session_state[cache_key] = {
            'crossing_years_by_location': crossing_years_by_location,
            'scenario_impacts_by_location': scenario_impacts_by_location
        }
    else:
        # Retrieve from session state cache
        crossing_years_by_location = st.session_state[cache_key]['crossing_years_by_location']
        scenario_impacts_by_location = st.session_state[cache_key]['scenario_impacts_by_location']

    # =========================================================================
    # Planning Horizons Table (color-coded as visual key for entire tab)
    # =========================================================================
    st.markdown("---")
    st.subheader(f"{EMOJI['calendar']} Planning Horizons - Color Key")

    horizons_data = [{
        "": "Dates",
        "Short Term": f"{short_start} to {mid_start}",
        "Medium Term": f"{mid_start} to {long_start}",
        "Long Term": f"{long_start} to {horizon_end}"
    }]
    horizons_df = pd.DataFrame(horizons_data)

    # Apply color styling to horizon columns based on column name
    def style_horizons_column(col):
        """Style entire column based on its name (horizon type)."""
        col_name = col.name

        if col_name == "Short Term":
            style_str = "background-color: rgba(52, 152, 219, 0.3);"  # Blue (no bold for data)
        elif col_name == "Medium Term":
            style_str = "background-color: rgba(243, 156, 18, 0.3);"  # Orange (no bold for data)
        elif col_name == "Long Term":
            style_str = "background-color: rgba(231, 76, 60, 0.3);"  # Red (no bold for data)
        else:
            style_str = "background-color: #f5f5f5;"  # Grey (no bold for data)

        # Return list of styles (one per cell in column)
        return [style_str for _ in col]

    styled_horizons = horizons_df.style.apply(style_horizons_column, axis=0).hide(axis="index").to_html(escape=False)
    st.markdown(styled_horizons, unsafe_allow_html=True)

    st.markdown("")

    # =========================================================================
    # Timeline Visualization (reuses pre-computed crossing_years_by_location)
    # =========================================================================
    st.markdown("---")
    st.subheader(f"{EMOJI['calendar']} Change from Preindustrial Timeline")

    # Display timeline for first location
    first_location = selected_locations[0]
    render_timeline_visualization(
        st,
        crossing_years_by_location,
        WARMING_THRESHOLDS,
        first_location,
        y0=y0,
        y1=y1,
        short_start=short_start,
        mid_start=mid_start,
        long_start=long_start,
        horizon_end=horizon_end
    )

    # =========================================================================
    # Impacts at Thresholds (per location, per threshold, per scenario)
    # =========================================================================
    st.markdown("---")

    for location in selected_locations:
        st.subheader(f"{EMOJI['pin']} Impacts at Global Warming Thresholds \u2014 {location}")

        st.markdown("""
        *Scenario analysis for: (a) increase in global average temperature well exceeding 
        2\u00b0C above pre-industrial levels; (b) increase limited to 1.5\u00b0C above pre-industrial levels*
        """)

        scenario_impacts = scenario_impacts_by_location[location]
        crossing_years = crossing_years_by_location[location]

        # Create a tab for each threshold
        threshold_tabs = st.tabs([f"{t}\u00b0C Global Warming" for t in WARMING_THRESHOLDS])

        for tab_idx, threshold in enumerate(WARMING_THRESHOLDS):
            with threshold_tabs[tab_idx]:
                # Determine which scenarios reach this threshold
                active_scenarios = [s for s in sorted_scenarios
                                    if scenario_impacts.get(s, {}).get(threshold) is not None]

                if not active_scenarios:
                    st.info(f"No selected scenarios reach {threshold}\u00b0C global warming.")
                    continue

                # Build table using pre-computed metric_display

                impact_data = []

                for key, metric_type, display_name in metric_display:
                    # Find unit from first scenario that has this metric
                    unit = ""
                    for scenario in active_scenarios:
                        conditions = scenario_impacts.get(scenario, {}).get(threshold, {})
                        if key in conditions:
                            unit = conditions[key].get('unit', '')
                            break

                    if not unit:
                        continue

                    row = {"Type": metric_type, "Metric": display_name, "Unit": unit}

                    for scenario in active_scenarios:
                        conditions = scenario_impacts.get(scenario, {}).get(threshold, {})
                        year = crossing_years.get(scenario, {}).get(threshold)

                        if key in conditions:
                            change = conditions[key].get('change')
                            value = conditions[key].get('value')

                            row[f"{scenario} ({year}) \u0394"] = format_value(change, is_change=True)
                            row[f"{scenario} Value"] = format_value(value)
                        else:
                            row[f"{scenario} ({year}) \u0394"] = "---"
                            row[f"{scenario} Value"] = "---"

                    impact_data.append(row)

                if impact_data:
                    impact_df = pd.DataFrame(impact_data)

                    # Apply horizon-based cell shading and display
                    styled_df = style_impact_table(impact_df, active_scenarios, crossing_years, threshold, short_start,
                                                   mid_start, long_start, horizon_end)
                    st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

                    st.markdown("")

                    st.caption(f"""
                    \u0394 = Change from scenario's own {start_year} value.  
                    Values use {effective_smooth_win}-year smoothing (same as Metrics tab).
                    """)
                else:
                    st.warning("No impact data available.")

        if location != selected_locations[-1]:
            st.markdown("---")

    # =========================================================================
    # PDF Export
    # =========================================================================
    st.markdown("---")

    pdf_bytes = generate_global_pdf_report(
        crossing_years_by_location,
        scenario_impacts_by_location,
        selected_locations,
        global_scenarios,
        WARMING_THRESHOLDS,
        AMPLIFICATION_FACTOR,
        BASELINE_PERIOD,
        start_year,
        short_start,
        mid_start,
        long_start,
        horizon_end,
        effective_smooth_win
    )

    if pdf_bytes:
        st.download_button(
            label=f"{EMOJI['page']} Download PDF Report",
            data=pdf_bytes,
            file_name=f"warming_thresholds_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            help="Download complete analysis as PDF"
        )
    else:
        st.info(f"{EMOJI['info']} Install fpdf2 for PDF export: `pip install fpdf2`")

    # =========================================================================
    # Methodology Notes
    # =========================================================================
    with st.expander(f"{EMOJI['info']} About This Analysis"):
        st.markdown(f"""
        ### Corporations Act s296D(2B) Compliance

        This analysis satisfies the scenario analysis requirements of the Corporations Act:

        - **(a)** Scenarios where global temperature **well exceeds 2\u00b0C**: Assessed via 2.0\u00b0C and 2.5\u00b0C threshold impacts
        - **(b)** Scenario where warming is **limited to 1.5\u00b0C**: Assessed via 1.5\u00b0C threshold impacts

        ### Methodology

        **Each Scenario Stands Alone:**
        - SSP1-26 and SSP3-70 (etc.) are separate stress tests, not bounds on a single estimate
        - Change is calculated WITHIN each scenario:
          - \u0394 = Scenario's value at threshold year \u2212 Scenario's value at {start_year}
        - No cross-scenario ranges or comparisons

        **Global vs Regional Temperature:**
        - Column headings (1.5\u00b0C, 2.0\u00b0C, 2.5\u00b0C) are **global** warming thresholds
        - Regional warming = Global warming \u00d7 {AMPLIFICATION_FACTOR} amplification factor
        - E.g., 1.5\u00b0C global warming corresponds to {1.5 * AMPLIFICATION_FACTOR:.2f}\u00b0C regional warming

        **Threshold Crossing Detection (IPCC AR6 methodology):**
        - 20-year centred rolling average applied to regional temperature
        - Crossing year is the centre year when rolling average first exceeds global threshold
        - End-state validation ensures threshold is sustained

        **Impact Values:**
        - Use {effective_smooth_win}-year smoothing (same as Metrics tab)
        - Values will match if you look up the same year/scenario in Metrics tab
        - Adjust smoothing in sidebar to see sensitivity to averaging period

        **Bias Correction:**
        - All data uses bias-corrected (BC) values
        - Scenarios aligned to AGCD at 2014-2015 transition

        ### Pre-industrial Baseline

        - **Period:** {BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]} (IPCC AR6 standard)
        - **Global mean:** {HARDCODED_GLOBAL_BASELINE}\u00b0C
        - **Regional amplification:** {AMPLIFICATION_FACTOR}x

        ### Why Scenarios Are Independent

        The Paris Agreement / Corporations Act requires assessing resilience under DIFFERENT futures:
        - SSP1-26: "What if we follow a low-emissions pathway?"
        - SSP3-70: "What if we follow a high-emissions pathway?"

        These are separate stress tests.  The actual future will follow ONE pathway (or something else),
        not a blend.  Presenting ranges across scenarios can misleadingly imply the outcome will fall
        "somewhere in between" when scenarios are fundamentally different futures.
        """)