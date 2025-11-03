"""
Climate Viewer Utilities
Shared utility functions used across multiple modules.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


def get_5year_average(
    data: pd.DataFrame,
    metric_type: str,
    metric_name: str,
    scenario: str,
    center_year: int,
    location: Optional[str] = None
) -> Optional[float]:
    """
    Get 5-year average centered on target year.
    
    Args:
        data: DataFrame with climate data
        metric_type: Type of metric (e.g., 'Temp', 'Rain', 'Wind')
        metric_name: Name of metric (e.g., 'Average', 'Total')
        scenario: Scenario name
        center_year: Center year for 5-year window
        location: Optional location filter
        
    Returns:
        Mean value or None if no data available
    """
    start_year = center_year - 2
    end_year = center_year + 2
    
    # Try exact match first
    subset = data[
        (data["Type"] == metric_type) &
        (data["Name"] == metric_name) &
        (data["Scenario"] == scenario) &
        (data["Year"].between(start_year, end_year))
    ]
    
    if location:
        subset = subset[subset["Location"] == location]
    
    if not subset.empty:
        return subset["Value"].mean()
    
    # Fallback: try partial name match
    subset = data[
        (data["Type"] == metric_type) &
        (data["Name"].str.contains(metric_name.split()[0], case=False, na=False)) &
        (data["Scenario"] == scenario) &
        (data["Year"].between(start_year, end_year))
    ]
    
    if location:
        subset = subset[subset["Location"] == location]
    
    if not subset.empty:
        return subset["Value"].mean()
    
    # Fallback: single year
    single_year = data[
        (data["Type"] == metric_type) &
        (data["Name"].str.contains(metric_name.split()[0], case=False, na=False)) &
        (data["Scenario"] == scenario) &
        (data["Year"] == center_year)
    ]
    
    if location:
        single_year = single_year[single_year["Location"] == location]
    
    if not single_year.empty:
        return single_year["Value"].mean()
    
    # Fallback: nearby years (within 5 years)
    nearby = data[
        (data["Type"] == metric_type) &
        (data["Name"].str.contains(metric_name.split()[0], case=False, na=False)) &
        (data["Scenario"] == scenario) &
        (data["Year"].between(center_year - 5, center_year + 5))
    ]
    
    if location:
        nearby = nearby[nearby["Location"] == location]
    
    if not nearby.empty:
        nearby = nearby.copy()
        nearby["distance"] = abs(nearby["Year"] - center_year)
        closest = nearby.nsmallest(min(5, len(nearby)), "distance")
        return closest["Value"].mean()
    
    return None


def find_metric_name(
    data: pd.DataFrame,
    metric_type: str,
    preferred_name: str
) -> str:
    """
    Find actual metric name in data, handling common variations.
    
    Args:
        data: DataFrame with climate data
        metric_type: Type of metric (e.g., 'Temp', 'Rain')
        preferred_name: Preferred/expected metric name
        
    Returns:
        Actual metric name found in data, or preferred_name if not found
    """
    available = data[data["Type"] == metric_type]["Name"].unique()
    
    # Exact match
    if preferred_name in available:
        return preferred_name
    
    # Handle Days variations (Days>37, Days>=37, etc.)
    if "Days" in preferred_name or "days" in preferred_name:
        variations = [
            preferred_name,
            preferred_name.replace(">=", ">"),
            preferred_name.replace(">", ">="),
            preferred_name.replace(">", " >"),
            preferred_name.replace(">=", " >="),
            preferred_name.replace(" ", ""),
            preferred_name.replace("Days", "Days "),
            preferred_name.replace("Tx", "Tx "),
        ]
        
        for var in variations:
            if var in available:
                return var
        
        # Fuzzy match for temperature thresholds
        preferred_lower = preferred_name.lower().replace(" ", "")
        for avail in available:
            avail_lower = avail.lower().replace(" ", "")
            if "37" in preferred_lower and "37" in avail_lower and (
                "day" in avail_lower or "tx" in avail_lower
            ):
                return avail
    
    # Try partial match
    for avail in available:
        if preferred_name.lower() in avail.lower() or avail.lower() in preferred_name.lower():
            return avail
    
    return preferred_name


def calculate_metric_change(
    data: pd.DataFrame,
    metric_type: str,
    metric_name: str,
    scenario: str,
    from_year: int,
    to_year: int,
    location: Optional[str] = None,
    baseline_dict: Optional[Dict] = None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate change in metric from baseline to target year.
    
    Args:
        data: DataFrame with climate data
        metric_type: Type of metric
        metric_name: Name of metric
        scenario: Scenario name
        from_year: Baseline year
        to_year: Target year
        location: Optional location filter
        baseline_dict: Optional dict of pre-industrial baselines {(type, name): value}
        
    Returns:
        Tuple of (change_from_baseline, change_from_preindustrial, target_value)
    """
    baseline_val = get_5year_average(data, metric_type, metric_name, scenario, from_year, location)
    target_val = get_5year_average(data, metric_type, metric_name, scenario, to_year, location)
    
    change_from_start = None
    change_from_preindustrial = None
    
    if baseline_val is not None and target_val is not None:
        change_from_start = target_val - baseline_val
    
    # Calculate change from pre-industrial if baseline provided
    if target_val is not None and baseline_dict:
        key = (metric_type, metric_name)
        if key in baseline_dict:
            change_from_preindustrial = target_val - baseline_dict[key]
    
    return change_from_start, change_from_preindustrial, target_val


def extract_conditions_at_year(
    df: pd.DataFrame,
    scenario: str,
    location: str,
    target_year: int,
    reference_year: int,
    metrics_list: list
) -> Dict[str, Dict[str, float]]:
    """
    Extract climate conditions at target year vs reference year.
    
    Args:
        df: DataFrame with climate data
        scenario: Scenario name
        location: Location name
        target_year: Year to extract conditions for
        reference_year: Reference year for comparison
        metrics_list: List of tuples (metric_type, name, key, unit)
        
    Returns:
        Dictionary of {key: {value, change, unit, label}}
    """
    conditions = {}
    
    # Filter for this scenario, location, and annual data only
    scenario_data = df[
        (df["Scenario"] == scenario) &
        (df["Location"] == location) &
        (df["Season"] == "Annual")
    ]
    
    if scenario_data.empty:
        return conditions
    
    for metric_type, name, key, unit in metrics_list:
        # Find actual metric name in data (handles variations like Days>=37 vs Days>37)
        actual_metric_name = find_metric_name(scenario_data, metric_type, name)
        
        # Get 5-year averages
        target_val = get_5year_average(
            scenario_data, metric_type, actual_metric_name, scenario, target_year
        )
        ref_val = get_5year_average(
            scenario_data, metric_type, actual_metric_name, scenario, reference_year
        )
        
        if target_val is not None and ref_val is not None:
            change = target_val - ref_val
            
            # Only store if we haven't already stored this key
            if key not in conditions:
                conditions[key] = {
                    "value": target_val,
                    "change": change,
                    "unit": unit,
                    "label": name
                }
    
    return conditions


def calculate_preindustrial_baseline(
    df: pd.DataFrame,
    baseline_period: Tuple[int, int],
    location: Optional[str] = None
) -> float:
    """
    Calculate pre-industrial baseline temperature (1850-1900).
    
    Args:
        df: DataFrame with climate data
        baseline_period: Tuple of (start_year, end_year)
        location: Optional location filter
        
    Returns:
        Mean temperature during baseline period or np.nan
    """
    baseline_data = df[
        (df["Year"].between(baseline_period[0], baseline_period[1])) &
        (df["Type"] == "Temp") &
        (df["Name"] == "Average") &
        (df["Season"] == "Annual")
    ]
    
    if location:
        baseline_data = baseline_data[baseline_data["Location"] == location]
    
    if baseline_data.empty:
        return np.nan
    
    return baseline_data["Value"].mean()


def calculate_preindustrial_baselines_by_location(
    all_data: pd.DataFrame,
    baseline_period: Tuple[int, int],
    locations: list
) -> Optional[Dict[str, float]]:
    """
    Calculate pre-industrial baseline temperatures for multiple locations.
    
    Args:
        all_data: DataFrame with all climate data
        baseline_period: Tuple of (start_year, end_year)
        locations: List of location names
        
    Returns:
        Dictionary of {location: baseline_temp} or None if no data
    """
    baseline_temps = {}
    
    historical_data = all_data[
        (all_data["Year"].between(baseline_period[0], baseline_period[1])) &
        (all_data["Type"] == "Temp") &
        (all_data["Name"] == "Average") &
        (all_data["Season"] == "Annual")
    ]
    
    if not historical_data.empty:
        for loc in locations:
            loc_data = historical_data[historical_data["Location"] == loc]
            if not loc_data.empty:
                baseline_temps[loc] = loc_data["Value"].mean()
    
    return baseline_temps if baseline_temps else None


def calculate_preindustrial_baselines_by_metric(
    historical_df: pd.DataFrame,
    baseline_period: Tuple[int, int],
    location: str
) -> Dict[Tuple[str, str], float]:
    """
    Calculate pre-industrial baselines for all metrics at a location.
    
    Args:
        historical_df: DataFrame with historical data
        baseline_period: Tuple of (start_year, end_year)
        location: Location name
        
    Returns:
        Dictionary of {(metric_type, metric_name): baseline_value}
    """
    baselines = {}
    
    historical_data = historical_df[
        (historical_df["Location"] == location) &
        (historical_df["Season"] == "Annual") &
        (historical_df["Year"].between(baseline_period[0], baseline_period[1]))
    ]
    
    if not historical_data.empty:
        for metric_type in historical_data["Type"].unique():
            for metric_name in historical_data[historical_data["Type"] == metric_type]["Name"].unique():
                key = (metric_type, metric_name)
                subset = historical_data[
                    (historical_data["Type"] == metric_type) &
                    (historical_data["Name"] == metric_name)
                ]
                if not subset.empty:
                    baselines[key] = subset["Value"].mean()
    
    return baselines


def find_year_at_global_warming_target(
    df: pd.DataFrame,
    scenario: str,
    location: str,
    regional_baseline: float,
    global_target: float,
    amplification_factor: float
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Find year when global temperature reliably reaches target (e.g., 1.5Â°C).
    
    Uses 5-year forward-looking average to determine when warming reliably crosses threshold.
    This ensures the year represents a sustained crossing, not a temporary spike.
    
    Args:
        df: DataFrame with climate data
        scenario: Scenario name
        location: Location name
        regional_baseline: Regional pre-industrial baseline temperature
        global_target: Target global warming (e.g., 1.5Â°C)
        amplification_factor: Regional warming / global warming ratio
        
    Returns:
        Tuple of (year, regional_warming, global_warming) or (None, None, None)
        Note: regional_warming and global_warming will be close to target values
    """
    # Get annual average temperatures for this scenario and location
    annual_temps = df[
        (df["Scenario"] == scenario) &
        (df["Location"] == location) &
        (df["Type"] == "Temp") &
        (df["Name"] == "Average") &
        (df["Season"] == "Annual")
    ].copy()
    
    if annual_temps.empty:
        return None, None, None
    
    # Group by year and get mean
    annual_temps = annual_temps.groupby("Year")["Value"].mean().reset_index()
    annual_temps = annual_temps.sort_values("Year")
    
    # Calculate regional warming from pre-industrial baseline
    annual_temps["Regional_Warming"] = annual_temps["Value"] - regional_baseline
    
    # Estimate global warming (regional warming is amplified, so divide to get global)
    annual_temps["Estimated_Global_Warming"] = annual_temps["Regional_Warming"] / amplification_factor
    
    # Calculate 5-year forward-looking rolling average
    # This represents the trend over the next 5 years from each year
    annual_temps["Rolling_Global_Warming"] = annual_temps["Estimated_Global_Warming"].rolling(
        window=5, min_periods=3
    ).mean()
    
    # Find first year where 5-year average reliably exceeds target
    warming_years = annual_temps[annual_temps["Rolling_Global_Warming"] >= global_target]
    
    if not warming_years.empty:
        first_row = warming_years.iloc[0]
        year = int(first_row["Year"])
        
        # Calculate regional warming corresponding to the global target
        # This ensures consistency: if global target is 1.5Â°C, regional should be 1.5 Ã— amplification
        target_regional_warming = global_target * amplification_factor
        
        # Return the target values (not the actual values) for consistency
        return year, target_regional_warming, global_target
    
    return None, None, None


def format_metric_card_html(
    display_name: str,
    change_value: Optional[float],
    unit: str,
    baseline_year: int,
    total_value: Optional[float],
    pi_change: Optional[float],
    is_inverse_delta: bool,
    border_color: str = '#e0e0e0',
    bg_color: str = '#fafafa'
) -> str:
    """
    Generate HTML for a metric display card (dashboard style).
    
    Args:
        display_name: Name of metric to display
        change_value: Change from baseline
        unit: Unit of measurement
        baseline_year: Baseline year for comparison
        total_value: Absolute value at target year
        pi_change: Change from pre-industrial baseline (optional)
        is_inverse_delta: Whether to use inverse coloring (increase = bad)
        border_color: Card border color
        bg_color: Card background color
        
    Returns:
        HTML string for the metric card
    """
    if change_value is None:
        return (
            f"<div style='text-align: center; border: 2px solid {border_color}; "
            f"border-radius: 8px; padding: 10px;'>"
            f"<p style='font-size: 24px; color: #333; font-weight: bold; margin: 0;'>{display_name}</p>"
            f"<p style='font-size: 16px; color: #999; margin: 8px 0;'>N/A</p>"
            f"</div>"
        )
    
    # Determine color based on delta direction and inverse flag
    if is_inverse_delta:
        color = '#00c853' if change_value < 0 else '#d32f2f' if change_value > 0 else '#666'
    else:
        color = '#00c853' if change_value > 0 else '#d32f2f' if change_value < 0 else '#666'
    
    display_html = (
        f"<div style='text-align: center; border: 2px solid {border_color}; "
        f"border-radius: 8px; padding: 10px; background: {bg_color};'>"
        f"<p style='font-size: 24px; margin: 0 0 8px 0; color: #333; font-weight: bold;'>{display_name}</p>"
        f"<p style='font-size: 24px; font-weight: bold; color: {color}; margin: 3px 0;'>"
        f"{change_value:+.1f} {unit}</p>"
        f"<p style='font-size: 11px; color: #999; margin: 0;'>from {baseline_year}</p>"
    )
    
    # Add absolute value if available
    if total_value is not None:
        display_html += (
            f"<p style='font-size: 12px; color: #666; margin: 8px 0 0 0;'>"
            f"{total_value:.1f} {unit}</p>"
        )
    
    # Add pre-industrial change if available
    if pi_change is not None:
        if is_inverse_delta:
            pi_color = '#00c853' if pi_change < 0 else '#d32f2f' if pi_change > 0 else '#666'
        else:
            pi_color = '#00c853' if pi_change > 0 else '#d32f2f' if pi_change < 0 else '#666'
        
        display_html += (
            f"<hr style='border: 0; border-top: 1px dashed #ccc; margin: 8px 0;'>"
            f"<p style='font-size: 24px; font-weight: bold; color: {pi_color}; margin: 3px 0;'>"
            f"{pi_change:+.1f} {unit}</p>"
            f"<p style='font-size: 11px; color: #999; margin: 0;'>Pre-Industrial</p>"
        )
    
    display_html += "</div>"
    
    return display_html
