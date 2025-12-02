#!/usr/bin/env python3
# climateBiasFactors.py - Calculate bias correction factors from Historical vs AGCD comparison
# Last Updated: 2025-11-27 19:30 AEST - Decade-based corrections with trend analysis and Temp extrapolation
# Previous Update: 2025-11-25 19:30 AEST
#
# Compares Historical CMIP6 ACCESS-CM2 metrics against AGCD observational metrics
# using all available overlapping data to calculate bias correction factors.
#
# METHODOLOGY:
# - Calculates bias per decade where AGCD data exists
# - Tests for statistically significant trend in bias over time
# - For Temperature: Extrapolates backwards to 1850 using slope if trend significant,
#   otherwise extends earliest decade's correction as constant
# - For other metrics: Only applies corrections where AGCD data exists (no extrapolation)
#
# AGCD data availability:
#   - Temp (tas, tmax, tmin) and Precip: 1910-2014
#   - Vapour pressure (and derived Humidity/VPD): 1971-2014
#
# Output: bias_correction_factors.json in project root (read by Metrics Generator)

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

# ========= Configuration =========
BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent  # Project root (parent of Utilities_Scenario)
ROOT_METRICS = PROJECT_ROOT / "metricsDataFiles"

# Output file location
OUTPUT_FILE = PROJECT_ROOT / "bias_correction_factors.json"

# Validation period end year (common to all metrics)
VALIDATION_END = 2014

# Pre-industrial period for Temperature extrapolation
PREINDUSTRIAL_START = 1850
PREINDUSTRIAL_END = 1900

# Significance threshold for trend detection
TREND_P_VALUE_THRESHOLD = 0.05

# Seasons to process
SEASONS = ['DJF', 'MAM', 'JJA', 'SON', 'Annual']

# Regions to process
REGIONS = ['Australia', 'Ravenswood']

# Metric types and their correction type
# Additive: bias = AGCD - Historical (used for temperature)
# Multiplicative: factor = AGCD / Historical (used for rainfall)
METRIC_CORRECTION_TYPES = {
    # Rain metrics - multiplicative
    'Rain_Total': 'multiplicative',
    'Rain_Max_Day': 'multiplicative',
    'Rain_Min_Day': 'multiplicative',
    'Rain_Max_5-Day': 'multiplicative',
    'Rain_Min_5-Day': 'multiplicative',
    'Rain_R10mm': 'multiplicative',
    'Rain_R20mm': 'multiplicative',
    'Rain_CDD': 'multiplicative',

    # Temp metrics - additive
    'Temp_Average': 'additive',
    'Temp_Max_Day': 'additive',
    'Temp_Avg_Max': 'additive',
    'Temp_5-Day_Avg_Max': 'additive',
    'Temp_Min_Day': 'additive',
    'Temp_Avg_Min': 'additive',
    'Temp_5-Day_Avg_Min': 'additive',

    # Wind metrics - additive
    'Wind_Average': 'additive',
    'Wind_Max_Day': 'additive',
    'Wind_95th_Percentile': 'additive',

    # Humidity metrics - additive
    'Humidity_Average': 'additive',

    # VPD metrics - additive
    'VPD_Average': 'additive',
}

# Metrics that should be extrapolated backwards (Temperature only)
EXTRAPOLATE_METRICS = {'Temp'}

# Significance threshold for including a correction (must exceed to include)
SIGNIFICANCE_THRESHOLD = {
    'additive': 0.5,  # Must differ by at least 0.5 units
    'multiplicative': 0.05,  # Ratio must differ from 1.0 by at least 5%
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_metrics(scenario_dir: Path) -> pd.DataFrame:
    """Load metrics.parquet from scenario directory."""
    metrics_path = scenario_dir / "metrics.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    return pd.read_parquet(metrics_path, engine="pyarrow")


def parse_data_type(data_type: str) -> Tuple[str, str, str]:
    """Parse canonical data type string: "Rain (Total, Australia)" -> ("Rain", "Total", "Australia")"""
    import re
    m = re.match(r'^(\w+) \(([^,]+), ([^)]+)\)$', data_type)
    if m:
        return m.group(1), m.group(2).strip(), m.group(3).strip()
    return None, None, None


def get_metric_key(metric_type: str, metric_name: str) -> str:
    """Create metric key: ("Temp", "Average") -> "Temp_Average" """
    name = metric_name.replace(' ', '_')
    return f"{metric_type}_{name}"


def get_decade(year: int) -> int:
    """Get decade start year: 1923 -> 1920, 1850 -> 1850"""
    return (year // 10) * 10


def calculate_decade_biases(
        hist_data: pd.DataFrame,
        agcd_data: pd.DataFrame,
        correction_type: str
) -> Dict[int, float]:
    """
    Calculate bias for each decade.

    Args:
        hist_data: Historical scenario data (filtered to metric/season)
        agcd_data: AGCD observational data (filtered to metric/season)
        correction_type: 'additive' or 'multiplicative'

    Returns:
        Dict mapping decade start year -> bias value
    """
    decade_biases = {}

    # Get common years
    hist_years = set(hist_data['Year'].unique())
    agcd_years = set(agcd_data['Year'].unique())
    common_years = sorted(hist_years & agcd_years)

    if not common_years:
        return {}

    # Group by decade
    decades = {}
    for year in common_years:
        decade = get_decade(year)
        if decade not in decades:
            decades[decade] = []
        decades[decade].append(year)

    # Calculate bias per decade
    for decade, years in sorted(decades.items()):
        hist_vals = hist_data[hist_data['Year'].isin(years)]['Value'].dropna()
        agcd_vals = agcd_data[agcd_data['Year'].isin(years)]['Value'].dropna()

        if len(hist_vals) < 3 or len(agcd_vals) < 3:
            # Need at least 3 years per decade for meaningful average
            continue

        hist_mean = float(hist_vals.mean())
        agcd_mean = float(agcd_vals.mean())

        if correction_type == 'additive':
            bias = agcd_mean - hist_mean
        elif correction_type == 'multiplicative':
            if hist_mean == 0:
                continue
            bias = agcd_mean / hist_mean
        else:
            continue

        decade_biases[decade] = bias

    return decade_biases


def check_trend_significance(decade_biases: Dict[int, float]) -> Tuple[bool, float, float, float]:
    """
    Check if there is a statistically significant trend in decade biases.

    Args:
        decade_biases: Dict mapping decade start year -> bias value

    Returns:
        Tuple of (is_significant, slope_per_decade, intercept, p_value)
        - is_significant: True if p < TREND_P_VALUE_THRESHOLD
        - slope_per_decade: Change in bias per decade
        - intercept: Bias at decade 0 (for extrapolation formula)
        - p_value: Statistical significance
    """
    if len(decade_biases) < 3:
        # Not enough data points for meaningful regression
        return False, 0.0, 0.0, 1.0

    decades = np.array(sorted(decade_biases.keys()))
    biases = np.array([decade_biases[d] for d in decades])

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(decades, biases)

    # Slope per decade (multiply by 10 since decades are in units of 10 years)
    slope_per_decade = slope * 10

    is_significant = p_value < TREND_P_VALUE_THRESHOLD

    return is_significant, slope_per_decade, intercept, p_value


def extrapolate_decades(
        decade_biases: Dict[int, float],
        slope_per_decade: float,
        target_start: int,
        target_end: int
) -> Dict[int, float]:
    """
    Extrapolate bias corrections to earlier decades using linear trend.

    Args:
        decade_biases: Existing decade biases
        slope_per_decade: Change in bias per decade
        target_start: Earliest year to extrapolate to
        target_end: Latest year to extrapolate to

    Returns:
        Dict with extrapolated decades added
    """
    if not decade_biases:
        return {}

    result = dict(decade_biases)

    # Find earliest decade with data
    earliest_data_decade = min(decade_biases.keys())
    earliest_bias = decade_biases[earliest_data_decade]

    # Extrapolate backwards
    target_start_decade = get_decade(target_start)

    current_decade = earliest_data_decade - 10
    current_bias = earliest_bias - slope_per_decade

    while current_decade >= target_start_decade:
        result[current_decade] = round(current_bias, 4)
        current_decade -= 10
        current_bias -= slope_per_decade

    return result


def calculate_bias_factors(
        historical_df: pd.DataFrame,
        agcd_df: pd.DataFrame
) -> Tuple[Dict, Dict]:
    """
    Calculate decade-based bias correction factors.

    Returns:
        Tuple of (corrections dict, metadata dict)
    """
    log("Calculating decade-based bias factors...")

    results = {}
    metadata = {
        'validation_periods': {},
        'trend_analysis': {}
    }

    for region in REGIONS:
        results[region] = {}

        for season in SEASONS:
            results[region][season] = {}

            hist_season = historical_df[historical_df['Season'] == season]
            agcd_season = agcd_df[agcd_df['Season'] == season]

            hist_metrics = hist_season[hist_season['Data Type'].str.contains(region)]['Data Type'].unique()
            agcd_metrics = agcd_season[agcd_season['Data Type'].str.contains(region)]['Data Type'].unique()
            common_metrics = set(hist_metrics) & set(agcd_metrics)

            for data_type in sorted(common_metrics):
                metric_type, metric_name, loc = parse_data_type(data_type)
                if not metric_type or loc != region:
                    continue

                metric_key = get_metric_key(metric_type, metric_name)
                correction_type = METRIC_CORRECTION_TYPES.get(metric_key)

                if not correction_type:
                    # Silently skip count metrics (Days > X)
                    if 'Days_>' in metric_key or 'Days_>=' in metric_key:
                        continue
                    continue

                # Get filtered data for this metric
                hist_metric = hist_season[hist_season['Data Type'] == data_type]
                agcd_metric = agcd_season[agcd_season['Data Type'] == data_type]

                # Calculate decade biases
                decade_biases = calculate_decade_biases(hist_metric, agcd_metric, correction_type)

                if not decade_biases:
                    continue

                # Check if any decade has significant bias
                if correction_type == 'additive':
                    max_bias = max(abs(b) for b in decade_biases.values())
                    if max_bias < SIGNIFICANCE_THRESHOLD['additive']:
                        continue
                elif correction_type == 'multiplicative':
                    max_deviation = max(abs(b - 1.0) for b in decade_biases.values())
                    if max_deviation < SIGNIFICANCE_THRESHOLD['multiplicative']:
                        continue

                # Check for trend
                is_significant, slope_per_decade, intercept, p_value = check_trend_significance(decade_biases)

                # Record validation period
                data_start = min(decade_biases.keys())
                data_end = max(decade_biases.keys()) + 9  # End of last decade
                validation_period = f"{data_start}-{min(data_end, VALIDATION_END)}"

                # Store trend analysis metadata
                trend_key = f"{region}_{season}_{metric_key}"
                metadata['trend_analysis'][trend_key] = {
                    'significant': bool(is_significant),  # Convert numpy bool to Python bool
                    'slope_per_decade': round(float(slope_per_decade), 6),
                    'p_value': round(float(p_value), 4),
                    'n_decades': len(decade_biases)
                }

                # For Temperature metrics: Extrapolate to pre-industrial if needed
                if metric_type in EXTRAPOLATE_METRICS:
                    if is_significant and abs(slope_per_decade) > 0.001:
                        # Trend is significant - extrapolate using slope
                        decade_biases = extrapolate_decades(
                            decade_biases,
                            slope_per_decade,
                            PREINDUSTRIAL_START,
                            PREINDUSTRIAL_END
                        )
                        log(f"    {metric_key} ({region}/{season}): Extrapolated to {PREINDUSTRIAL_START} using slope {slope_per_decade:.4f}/decade")
                    else:
                        # No significant trend - use earliest available correction for all prior decades
                        earliest_decade = min(decade_biases.keys())
                        earliest_bias = decade_biases[earliest_decade]

                        current_decade = earliest_decade - 10
                        while current_decade >= get_decade(PREINDUSTRIAL_START):
                            decade_biases[current_decade] = earliest_bias
                            current_decade -= 10

                        log(f"    {metric_key} ({region}/{season}): Extended constant bias {earliest_bias:.4f} to {PREINDUSTRIAL_START}")

                # Round decade biases and convert keys to Python int for JSON serialization
                decade_biases = {int(d): round(float(b), 4) for d, b in sorted(decade_biases.items())}

                # Build result entry
                result_entry = {
                    'type': correction_type,
                    'time_varying': True,
                    'validation_period': validation_period,
                    'trend_significant': bool(is_significant),  # Convert numpy bool to Python bool
                    'slope_per_decade': round(float(slope_per_decade), 6) if is_significant else None,
                    'decades': decade_biases
                }

                # Also include simple average for reference/fallback
                avg_bias = sum(decade_biases.values()) / len(decade_biases)
                result_entry['average'] = round(avg_bias, 4)

                results[region][season][metric_key] = result_entry

                # Track validation period
                if metric_type not in metadata['validation_periods']:
                    metadata['validation_periods'][metric_type] = validation_period

    return results, metadata


def main():
    log("Climate Bias Factors Calculator (Decade-Based)")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Metrics folder: {ROOT_METRICS}")
    log(f"Trend significance threshold: p < {TREND_P_VALUE_THRESHOLD}")

    # Check required scenarios exist
    historical_dir = ROOT_METRICS / "historical"
    agcd_dir = ROOT_METRICS / "AGCD"

    if not historical_dir.exists():
        log(f"[X] Historical scenario not found: {historical_dir}")
        sys.exit(1)

    if not agcd_dir.exists():
        log(f"[X] AGCD scenario not found: {agcd_dir}")
        sys.exit(1)

    # Load metrics
    log("Loading Historical metrics...")
    historical_df = load_metrics(historical_dir)
    log(f"  Loaded {len(historical_df):,} records")
    log(f"  Year range: {historical_df['Year'].min()}-{historical_df['Year'].max()}")

    log("Loading AGCD metrics...")
    agcd_df = load_metrics(agcd_dir)
    log(f"  Loaded {len(agcd_df):,} records")
    log(f"  Year range: {agcd_df['Year'].min()}-{agcd_df['Year'].max()}")

    # Calculate bias factors
    results, calc_metadata = calculate_bias_factors(historical_df, agcd_df)

    # Count total corrections
    total_corrections = sum(
        len(season_data)
        for region_data in results.values()
        for season_data in region_data.values()
    )
    log(f"\nTotal metric corrections: {total_corrections}")

    # Count extrapolated metrics
    extrapolated_count = 0
    for region_data in results.values():
        for season_data in region_data.values():
            for metric_key, metric_data in season_data.items():
                if metric_key.startswith('Temp_') and PREINDUSTRIAL_START // 10 * 10 in metric_data.get('decades', {}):
                    extrapolated_count += 1
    log(f"Temperature metrics extrapolated to {PREINDUSTRIAL_START}: {extrapolated_count}")

    # Add metadata
    output = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "methodology": "Decade-based bias correction with trend analysis",
            "trend_p_threshold": TREND_P_VALUE_THRESHOLD,
            "extrapolation": {
                "enabled_for": list(EXTRAPOLATE_METRICS),
                "target_period": f"{PREINDUSTRIAL_START}-{PREINDUSTRIAL_END}"
            },
            "validation_periods": calc_metadata['validation_periods'],
            "total_corrections": total_corrections
        },
        "trend_analysis": calc_metadata['trend_analysis'],
        "corrections": results
    }

    # Write JSON file
    log(f"\nWriting: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nValidation periods by metric type:")
    for metric_type, period in sorted(calc_metadata['validation_periods'].items()):
        extrapolated = " (extrapolated to 1850)" if metric_type in EXTRAPOLATE_METRICS else ""
        print(f"  {metric_type}: {period}{extrapolated}")

    print("\nTrend analysis summary:")
    significant_trends = sum(1 for v in calc_metadata['trend_analysis'].values() if v['significant'])
    total_trends = len(calc_metadata['trend_analysis'])
    print(f"  Significant trends: {significant_trends}/{total_trends}")

    # Show significant trends
    if significant_trends > 0:
        print("\n  Metrics with significant trends:")
        for key, analysis in sorted(calc_metadata['trend_analysis'].items()):
            if analysis['significant']:
                print(f"    {key}: slope={analysis['slope_per_decade']:.4f}/decade (p={analysis['p_value']:.4f})")

    print("\nCorrections by region/season:")
    for region in REGIONS:
        print(f"\n{region}:")
        for season in SEASONS:
            n = len(results.get(region, {}).get(season, {}))
            if n > 0:
                print(f"  {season}: {n} metrics")

    print("\n" + "=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)

    log("\n[DONE]")


if __name__ == "__main__":
    main()