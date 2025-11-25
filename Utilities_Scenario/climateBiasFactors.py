#!/usr/bin/env python3
# climateBiasFactors.py - Calculate bias correction factors from Historical vs AGCD comparison
# Last Updated: 2025-11-25 19:30 AEST
#
# Compares Historical CMIP6 ACCESS-CM2 metrics against AGCD observational metrics
# for the validation period (1971-2014) to calculate bias correction factors.
#
# Output: bias_correction_factors.json in project root (read by Climate Viewer)

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

# ========= Configuration =========
BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent  # Project root (parent of Utilities_Scenario)
ROOT_METRICS = PROJECT_ROOT / "metricsDataFiles"

# Output file location
OUTPUT_FILE = PROJECT_ROOT / "bias_correction_factors.json"

# Validation period for bias calculation
VALIDATION_START = 1971
VALIDATION_END = 2014

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

    # Temp metrics - additive (tasmax)
    'Temp_Average': 'additive',
    'Temp_Max_Day': 'additive',
    'Temp_Avg_Max': 'additive',
    'Temp_5-Day_Avg_Max': 'additive',

    # Temp metrics - additive (tasmin)
    'Temp_Min_Day': 'additive',
    'Temp_Avg_Min': 'additive',
    'Temp_5-Day_Avg_Min': 'additive',
    # NOTE: Temp_Days count metrics excluded - bias correction not applicable to counts

    # Wind metrics - additive
    'Wind_Average': 'additive',
    'Wind_Max_Day': 'additive',
    'Wind_95th_Percentile': 'additive',

    # Humidity metrics - additive
    'Humidity_Average': 'additive',

    # VPD metrics - additive
    'VPD_Average': 'additive',
}

# Significance threshold (must exceed to include correction)
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


def calculate_bias_factors(
        historical_df: pd.DataFrame,
        agcd_df: pd.DataFrame
) -> Dict:
    """Calculate bias correction factors by comparing Historical vs AGCD."""
    log(f"Calculating bias factors for period {VALIDATION_START}-{VALIDATION_END}")

    # Filter to validation period
    hist = historical_df[(historical_df['Year'] >= VALIDATION_START) &
                         (historical_df['Year'] <= VALIDATION_END)].copy()
    agcd = agcd_df[(agcd_df['Year'] >= VALIDATION_START) &
                   (agcd_df['Year'] <= VALIDATION_END)].copy()

    n_years = VALIDATION_END - VALIDATION_START + 1
    log(f"  Historical records: {len(hist):,}")
    log(f"  AGCD records: {len(agcd):,}")
    log(f"  Validation years: {n_years}")

    results = {}

    for region in REGIONS:
        results[region] = {}

        for season in SEASONS:
            results[region][season] = {}

            hist_season = hist[hist['Season'] == season]
            agcd_season = agcd[agcd['Season'] == season]

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
                    # Silently skip count metrics (Days > X) - bias correction not applicable
                    if 'Days_>' in metric_key or 'Days_>=' in metric_key:
                        continue
                    log(f"    [!] Unknown metric type: {metric_key}")
                    continue

                hist_vals = hist_season[hist_season['Data Type'] == data_type]['Value'].dropna()
                agcd_vals = agcd_season[agcd_season['Data Type'] == data_type]['Value'].dropna()

                if len(hist_vals) == 0 or len(agcd_vals) == 0:
                    continue

                hist_mean = float(hist_vals.mean())
                agcd_mean = float(agcd_vals.mean())
                n_samples = min(len(hist_vals), len(agcd_vals))

                if correction_type == 'additive':
                    bias = agcd_mean - hist_mean
                    if abs(bias) < SIGNIFICANCE_THRESHOLD['additive']:
                        continue
                    results[region][season][metric_key] = {
                        'value': round(bias, 4),
                        'type': 'additive'
                    }

                elif correction_type == 'multiplicative':
                    if hist_mean == 0:
                        continue
                    factor = agcd_mean / hist_mean
                    if abs(factor - 1.0) < SIGNIFICANCE_THRESHOLD['multiplicative']:
                        continue
                    results[region][season][metric_key] = {
                        'value': round(factor, 4),
                        'type': 'multiplicative'
                    }

    return results


def main():
    log("Climate Bias Factors Calculator")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Metrics folder: {ROOT_METRICS}")

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

    log("Loading AGCD metrics...")
    agcd_df = load_metrics(agcd_dir)
    log(f"  Loaded {len(agcd_df):,} records")

    # Calculate bias factors
    results = calculate_bias_factors(historical_df, agcd_df)

    # Count total corrections
    total_corrections = sum(
        len(season_data)
        for region_data in results.values()
        for season_data in region_data.values()
    )
    log(f"\nTotal bias corrections: {total_corrections}")

    # Add metadata
    output = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "validation_period": f"{VALIDATION_START}-{VALIDATION_END}",
            "source": "Historical vs AGCD comparison",
            "total_corrections": total_corrections
        },
        "corrections": results
    }

    # Write single JSON file to project root
    log(f"\nWriting: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for region in REGIONS:
        print(f"\n{region}:")
        for season in SEASONS:
            n = len(results.get(region, {}).get(season, {}))
            if n > 0:
                print(f"  {season}: {n} corrections")
    print("\n" + "=" * 50)
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 50)

    log("\n[DONE]")


if __name__ == "__main__":
    main()