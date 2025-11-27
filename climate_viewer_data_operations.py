#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Viewer Data Operations
================================
Last Updated: 2025-11-26 14:20 AEST - Added align_smoothed_to_agcd function
Previous Update: 2025-11-26 13:30 AEST

Consolidated module containing all data loading, transformation, calculation
and analysis functions.

NO CACHING - All operations run fresh each time to avoid stale data bugs.

Note: Bias correction and scenario alignment are now pre-computed in the
metrics generator (Value_BC column).  The viewer simply selects which column
to use based on the BC toggle.

Post-smoothing alignment: When smoothing is applied, the pre-computed alignment
shifts due to averaging.  The align_smoothed_to_agcd() function re-aligns
scenarios to AGCD at the 2014 transition point after smoothing.

Sections:
  1. Basic Helpers
  2. Data Loading & Schema
  3. Data Transformations
  4. Utility Functions
  5. Analysis Functions
"""

import os
import sys
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence
from io import BytesIO

from climate_viewer_constants import REQUIRED_COLUMNS, BASELINE_PERIOD


# ============================================================================
# SECTION 1: BASIC HELPERS
# ============================================================================

def resolve_folder() -> str:
    """Resolve and validate metrics data folder."""
    from climate_viewer_constants import FOLDER
    if not os.path.isdir(FOLDER):
        sys.exit(f"Invalid folder (metrics root): {FOLDER}")
    return FOLDER


def dedupe_preserve_order(items):
    """Remove duplicates while preserving order."""
    seen, out = set(), []
    for x in items or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def slugify(s: str) -> str:
    """Convert string to URL-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def parse_data_type(series: pd.Series) -> pd.DataFrame:
    """Parse 'Data Type' column into Location, Type and Name components."""
    return series.str.extract(
        r"^(?P<Type>[^ ]+) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$"
    )


# ============================================================================
# SECTION 2: DATA LOADING & SCHEMA
# ============================================================================

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has required columns with correct types.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with ensured schema
    """
    df = df.copy()

    if "Season" not in df.columns:
        df["Season"] = "Annual"

    for col, dtype in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            df[col] = pd.NA
        if dtype == str:
            df[col] = df[col].astype(str).str.strip()
        elif dtype == float:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def discover_scenarios(base_folder: str) -> list[Tuple[str, str, str]]:
    """
    Discover available scenarios in the base folder.

    Args:
        base_folder: Path to folder containing scenario subdirectories

    Returns:
        List of tuples (scenario_name, folder_path, parquet_file_path)
    """
    scenarios = []

    try:
        for name in sorted(os.listdir(base_folder)):
            path = os.path.join(base_folder, name)
            if not os.path.isdir(path):
                continue

            parquet_files = [
                f for f in os.listdir(path)
                if f.startswith("metrics") and f.endswith(".parquet")
            ]

            if parquet_files:
                parquet_path = os.path.join(path, sorted(parquet_files)[0])
                scenarios.append((name, path, parquet_path))
    except Exception as e:
        sys.exit(f"Error discovering scenarios: {e}")

    return scenarios


def load_metrics_file(path: str, use_bc: bool = False) -> pd.DataFrame:
    """
    Load a single metrics parquet file with data quality validation.

    Args:
        path: Path to parquet file
        use_bc: If True and Value_BC column exists, use it as the Value column

    Returns:
        DataFrame with loaded data

    Raises:
        RuntimeError: If NaN values are found in critical columns
    """
    try:
        df = pd.read_parquet(path, engine="pyarrow")

        # Handle column naming: Metric -> Name
        if 'Metric' in df.columns and 'Name' not in df.columns:
            df = df.rename(columns={'Metric': 'Name'})

        # If use_bc and Value_BC column exists, use it as Value
        if use_bc and "Value_BC" in df.columns:
            df["Value"] = df["Value_BC"]

        # Check if we need to parse Data Type or if columns already exist
        has_parsed_cols = all(col in df.columns for col in ['Type', 'Name', 'Location'])

        if not has_parsed_cols:
            # Need to parse Data Type column
            if "Data Type" not in df.columns:
                raise RuntimeError(
                    f"Missing 'Data Type' column in {path}\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Expected columns: Year, Season, Data Type, Value\n"
                    f"Please regenerate the metrics file using climateMetricsGenerator.py"
                )

            # Parse Data Type into Type, Name, Location
            data_type_col = df["Data Type"]
            parts = parse_data_type(data_type_col)

            # Rename 'n' column to 'Name'
            parts = parts.rename(columns={'n': 'Name'})

            # Add parsed columns
            for col in parts.columns:
                df[col] = parts[col]

        df = ensure_schema(df)

        # Data quality check - NaN values should not exist in metrics files
        critical_columns = ["Year", "Value", "Location", "Type", "Name", "Season"]
        nan_issues = {}

        for col in critical_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    nan_issues[col] = nan_count

        if nan_issues:
            error_msg = f"\n{'=' * 70}\n"
            error_msg += f"DATA QUALITY ERROR: NaN values found in metrics file\n"
            error_msg += f"{'=' * 70}\n"
            error_msg += f"File: {path}\n\n"
            error_msg += "NaN values found in the following columns:\n"
            for col, count in nan_issues.items():
                error_msg += f"  - {col}: {count} NaN values ({count / len(df) * 100:.2f}% of data)\n"
            error_msg += f"\nTotal rows in file: {len(df)}\n"
            error_msg += "\nThis indicates a problem with the metrics generation process.\n"
            error_msg += "All metrics files should be complete with no missing values.\n"
            error_msg += "Please regenerate the metrics file using climateMetricsGenerator.py\n"
            error_msg += f"{'=' * 70}\n"
            raise RuntimeError(error_msg)

        df = df.dropna(subset=["Year"])

        return df
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")


def load_minimal_metadata(pairs: Sequence[Tuple[str, str, float]]) -> pd.DataFrame:
    """
    Load minimal metadata (Year, Season, Type, Name, Location) from multiple files.

    Args:
        pairs: List of tuples (label, path, mtime)

    Returns:
        Combined DataFrame with metadata
    """
    frames = []
    for label, path, mtime in pairs:
        _ = mtime

        # Load columns we need - try to get parsed columns if they exist
        df = pd.read_parquet(path, engine="pyarrow")

        # Handle column naming: Metric -> Name
        if 'Metric' in df.columns and 'Name' not in df.columns:
            df = df.rename(columns={'Metric': 'Name'})

        # Check if we need to parse Data Type or if columns already exist
        has_parsed_cols = all(col in df.columns for col in ['Type', 'Name', 'Location'])

        if not has_parsed_cols:
            # Need to parse Data Type column
            if "Data Type" not in df.columns:
                raise RuntimeError(
                    f"Missing 'Data Type' column in {path}\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Please regenerate the metrics file using climateMetricsGenerator.py"
                )

            # Parse Data Type into Type, Name, Location
            parts = parse_data_type(df["Data Type"])
            parts = parts.rename(columns={'n': 'Name'})

            # Add parsed columns
            for col in parts.columns:
                df[col] = parts[col]

        # Keep only metadata columns
        keep_cols = ["Year", "Season", "Type", "Name", "Location"]
        df = df[[c for c in keep_cols if c in df.columns]].copy()

        df["Scenario"] = label

        # Data quality check
        critical_columns = ["Year", "Season", "Location", "Type", "Name"]
        nan_issues = {}

        for col in critical_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    nan_issues[col] = nan_count

        if nan_issues:
            error_msg = f"\n{'=' * 70}\n"
            error_msg += f"DATA QUALITY ERROR: NaN values found in metadata\n"
            error_msg += f"{'=' * 70}\n"
            error_msg += f"Scenario: {label}\n"
            error_msg += f"File: {path}\n\n"
            error_msg += "NaN values found in columns:\n"
            for col, count in nan_issues.items():
                error_msg += f"  - {col}: {count} NaN ({count / len(df) * 100:.2f}%)\n"
            error_msg += f"\nTotal rows: {len(df)}\n"
            error_msg += f"{'=' * 70}\n"
            raise RuntimeError(error_msg)

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = ensure_schema(result)
    return result.dropna(subset=["Year"])


# ============================================================================
# SECTION 3: DATA TRANSFORMATIONS
# ============================================================================

def apply_deltas_vs_base(view: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Calculate deltas relative to base scenario."""
    join_keys = ["Year", "Season", "Data Type", "Location", "Type", "Name"]
    base_values = base[join_keys + ["Value"]].rename(columns={"Value": "BaseValue"})

    merged = view.merge(base_values, on=join_keys, how="left")

    group_keys = ["Season", "Data Type", "Location", "Type", "Name"]

    first_base = base.sort_values("Year").groupby(group_keys, as_index=False).first()
    first_base = first_base[group_keys + ["Value"]].rename(columns={"Value": "FirstBaseValue"})

    merged = merged.merge(first_base, on=group_keys, how="left")
    merged["BaseValue"] = merged["BaseValue"].fillna(merged["FirstBaseValue"])
    merged["Value"] = merged["Value"] - merged["BaseValue"]

    return merged.drop(columns=["BaseValue", "FirstBaseValue"])


def apply_baseline_from_start(view: pd.DataFrame, baseline_year: int = None) -> pd.DataFrame:
    """
    Calculate change from a common baseline year across ALL scenarios.

    This ensures SSP scenarios continue SMOOTHLY from where Historical left off,
    rather than jumping due to model initialisation discontinuities.

    The alignment works by:
    1. Baseline Historical to the start year (shows change from baseline)
    2. For SSP scenarios, apply same baseline PLUS an alignment offset
       so SSP first year matches Historical last year

    Args:
        view: DataFrame with climate data
        baseline_year: Year to use as baseline.  If None, uses minimum year in data.

    Returns:
        DataFrame with values as changes from baseline, with SSP aligned to Historical
    """
    if view.empty:
        return view

    view = view.sort_values("Year").copy()

    # Metric grouping keys (excludes Scenario - we want common baseline across scenarios)
    metric_keys = ["Location", "Type", "Name", "Season", "Data Type"]

    # Determine baseline year
    if baseline_year is None:
        baseline_year = view["Year"].min()

    # Identify Historical scenario (case-insensitive)
    scenarios = view["Scenario"].unique()
    historical_scenario = None
    for s in scenarios:
        if s.lower() == "historical":
            historical_scenario = s
            break

    # Get baseline values from the baseline year
    # Use data from any scenario that has data at baseline_year (typically Historical)
    baseline_data = view[view["Year"] == baseline_year].copy()

    if baseline_data.empty:
        # Fallback: if no data at baseline_year, use minimum year in data
        baseline_year = view["Year"].min()
        baseline_data = view[view["Year"] == baseline_year].copy()

    # For each metric group, get the baseline value (average if multiple scenarios have data)
    baseline_values = baseline_data.groupby(metric_keys, as_index=False)["Value"].mean()
    baseline_values = baseline_values.rename(columns={"Value": "Baseline"})

    # Merge baseline onto all data (applies same baseline to all scenarios)
    result = view.merge(baseline_values, on=metric_keys, how="left")

    # For metrics where no baseline was found (e.g., metric only exists in SSP scenarios),
    # fall back to each scenario's first value
    missing_baseline = result["Baseline"].isna()
    if missing_baseline.any():
        # Get first values per scenario for metrics missing baseline
        scenario_keys = ["Scenario"] + metric_keys
        first_by_scenario = view.sort_values("Year").groupby(scenario_keys, as_index=False).first()
        first_by_scenario = first_by_scenario[scenario_keys + ["Value"]].rename(columns={"Value": "FallbackBaseline"})

        result = result.merge(first_by_scenario, on=scenario_keys, how="left")
        result.loc[missing_baseline, "Baseline"] = result.loc[missing_baseline, "FallbackBaseline"]
        result = result.drop(columns=["FallbackBaseline"])

    # Apply baseline subtraction
    result["Value"] = result["Value"] - result["Baseline"]

    # =========================================================================
    # ALIGNMENT: Make SSP scenarios continue smoothly from Historical
    # =========================================================================
    if historical_scenario is not None:
        # Find transition point: last year of Historical
        historical_data = view[view["Scenario"] == historical_scenario]
        if not historical_data.empty:
            historical_last_year = historical_data["Year"].max()

            # For each SSP scenario, calculate alignment offset
            for scenario in scenarios:
                if scenario == historical_scenario:
                    continue

                scenario_data = view[view["Scenario"] == scenario]
                if scenario_data.empty:
                    continue

                scenario_first_year = scenario_data["Year"].min()

                # Only align if SSP starts after Historical ends (typical case: 2015 vs 2014)
                if scenario_first_year > historical_last_year:
                    # For each metric, calculate alignment offset
                    for _, metric_group in result[result["Scenario"] == scenario].groupby(metric_keys):
                        # Get Historical's baseline-adjusted value at its last year
                        hist_mask = (
                                (result["Scenario"] == historical_scenario) &
                                (result["Year"] == historical_last_year)
                        )
                        for key in metric_keys:
                            hist_mask = hist_mask & (result[key] == metric_group[key].iloc[0])

                        hist_at_transition = result.loc[hist_mask, "Value"]

                        # Get SSP's baseline-adjusted value at its first year
                        ssp_mask = (
                                (result["Scenario"] == scenario) &
                                (result["Year"] == scenario_first_year)
                        )
                        for key in metric_keys:
                            ssp_mask = ssp_mask & (result[key] == metric_group[key].iloc[0])

                        ssp_at_start = result.loc[ssp_mask, "Value"]

                        if not hist_at_transition.empty and not ssp_at_start.empty:
                            # Alignment offset = Historical value - SSP value at transition
                            offset = hist_at_transition.iloc[0] - ssp_at_start.iloc[0]

                            # Apply offset to all rows of this metric in this SSP scenario
                            ssp_metric_mask = (result["Scenario"] == scenario)
                            for key in metric_keys:
                                ssp_metric_mask = ssp_metric_mask & (result[key] == metric_group[key].iloc[0])

                            result.loc[ssp_metric_mask, "Value"] = result.loc[ssp_metric_mask, "Value"] + offset

    return result.drop(columns=["Baseline"])


def apply_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling average smoothing to time series."""
    if window <= 1 or df.empty:
        return df

    window = window if window % 2 == 1 else window + 1
    min_periods = max(1, window // 2)

    group_cols = ["Scenario"] + ["Location", "Type", "Name", "Season", "Data Type"]

    smoothed_groups = []
    for _, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("Year").copy()
        group["Value"] = group["Value"].rolling(
            window, center=True, min_periods=min_periods
        ).mean()
        smoothed_groups.append(group)

    result = pd.concat(smoothed_groups, ignore_index=True)
    return result.dropna(subset=["Value"])


def align_smoothed_to_agcd(df: pd.DataFrame, alignment_year: int = 2014) -> pd.DataFrame:
    """
    Re-align scenarios to AGCD after smoothing has been applied.

    Smoothing shifts values due to averaging, so scenarios that were aligned
    at the raw data level may no longer meet at the transition point.

    This function re-aligns all scenarios to match AGCD at the alignment year.

    Args:
        df: DataFrame with smoothed data (must contain AGCD scenario)
        alignment_year: Year to align to (default 2014, last year of AGCD/Historical overlap)

    Returns:
        DataFrame with re-aligned values
    """
    if df.empty:
        return df

    # Check if AGCD is present
    scenarios = df["Scenario"].unique()
    if "AGCD" not in scenarios:
        # No AGCD to align to - return unchanged
        return df

    df = df.copy()
    metric_keys = ["Location", "Type", "Name", "Season"]

    # Get AGCD data at alignment year
    agcd_data = df[(df["Scenario"] == "AGCD") & (df["Year"] == alignment_year)]

    if agcd_data.empty:
        # AGCD doesn't have alignment year data - return unchanged
        return df

    # Build lookup of AGCD values at alignment year
    agcd_lookup = {}
    for _, row in agcd_data.iterrows():
        key = tuple(row[k] for k in metric_keys)
        agcd_lookup[key] = row["Value"]

    # Process each non-AGCD scenario
    for scenario in scenarios:
        if scenario == "AGCD":
            continue

        # Determine which year to use for this scenario
        is_ssp = scenario.upper().startswith("SSP")
        if is_ssp:
            # SSP scenarios: use 2015 (first year) to align to AGCD 2014
            scen_align_year = 2015
        else:
            # Historical and others: use same year as AGCD
            scen_align_year = alignment_year

        # Get scenario data at its alignment year
        scen_at_year = df[(df["Scenario"] == scenario) & (df["Year"] == scen_align_year)]

        if scen_at_year.empty:
            continue

        # Calculate and apply offsets for each metric
        for _, row in scen_at_year.iterrows():
            key = tuple(row[k] for k in metric_keys)

            if key not in agcd_lookup:
                continue

            agcd_value = agcd_lookup[key]
            scen_value = row["Value"]
            offset = agcd_value - scen_value

            if abs(offset) > 0.001:
                # Apply offset to all rows of this scenario/metric combination
                mask = (df["Scenario"] == scenario)
                for i, k in enumerate(metric_keys):
                    mask = mask & (df[k] == key[i])

                df.loc[mask, "Value"] = df.loc[mask, "Value"] + offset

    return df


# ============================================================================
# SECTION 4: UTILITY FUNCTIONS
# ============================================================================

def get_5year_average(
        data: pd.DataFrame,
        metric_type: str,
        metric_name: str,
        scenario: str,
        center_year: int,
        location: Optional[str] = None
) -> Optional[float]:
    """Get 5-year average centred on target year."""
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

    return None


def find_metric_name(
        data: pd.DataFrame,
        metric_type: str,
        preferred_name: str
) -> str:
    """Find actual metric name in data, handling common variations."""
    available = data[data["Type"] == metric_type]["Name"].unique()

    if preferred_name in available:
        return preferred_name

    # Handle Days variations
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
    """Calculate change in metric from baseline to target year."""
    baseline_val = get_5year_average(data, metric_type, metric_name, scenario, from_year, location)
    target_val = get_5year_average(data, metric_type, metric_name, scenario, to_year, location)

    change_from_start = None
    change_from_preindustrial = None

    if baseline_val is not None and target_val is not None:
        change_from_start = target_val - baseline_val

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
    """Extract climate conditions at target year vs reference year."""
    conditions = {}

    scenario_data = df[
        (df["Scenario"] == scenario) &
        (df["Location"] == location) &
        (df["Season"] == "Annual")
        ]

    if scenario_data.empty:
        return conditions

    for metric_type, name, key, unit in metrics_list:
        actual_metric_name = find_metric_name(scenario_data, metric_type, name)

        target_value = get_5year_average(
            scenario_data, metric_type, actual_metric_name, scenario, target_year
        )
        ref_value = get_5year_average(
            scenario_data, metric_type, actual_metric_name, scenario, reference_year
        )

        if target_value is not None:
            change = target_value - ref_value if ref_value is not None else np.nan

            conditions[key] = {
                'value': target_value,
                'change': change,
                'unit': unit,
                'label': f"{actual_metric_name}"
            }

    return conditions


def calculate_preindustrial_baseline(
        all_data: pd.DataFrame,
        baseline_period: Tuple[int, int],
        location: str
) -> float:
    """Calculate pre-industrial baseline temperature for a location."""
    baseline_data = all_data[
        (all_data["Year"].between(baseline_period[0], baseline_period[1])) &
        (all_data["Type"] == "Temp") &
        (all_data["Name"] == "Average") &
        (all_data["Season"] == "Annual")
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
    """Calculate pre-industrial baseline temperatures for multiple locations."""
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
    """Calculate pre-industrial baselines for all metrics at a location."""
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
    Find year when global temperature reliably reaches target (e.g. 1.5C).

    Uses 5-year forward-looking average to determine sustained crossing.
    """
    annual_temps = df[
        (df["Scenario"] == scenario) &
        (df["Location"] == location) &
        (df["Type"] == "Temp") &
        (df["Name"] == "Average") &
        (df["Season"] == "Annual")
        ].copy()

    if annual_temps.empty:
        return None, None, None

    annual_temps = annual_temps.groupby("Year")["Value"].mean().reset_index()
    annual_temps = annual_temps.sort_values("Year")

    annual_temps["Regional_Warming"] = annual_temps["Value"] - regional_baseline
    annual_temps["Estimated_Global_Warming"] = annual_temps["Regional_Warming"] / amplification_factor

    annual_temps["Rolling_Global_Warming"] = annual_temps["Estimated_Global_Warming"].rolling(
        window=5, min_periods=3
    ).mean()

    warming_years = annual_temps[annual_temps["Rolling_Global_Warming"] >= global_target]

    if not warming_years.empty:
        first_row = warming_years.iloc[0]
        year = int(first_row["Year"])
        target_regional_warming = global_target * amplification_factor
        return year, target_regional_warming, global_target

    return None, None, None


def format_compact_metric_card_html(
        display_name: str,
        change_value: Optional[float],
        unit: str,
        total_value: Optional[float],
        pi_change: Optional[float],
        is_inverse_delta: bool,
        border_color: str = '#e0e0e0',
        bg_color: str = '#fafafa'
) -> str:
    """
    Generate compact HTML for a metric display card.

    Features:
    - No "from XXXX" baseline text (baseline year shown in column header)
    - Tighter margins and padding (6px)
    - Font sizes: name 18px, change 18px, total 11px

    Args:
        display_name: Metric name to display
        change_value: Change from baseline (signed)
        unit: Unit string
        total_value: Absolute value at this time point
        pi_change: Change from pre-industrial baseline
        is_inverse_delta: If True, negative changes are good (green)
        border_color: Card border colour
        bg_color: Card background colour

    Returns:
        HTML string for the card
    """
    if change_value is None:
        return (
            f"<div style='text-align:center;border:2px solid {border_color};"
            f"border-radius:8px;padding:6px;background:{bg_color};'>"
            f"<p style='font-size:18px;color:#333;font-weight:bold;margin:0;'>{display_name}</p>"
            f"<p style='font-size:14px;color:#999;margin:2px 0;'>N/A</p>"
            f"</div>"
        )

    # Determine colours based on whether increase is good or bad
    if is_inverse_delta:
        color = '#00c853' if change_value < 0 else '#d32f2f' if change_value > 0 else '#666'
    else:
        color = '#00c853' if change_value > 0 else '#d32f2f' if change_value < 0 else '#666'

    # Build compact card - smaller fonts, tighter margins, no "from XXXX"
    html = (
        f"<div style='text-align:center;border:2px solid {border_color};"
        f"border-radius:8px;padding:6px;background:{bg_color};'>"
        f"<p style='font-size:18px;color:#333;font-weight:bold;margin:0 0 2px 0;'>{display_name}</p>"
        f"<p style='font-size:18px;font-weight:bold;color:{color};margin:2px 0;'>"
        f"{change_value:+.1f} {unit}</p>"
    )

    # Total value
    if total_value is not None:
        html += (
            f"<p style='font-size:11px;color:#666;margin:2px 0 0 0;'>"
            f"{total_value:.1f} {unit}</p>"
        )

    # Pre-industrial change
    if pi_change is not None:
        if is_inverse_delta:
            pi_color = '#00c853' if pi_change < 0 else '#d32f2f' if pi_change > 0 else '#666'
        else:
            pi_color = '#00c853' if pi_change > 0 else '#d32f2f' if pi_change < 0 else '#666'

        html += (
            f"<hr style='border:0;border-top:1px dashed #ccc;margin:4px 0;'>"
            f"<p style='font-size:18px;font-weight:bold;color:{pi_color};margin:2px 0;'>"
            f"{pi_change:+.1f} {unit}</p>"
            f"<p style='font-size:10px;color:#999;margin:0;'>vs Pre-Industrial</p>"
        )

    html += "</div>"
    return html