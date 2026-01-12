#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Viewer Data Operations
================================
Last Updated: 2025-12-02 15:30 AEST - Optimised data loading
Previous Update: 2025-11-26 14:20 AEST

Consolidated module containing all data loading, transformation, calculation
and analysis functions.

Optimisations (2025-12-02):
  - PyArrow direct read with column selection
  - Categorical dtypes for string columns (faster groupby, less memory)
  - Streamlit caching with hash-based invalidation
  - Optimised smoothing via groupby().transform()

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
import hashlib
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
from io import BytesIO

from climate_viewer_constants import REQUIRED_COLUMNS, BASELINE_PERIOD, get_preindustrial_baseline


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


# ============================================================================
# OPTIMISED DATA LOADING
# ============================================================================

# Columns needed for the viewer (skip unused columns for faster loading)
VIEWER_COLUMNS = [
    "Year", "Season", "Location", "Type", "Name",
    "Data Type", "Value", "Value_BC"
]

# Columns to convert to categorical (reduces memory, faster groupby)
CATEGORICAL_COLUMNS = ["Season", "Location", "Type", "Name", "Data Type", "Scenario"]


def load_parquet_optimised(path: str, scenario_label: str,
                           year_min: int = None, year_max: int = None) -> pd.DataFrame:
    """
    Load parquet file with optimisations.

    - Uses PyArrow for faster reads
    - Only loads required columns
    - Filters by year range at read time (predicate pushdown)
    - Converts strings to categoricals

    Args:
        path: Path to parquet file
        scenario_label: Label for this scenario
        year_min: Minimum year to load (inclusive), None for no limit
        year_max: Maximum year to load (inclusive), None for no limit
    """
    # Read only needed columns via PyArrow
    try:
        # Get available columns
        parquet_file = pq.ParquetFile(path)
        available_cols = parquet_file.schema.names

        # Build list of columns to read
        cols_to_read = []
        for col in VIEWER_COLUMNS:
            if col in available_cols:
                cols_to_read.append(col)
            elif col == 'Name' and 'Metric' in available_cols:
                # Handle Metric -> Name rename
                cols_to_read.append('Metric')

        # Build year filter for predicate pushdown
        filters = None
        if year_min is not None and year_max is not None:
            filters = [('Year', '>=', year_min), ('Year', '<=', year_max)]
        elif year_min is not None:
            filters = [('Year', '>=', year_min)]
        elif year_max is not None:
            filters = [('Year', '<=', year_max)]

        # Read with PyArrow (faster than pandas default)
        table = pq.read_table(path, columns=cols_to_read, filters=filters)
        df = table.to_pandas()

    except Exception:
        # Fallback to pandas (no predicate pushdown)
        df = pd.read_parquet(path, engine="pyarrow")
        if year_min is not None:
            df = df[df["Year"] >= year_min]
        if year_max is not None:
            df = df[df["Year"] <= year_max]

    # Rename Metric -> Name if needed
    if 'Metric' in df.columns and 'Name' not in df.columns:
        df = df.rename(columns={'Metric': 'Name'})

    # Add scenario column
    df["Scenario"] = scenario_label

    # Convert to categoricals (big memory and speed win)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_all_data_cached(cache_key: str, labels: tuple, paths: tuple,
                         year_min: int = None, year_max: int = None) -> Dict:
    """
    Load all scenario data with Streamlit caching.

    Args:
        cache_key: Hash of file modification times for cache invalidation
        labels: Tuple of scenario labels
        paths: Tuple of file paths (must match labels order)
        year_min: Minimum year to load (inclusive)
        year_max: Maximum year to load (inclusive)

    Returns:
        Dict with 'df_combined', 'baselines', 'year_range_loaded'
    """
    all_dfs = []

    for label, path in zip(labels, paths):
        df = load_parquet_optimised(path, label, year_min, year_max)
        all_dfs.append(df)

    # Combine all scenarios
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure Scenario is categorical after concat
    if 'Scenario' in df_combined.columns:
        df_combined['Scenario'] = df_combined['Scenario'].astype('category')

    # Get baselines
    if hasattr(df_combined["Location"], 'cat'):
        locations = df_combined["Location"].cat.categories
    else:
        locations = df_combined["Location"].unique()

    baselines = {}
    for loc in locations:
        baseline = get_preindustrial_baseline(loc)
        if baseline is not None:
            baselines[loc] = baseline

    # Track what year range was loaded
    actual_min = int(df_combined["Year"].min()) if not df_combined.empty else year_min
    actual_max = int(df_combined["Year"].max()) if not df_combined.empty else year_max

    return {
        "df_combined": df_combined,
        "baselines": baselines,
        "year_range_loaded": (actual_min, actual_max)
    }


def load_and_process_all_data(
        labels: List[str],
        label_to_path: Dict[str, str],
        year_start: int = None,
        year_end: int = None
) -> Dict:
    """
    Load data from parquet files with optimisations.

    Optimisations:
    1. Uses PyArrow for faster parquet reads with predicate pushdown
    2. Only loads required columns
    3. Only loads years within specified range
    4. Converts strings to categoricals (faster groupby, less memory)
    5. Uses Streamlit's built-in caching with hash-based invalidation
    6. Incremental loading: expands cached data when year range widens

    Args:
        labels: List of scenario labels to load
        label_to_path: Dict mapping labels to file paths
        year_start: Start year (inclusive), None for no limit
        year_end: End year (inclusive), None for no limit

    Returns:
        - df_raw: Combined data using Value column
        - df_bc: Combined data with Value_BC -> Value
        - baselines: Pre-industrial baselines from config
        - year_range_loaded: Tuple of (min_year, max_year) actually loaded
    """
    # Create base cache key from file modification times
    cache_parts = []
    for label in sorted(labels):
        path = label_to_path[label]
        mtime = os.path.getmtime(path)
        cache_parts.append(f"{label}:{mtime}")
    cache_parts.append("v8_yearrange")
    base_cache_key = hashlib.md5("|".join(cache_parts).encode()).hexdigest()[:16]

    # Session state key for tracking loaded range
    range_key = f"loaded_range_{base_cache_key}"

    # Check if we have cached data and what range it covers
    cached_data_key = f"cached_data_{base_cache_key}"

    # Get previously loaded range from session state
    prev_range = st.session_state.get(range_key, (None, None))
    prev_min, prev_max = prev_range

    # Determine what we need to load
    need_full_reload = False
    need_expansion = False
    expand_ranges = []  # List of (min, max) ranges to load additionally

    if cached_data_key not in st.session_state:
        # No cached data - do full load
        need_full_reload = True
    else:
        # Have cached data - check if range expansion needed
        if year_start is not None and prev_min is not None and year_start < prev_min:
            # Need earlier years
            expand_ranges.append((year_start, prev_min - 1))
            need_expansion = True
        if year_end is not None and prev_max is not None and year_end > prev_max:
            # Need later years
            expand_ranges.append((prev_max + 1, year_end))
            need_expansion = True

    if need_full_reload:
        # Full load with year range
        sorted_labels = tuple(sorted(labels))
        sorted_paths = tuple(label_to_path[lbl] for lbl in sorted_labels)

        # Include year range in cache key for Streamlit cache
        range_cache_key = f"{base_cache_key}_{year_start}_{year_end}"

        cached = load_all_data_cached(range_cache_key, sorted_labels, sorted_paths,
                                      year_start, year_end)

        df_combined = cached["df_combined"]
        baselines = cached["baselines"]
        year_range_loaded = cached["year_range_loaded"]

        # Store in session state
        st.session_state[cached_data_key] = {
            "df_combined": df_combined,
            "baselines": baselines
        }
        st.session_state[range_key] = year_range_loaded

    elif need_expansion:
        # Load only the new ranges and append
        existing = st.session_state[cached_data_key]
        df_combined = existing["df_combined"]
        baselines = existing["baselines"]

        sorted_labels = tuple(sorted(labels))
        sorted_paths = tuple(label_to_path[lbl] for lbl in sorted_labels)

        for exp_min, exp_max in expand_ranges:
            # Load the expansion range
            exp_cache_key = f"{base_cache_key}_{exp_min}_{exp_max}"
            expansion = load_all_data_cached(exp_cache_key, sorted_labels, sorted_paths,
                                             exp_min, exp_max)

            # Append to existing data
            df_combined = pd.concat([df_combined, expansion["df_combined"]],
                                    ignore_index=True)

            # Ensure categoricals are preserved
            for col in CATEGORICAL_COLUMNS:
                if col in df_combined.columns:
                    df_combined[col] = df_combined[col].astype('category')

        # Update stored range
        new_min = min(prev_min or year_start, year_start) if year_start else prev_min
        new_max = max(prev_max or year_end, year_end) if year_end else prev_max
        year_range_loaded = (new_min, new_max)

        # Update session state
        st.session_state[cached_data_key] = {
            "df_combined": df_combined,
            "baselines": baselines
        }
        st.session_state[range_key] = year_range_loaded

    else:
        # Use existing cached data (range contraction or same range)
        existing = st.session_state[cached_data_key]
        df_combined = existing["df_combined"]
        baselines = existing["baselines"]
        year_range_loaded = prev_range

    # Filter to requested range (handles contraction case)
    if year_start is not None or year_end is not None:
        mask = pd.Series(True, index=df_combined.index)
        if year_start is not None:
            mask = mask & (df_combined["Year"] >= year_start)
        if year_end is not None:
            mask = mask & (df_combined["Year"] <= year_end)
        df_filtered = df_combined[mask]
    else:
        df_filtered = df_combined

    # Create raw view
    df_raw = df_filtered

    # Create BC view (copy Value_BC to Value)
    if "Value_BC" in df_filtered.columns:
        df_bc = df_filtered.copy()
        df_bc["Value"] = df_bc["Value_BC"]
    else:
        df_bc = df_filtered

    return {
        "df_raw": df_raw,
        "df_bc": df_bc,
        "baselines": baselines,
        "year_range_loaded": year_range_loaded,
    }


# ============================================================================
# BACKWARDS COMPATIBILITY WRAPPERS
# ============================================================================

def load_metrics_file(path: str, use_bc: bool = False) -> pd.DataFrame:
    """
    Load a single metrics parquet file.

    Backwards-compatible wrapper around load_parquet_optimised.

    Args:
        path: Path to parquet file
        use_bc: If True and Value_BC column exists, use it as the Value column

    Returns:
        DataFrame with loaded data
    """
    df = load_parquet_optimised(path, scenario_label="")

    # Remove the empty Scenario column we added
    if "Scenario" in df.columns:
        df = df.drop(columns=["Scenario"])

    # Apply BC if requested
    if use_bc and "Value_BC" in df.columns:
        df["Value"] = df["Value_BC"]

    # Ensure schema
    df = ensure_schema(df)

    return df


def load_minimal_metadata(pairs: Sequence[Tuple[str, str, float]]) -> pd.DataFrame:
    """
    Load minimal metadata (Year, Season, Type, Name, Location) from multiple files.

    Backwards-compatible wrapper using optimised loading.

    Args:
        pairs: List of tuples (label, path, mtime)

    Returns:
        Combined DataFrame with metadata
    """
    frames = []

    for label, path, mtime in pairs:
        _ = mtime  # Not used but kept for API compatibility

        # Use optimised loader
        df = load_parquet_optimised(path, label)

        # Keep only metadata columns
        keep_cols = ["Year", "Season", "Type", "Name", "Location", "Scenario"]
        df = df[[c for c in keep_cols if c in df.columns]].copy()

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    # Ensure categoricals after concat
    for col in CATEGORICAL_COLUMNS:
        if col in result.columns:
            result[col] = result[col].astype('category')

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
    """
    Apply rolling average smoothing to time series.

    Optimised version using groupby().transform() instead of explicit loops.
    """
    if window <= 1 or df.empty:
        return df

    window = window if window % 2 == 1 else window + 1
    min_periods = max(1, window // 2)

    group_cols = ["Scenario", "Location", "Type", "Name", "Season", "Data Type"]

    # Sort once
    df = df.sort_values(group_cols + ["Year"]).copy()

    # Apply rolling mean via transform (faster than explicit loop)
    df["Value"] = df.groupby(group_cols, observed=True)["Value"].transform(
        lambda x: x.rolling(window, center=True, min_periods=min_periods).mean()
    )

    return df.dropna(subset=["Value"])


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