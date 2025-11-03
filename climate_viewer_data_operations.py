"""
Climate Viewer Data Operations
Data loading, transformation, and discovery functions.
NO CACHING - all operations run fresh each time to avoid stale data bugs.
"""

import os
import sys
from typing import Sequence, Tuple
import pandas as pd
import numpy as np

from climate_viewer_config import REQUIRED_COLUMNS
from climate_viewer_helpers import parse_data_type
from climate_viewer_constants import BASELINE_PERIOD
from climate_viewer_utils import calculate_preindustrial_baselines_by_location


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


def apply_deltas_vs_base(view: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate deltas relative to base scenario.
    
    Args:
        view: DataFrame with values to transform
        base: DataFrame with base scenario values
        
    Returns:
        DataFrame with delta values
    """
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


def apply_baseline_from_start(view: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate change from first year in each group.
    
    Args:
        view: DataFrame with values
        
    Returns:
        DataFrame with baseline-adjusted values
    """
    if view.empty:
        return view
    
    keys = ["Scenario"] + ["Location", "Type", "Name", "Season", "Data Type"]
    
    view = view.sort_values("Year").copy()
    
    first_values = view.groupby(keys, as_index=False).first()
    first_values = first_values[keys + ["Value"]].rename(columns={"Value": "Baseline"})
    
    result = view.merge(first_values, on=keys, how="left")
    
    result["Value"] = result["Value"] - result["Baseline"]
    
    return result.drop(columns=["Baseline"])


def apply_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Apply rolling average smoothing to time series.
    
    Args:
        df: DataFrame with values
        window: Window size for rolling average (odd number recommended)
        
    Returns:
        DataFrame with smoothed values
    """
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


def load_metrics_file(path: str) -> pd.DataFrame:
    """
    Load a single metrics parquet file.
    
    Args:
        path: Path to parquet file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        
        parts = parse_data_type(df["Data Type"])
        df = pd.concat([df, parts], axis=1)
        
        df = ensure_schema(df)
        
        return df.dropna(subset=["Year"])
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")


def load_minimal_metadata(pairs: Sequence[Tuple[str, str, float]]) -> pd.DataFrame:
    """
    Load minimal metadata (Year, Season, Data Type) from multiple files.
    
    Args:
        pairs: List of tuples (label, path, mtime)
        
    Returns:
        Combined DataFrame with metadata
    """
    frames = []
    for label, path, mtime in pairs:
        _ = mtime  # Not used, kept for signature compatibility
        df = pd.read_parquet(
            path, engine="pyarrow",
            columns=["Year", "Season", "Data Type"]
        )
        parts = parse_data_type(df["Data Type"])
        df = pd.concat([df.drop(columns=["Data Type"]), parts], axis=1)
        df["Scenario"] = label
        frames.append(df)
    
    result = pd.concat(frames, ignore_index=True)
    result = ensure_schema(result)
    return result.dropna(subset=["Year"])


def calculate_preindustrial_baseline(
    all_data: pd.DataFrame,
    locations: list
) -> dict:
    """
    Calculate pre-industrial baseline temperatures for locations.
    Wrapper for utils function.
    
    Args:
        all_data: DataFrame with all climate data
        locations: List of location names
        
    Returns:
        Dictionary of {location: baseline_temp} or None
    """
    return calculate_preindustrial_baselines_by_location(
        all_data, BASELINE_PERIOD, locations
    )
