#!/usr/bin/env python3
# climateAGCDImporter.py — AGCD Observational Data Importer
#
# Imports Australian Gridded Climate Data (AGCD) observational records
# Outputs identical format to CMIP6 importer for seamless metrics generation
#
# Features:
# - Loads local AGCD NetCDF files
# - IPCC-standard land-sea masking (>50% land threshold)
# - Regional mean (Australia land-only) + point extraction (Ravenswood)
# - Compatible output format with CMIP6 importer
# - Daily temporal resolution
# - Improved progress tracking with time estimates

import os
import sys
import json
import warnings
import gc
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# ===== Configuration =====
BASE = Path(__file__).resolve().parent
ROOT_METRICS = BASE / "metricsDataFiles"

# AGCD variable mapping
AGCD_VARIABLES = {
    'tmax': {
        'nc_name': 'tmax',
        'description': 'Daily maximum temperature',
        'unit': 'K → °C',
        'output_name': 'tasmax'
    },
    'tmin': {
        'nc_name': 'tmin',
        'description': 'Daily minimum temperature',
        'unit': 'K → °C',
        'output_name': 'tasmin'
    },
    'precip': {
        'nc_name': 'precip',
        'description': 'Daily precipitation',
        'unit': 'mm',
        'output_name': 'pr'
    },
    'vprp09': {
        'nc_name': 'vapourpres',
        'description': 'Vapour pressure at 9am',
        'unit': 'hPa',
        'output_name': 'vprp09'
    }
}

# Ravenswood location
RAVENSWOOD_LAT = -20.115
RAVENSWOOD_LON = 146.900


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ===== Grid Functions (from CMIP6 importer) =====

def calculate_grid_fingerprint(lats, lons):
    """Calculate fingerprint/hash for grid comparison"""
    grid_signature = {
        'shape': [len(lats), len(lons)],
        'lat_first': float(lats[0]),
        'lat_last': float(lats[-1]),
        'lon_first': float(lons[0]),
        'lon_last': float(lons[-1]),
        'lat_spacing_mean': float(np.mean(np.diff(lats))),
        'lon_spacing_mean': float(np.mean(np.diff(lons)))
    }
    sig_str = json.dumps(grid_signature, sort_keys=True)
    return hashlib.md5(sig_str.encode()).hexdigest(), grid_signature


def _calculate_cell_bounds_from_centres(centres: np.ndarray) -> np.ndarray:
    """Calculate cell boundaries from centre points"""
    centres = np.asarray(centres)
    midpoints = (centres[:-1] + centres[1:]) / 2
    first_edge = centres[0] - (centres[1] - centres[0]) / 2
    last_edge = centres[-1] + (centres[-1] - centres[-2]) / 2
    return np.concatenate([[first_edge], midpoints, [last_edge]])


# ===== Land-Sea Masking Functions =====

def find_local_sftlf(base_dir: Path = None) -> Optional[Path]:
    """Search for local sftlf (land fraction) file for Australia"""
    if base_dir is None:
        base_dir = Path(__file__).parent

    patterns = [
        "sftlf_australia.nc",
        "sftlf_AGCD.nc",
        "australia_landfrac.nc",
        "sftlf_fx_australia.nc"
    ]

    for pattern in patterns:
        candidate = base_dir / pattern
        if candidate.exists():
            log(f"✓ Found local sftlf file: {candidate.name}")
            return candidate

    return None


def load_sftlf_from_file(sftlf_path: Path, bbox: Optional[tuple] = None) -> xr.Dataset:
    """Load sftlf from local NetCDF file"""
    log(f"Loading land fraction from {sftlf_path.name}")

    try:
        ds = xr.open_dataset(sftlf_path)

        if bbox is not None:
            try:
                lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
                lon_name = 'lon' if 'lon' in ds.coords else 'longitude'

                if lon_name in ds.coords:
                    ds = ds.assign_coords({
                        lon_name: ((ds[lon_name] + 180) % 360) - 180
                    }).sortby(lon_name)

                lat_slice = slice(bbox[2], bbox[0])
                lon_slice = slice(bbox[1], bbox[3])

                ds = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
                log(f"   Subset to bounding box: {bbox}")
            except Exception as e:
                log(f"   ⚠️  Could not subset to bbox: {e}")

        return ds

    except Exception as e:
        log(f"❌ Failed to load {sftlf_path}: {e}")
        return None


def create_land_mask(sftlf_da: xr.DataArray, threshold: float = 50.0) -> xr.DataArray:
    """Create binary land mask from land fraction data (IPCC >50% threshold)"""
    land_mask = xr.where(sftlf_da > threshold, 1.0, 0.0)
    land_mask = xr.where(land_mask > 0, 1.0, np.nan)
    return land_mask


def area_mean_land_only(da: xr.DataArray, lat: str, lon: str,
                        land_mask: Optional[xr.DataArray] = None) -> xr.DataArray:
    """Calculate area-weighted mean over LAND ONLY (IPCC methodology)"""
    if land_mask is not None:
        da_masked = da * land_mask
    else:
        da_masked = da

    weights = np.cos(np.deg2rad(da_masked[lat]))
    weighted_mean = da_masked.weighted(weights).mean(dim=[lat, lon], skipna=True)

    return weighted_mean


# ===== Coordinate & Unit Functions =====

def find_lat_lon(ds):
    """Find latitude/longitude coordinate names"""
    lat = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
    lon = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
    if not lat or not lon:
        raise ValueError("Could not find lat/lon coordinates")
    return lat, lon


def normalise_lon(ds, lon):
    """Normalise longitude to -180 to 180 range"""
    try:
        ds = ds.assign_coords({lon: ((ds[lon] + 180) % 360) - 180}).sortby(lon)
    except Exception:
        pass
    return ds


def temp_to_degC(da):
    """Convert temperature from Kelvin to degrees Celsius"""
    units = (da.attrs.get("units", "").lower())
    if units in ("k", "kelvin"):
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da


def precip_to_mm(da):
    """Ensure precipitation is in mm"""
    units = (da.attrs.get("units") or "").lower()
    if units in {"mm", "mm/day"}:
        out = da.copy()
        out.attrs["units"] = "mm"
    else:
        out = da.copy()
        out.attrs["units"] = "mm"
    return out


def extract_point(da, lat, lon, lat_tgt, lon_tgt):
    """Extract nearest grid cell to target point"""
    return da.sel({lat: lat_tgt, lon: lon_tgt}, method="nearest")


def to_series(da):
    """Convert DataArray to pandas Series"""
    da = da.compute() if hasattr(da.data, "compute") else da
    t = da["time"].values
    idx = [pd.to_datetime(x).strftime("%Y-%m-%d") for x in t]
    return pd.Series(np.asarray(da.values).astype(float), index=pd.Index(idx, name="time"))


def calculate_tas_from_tmax_tmin(datasets: Dict[str, xr.Dataset]) -> Optional[xr.Dataset]:
    """
    Calculate mean temperature (tas) from tmax and tmin

    tas = (tmax + tmin) / 2

    Args:
        datasets: Dictionary of loaded datasets

    Returns:
        Dataset with tas variable, or None if tmax/tmin not available
    """
    if 'tasmax' not in datasets or 'tasmin' not in datasets:
        return None

    log("Calculating tas (mean temperature) from tmax and tmin...")

    tmax_da = datasets['tasmax']['tasmax']
    tmin_da = datasets['tasmin']['tasmin']

    # Calculate mean
    tas_da = (tmax_da + tmin_da) / 2.0
    tas_da.name = 'tas'
    tas_da.attrs['long_name'] = 'Daily mean near-surface air temperature'
    tas_da.attrs['description'] = 'Calculated as (tmax + tmin) / 2'

    # Create dataset
    tas_ds = tas_da.to_dataset()

    log("   ✓ Calculated tas from tmax and tmin")

    return tas_ds


# ===== File Finding =====

def find_agcd_files(agcd_dir: Path, variables: List[str]) -> Dict[str, List[Path]]:
    """
    Find AGCD NetCDF files for specified variables

    AGCD data comes as one file per year, so this returns lists of files.

    Expected naming conventions:
    - tmax: agcd_v1_tmax_mean_r005_01day_YYYY.nc
    - tmin: agcd_v1_tmin_mean_r005_01day_YYYY.nc
    - precip: agcd_v1_precip_total_r005_01day_YYYY.nc

    Or similar patterns
    """
    found_files = {}

    for var in variables:
        nc_name = AGCD_VARIABLES[var]['nc_name']

        # Try multiple naming patterns
        patterns = [
            f"*{nc_name}*daily*.nc",
            f"*{nc_name}*01day*.nc",
            f"agcd*{nc_name}*.nc",
            f"AGCD*{nc_name}*.nc",
            f"*vapourpres*h09*.nc" if var == 'vprp09' else None
        ]

        # Filter out None values
        patterns = [p for p in patterns if p is not None]

        matches = []
        for pattern in patterns:
            found = list(agcd_dir.glob(pattern))
            if found:
                matches.extend(found)

        # Remove duplicates and sort by name (which sorts by year)
        matches = sorted(set(matches))

        if matches:
            found_files[var] = matches
            log(f"✓ Found {var}: {len(matches)} file(s)")
        else:
            log(f"⚠️  Could not find {var} files in {agcd_dir}")

    return found_files


# ===== Sequential File-by-File Processing (NO MERGE) =====

def process_single_file(file_path: Path, var: str,
                        land_mask: Optional[xr.DataArray],
                        region_name: str,
                        bbox: Optional[Tuple[float, float, float, float]] = None,
                        year_range: Optional[Tuple[int, int]] = None) -> Optional[pd.DataFrame]:
    """
    Process a single AGCD file and return the extracted data

    Returns:
        DataFrame with regional mean and point data, or None if error
    """
    try:
        ds = xr.open_dataset(file_path)

        # Find the data variable
        nc_name = AGCD_VARIABLES[var]['nc_name']
        if nc_name not in ds:
            data_vars = [v for v in ds.data_vars if v not in ['time', 'lat', 'lon', 'latitude', 'longitude']]
            if data_vars:
                nc_name = data_vars[0]
            else:
                return None

        # Rename to standard name
        output_name = AGCD_VARIABLES[var]['output_name']
        if nc_name != output_name:
            ds = ds.rename({nc_name: output_name})

        # Find lat/lon
        lat, lon = find_lat_lon(ds)
        ds = normalise_lon(ds, lon)

        # Subset to bbox if provided
        if bbox is not None:
            try:
                lat_slice = slice(bbox[2], bbox[0])
                lon_slice = slice(bbox[1], bbox[3])
                ds = ds.sel({lat: lat_slice, lon: lon_slice})
            except Exception:
                pass

        # Subset to year range if provided
        if year_range is not None:
            try:
                start_date = f"{year_range[0]}-01-01"
                end_date = f"{year_range[1]}-12-31"
                ds = ds.sel(time=slice(start_date, end_date))
            except Exception:
                pass

        da = ds[output_name]

        # Unit conversions
        if output_name in ['tas', 'tasmax', 'tasmin']:
            da = temp_to_degC(da)
            unit = "degC"
        elif output_name == 'pr':
            da = precip_to_mm(da)
            unit = "mm_day"
        elif output_name == 'vprp09':
            unit = "hPa"
        else:
            unit = "unknown"

        # Compute regional mean WITH LAND MASKING
        area_col = f"{output_name}_{region_name}_land_only_{unit}"
        area_mean_result = area_mean_land_only(da, lat, lon, land_mask)

        # Compute point extraction
        point_col = f"{output_name}_Ravenswood_{unit}"
        point_result = extract_point(da, lat, lon, RAVENSWOOD_LAT, RAVENSWOOD_LON)

        # Convert to series
        data_dict = {
            area_col: to_series(area_mean_result),
            point_col: to_series(point_result)
        }

        df = pd.DataFrame(data_dict)

        ds.close()
        del ds, da
        gc.collect()

        return df

    except Exception as e:
        return None


def load_and_process_agcd_sequential(agcd_dir: Path, variables: List[str],
                                     land_mask: Optional[xr.DataArray],
                                     region_name: str,
                                     year_range: Optional[Tuple[int, int]] = None,
                                     bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[
    pd.DataFrame, Dict]:
    """
    Load and process AGCD files sequentially (one file at a time)

    This approach is MUCH FASTER than loading all files and merging:
    - No expensive xarray merge operation
    - Lower memory usage (one file at a time)
    - Better progress tracking
    - More robust to failures

    Returns:
        Tuple of (combined DataFrame with all variables, grid_info dict)
    """
    log(f"Loading and processing AGCD data from {agcd_dir}")

    file_map = find_agcd_files(agcd_dir, variables)

    if not file_map:
        log("❌ No AGCD files found")
        return pd.DataFrame(), {}

    # We'll build grid info from the first file we process
    grid_info = None

    # Store all DataFrames by variable
    variable_dfs = {}

    for var, file_list in file_map.items():
        output_name = AGCD_VARIABLES[var]['output_name']
        log(f"\nProcessing {var} → {output_name} ({len(file_list)} file(s))...")

        # Sort files by filename (year)
        file_list_sorted = sorted(file_list)
        total_files = len(file_list_sorted)

        # Track timing
        start_time = time.time()

        # Process files one by one
        all_dfs = []

        for i, fpath in enumerate(file_list_sorted, 1):
            # Time estimate
            if i > 1:
                elapsed = time.time() - start_time
                avg_per_file = elapsed / (i - 1)
                remaining = (total_files - i) * avg_per_file
                eta_str = f"ETA {int(remaining)}s"
            else:
                eta_str = "calculating..."

            # Extract year from filename for display
            fname = fpath.name
            year = "????"
            for part in fname.split('_'):
                if part.isdigit() and len(part) == 4:
                    year = part
                    break

            print(f"\r   Year {year}: {i}/{total_files} ({eta_str})    ", end="", flush=True)

            df = process_single_file(fpath, var, land_mask, region_name, bbox, year_range)

            if df is not None and not df.empty:
                all_dfs.append(df)

            # Build grid info from first file (only once)
            if grid_info is None and df is not None:
                try:
                    ds_temp = xr.open_dataset(fpath)
                    lat, lon = find_lat_lon(ds_temp)
                    ds_temp = normalise_lon(ds_temp, lon)

                    lats = ds_temp[lat].values
                    lons = ds_temp[lon].values
                    fingerprint, signature = calculate_grid_fingerprint(lats, lons)

                    lat_idx = int(np.abs(lats - RAVENSWOOD_LAT).argmin())
                    lon_idx = int(np.abs(lons - RAVENSWOOD_LON).argmin())

                    actual_lat = float(lats[lat_idx])
                    actual_lon = float(lons[lon_idx])

                    lat_bounds = _calculate_cell_bounds_from_centres(lats)
                    lon_bounds = _calculate_cell_bounds_from_centres(lons)

                    lat_min = float(lat_bounds[lat_idx])
                    lat_max = float(lat_bounds[lat_idx + 1])
                    lon_min = float(lon_bounds[lon_idx])
                    lon_max = float(lon_bounds[lon_idx + 1])

                    grid_info = {
                        "fingerprint": fingerprint,
                        "signature": signature,
                        "grid_shape": [len(lats), len(lons)],
                        "lat_values": lats.tolist(),
                        "lon_values": lons.tolist(),
                        "lat_coord_name": lat,
                        "lon_coord_name": lon,
                        "ravenswood": {
                            "target": {"lat": RAVENSWOOD_LAT, "lon": RAVENSWOOD_LON},
                            "grid_indices": [int(lat_idx), int(lon_idx)],
                            "cell_centre": {"lat": actual_lat, "lon": actual_lon},
                            "cell_bounds": {"lat": [lat_min, lat_max], "lon": [lon_min, lon_max]},
                            "offset_from_target": {"lat": actual_lat - RAVENSWOOD_LAT,
                                                   "lon": actual_lon - RAVENSWOOD_LON},
                            "cell_area_sq_deg": (lat_max - lat_min) * (lon_max - lon_min)
                        }
                    }

                    # Add land mask info
                    if land_mask is not None:
                        land_cell_indices = []
                        mask_array = land_mask.values

                        for i_lat in range(len(lats)):
                            for i_lon in range(len(lons)):
                                if not np.isnan(mask_array[i_lat, i_lon]):
                                    land_cell_indices.append([int(i_lat), int(i_lon)])

                        grid_info["land_sea_mask"] = {
                            "applied": True,
                            "threshold_percent": 50.0,
                            "method": "IPCC AR6 standard",
                            "total_cells": int(land_mask.size),
                            "land_cells": len(land_cell_indices),
                            "ocean_cells": int(land_mask.size) - len(land_cell_indices),
                            "land_percentage": (len(land_cell_indices) / land_mask.size) * 100,
                            "land_cell_indices": land_cell_indices,
                            "description": "Grid cells with >50% land fraction included in regional statistics"
                        }
                    else:
                        grid_info["land_sea_mask"] = {
                            "applied": False,
                            "description": "No land masking applied"
                        }

                    ds_temp.close()
                    del ds_temp

                except Exception as e:
                    pass

        # Combine all DataFrames for this variable
        if all_dfs:
            combined_df = pd.concat(all_dfs, axis=0).sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

            total_time = int(time.time() - start_time)
            print(f"\r   ✓ Processed {total_files} files in {total_time}s ({len(combined_df):,} timesteps)" + " " * 20)

            variable_dfs[output_name] = combined_df
        else:
            log(f"   ⚠️  No data extracted for {var}")

    # Combine all variables into one DataFrame
    if variable_dfs:
        # Calculate tas (mean temperature) from tmax and tmin if both are available
        if 'tasmax' in variable_dfs and 'tasmin' in variable_dfs:
            log("\nCalculating tas (mean temperature) from tasmax and tasmin...")

            tasmax_df = variable_dfs['tasmax']
            tasmin_df = variable_dfs['tasmin']

            # Find the column names
            tasmax_cols = [c for c in tasmax_df.columns if c.startswith('tasmax_')]
            tasmin_cols = [c for c in tasmin_df.columns if c.startswith('tasmin_')]

            # Calculate tas for each column (regional mean and point)
            tas_data = {}
            for tmax_col, tmin_col in zip(tasmax_cols, tasmin_cols):
                # Create tas column name by replacing tasmax/tasmin with tas
                tas_col = tmax_col.replace('tasmax_', 'tas_')
                tas_data[tas_col] = (tasmax_df[tmax_col] + tasmin_df[tmin_col]) / 2.0

            variable_dfs['tas'] = pd.DataFrame(tas_data)
            log("   ✓ Calculated tas from tasmax and tasmin")

        # Find common date range across all variables to avoid null values
        log("\nAligning variables to common date range...")

        # Convert indices to datetime if they're strings
        for var_name, var_df in variable_dfs.items():
            if not isinstance(var_df.index, pd.DatetimeIndex):
                variable_dfs[var_name].index = pd.to_datetime(var_df.index)

        # Get date range for each variable
        date_ranges = {}
        for var_name, var_df in variable_dfs.items():
            date_ranges[var_name] = (var_df.index.min(), var_df.index.max())
            log(f"   {var_name}: {var_df.index.min().strftime('%Y-%m-%d')} to {var_df.index.max().strftime('%Y-%m-%d')} ({len(var_df):,} days)")

        # Find intersection (common period across all variables)
        common_start = max(start for start, end in date_ranges.values())
        common_end = min(end for start, end in date_ranges.values())

        log(f"\n   Common period: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")

        if common_start > common_end:
            log("   ❌ No overlapping date range across all variables!")
            return pd.DataFrame(), grid_info

        # Trim all DataFrames to common range
        trimmed_dfs = {}
        for var_name, var_df in variable_dfs.items():
            trimmed = var_df.loc[common_start:common_end]
            original_len = len(var_df)
            trimmed_len = len(trimmed)
            dropped = original_len - trimmed_len
            if dropped > 0:
                log(f"   {var_name}: dropped {dropped:,} days outside common range")
            trimmed_dfs[var_name] = trimmed

        # Create continuous date range (no gaps)
        log(f"\n   Creating continuous date range...")
        continuous_dates = pd.date_range(start=common_start, end=common_end, freq='D')
        log(f"   Expected days: {len(continuous_dates):,}")

        # Reindex each DataFrame to continuous dates (fills gaps with NaN)
        reindexed_dfs = {}
        total_gaps = 0
        for var_name, var_df in trimmed_dfs.items():
            reindexed = var_df.reindex(continuous_dates)
            gaps = reindexed.isna().sum().sum()
            if gaps > 0:
                log(f"   {var_name}: {gaps:,} missing values to interpolate")
                total_gaps += gaps
            reindexed_dfs[var_name] = reindexed

        log(f"   Total gaps across all variables: {total_gaps:,}")

        # Combine all variables
        final_df = pd.concat(reindexed_dfs.values(), axis=1).sort_index()
        final_df = final_df[~final_df.index.duplicated(keep="first")]

        # Interpolate missing values using linear interpolation
        null_count_before = final_df.isna().sum().sum()
        if null_count_before > 0:
            log(f"\n   Interpolating {null_count_before:,} missing values...")

            # Use linear interpolation with time-based method
            final_df = final_df.interpolate(method='time', limit_direction='both')

            # Check for any remaining nulls
            null_count_after = final_df.isna().sum().sum()

            if null_count_after > 0:
                log(f"   ⚠️  {null_count_after:,} nulls remain after interpolation")
                log(f"   Filling remaining nulls with forward/backward fill...")
                final_df = final_df.fillna(method='ffill').fillna(method='bfill')

                # Final check
                null_count_final = final_df.isna().sum().sum()
                if null_count_final > 0:
                    log(f"   ⚠️  {null_count_final:,} nulls still remain - dropping these rows")
                    final_df = final_df.dropna(how='any')
                else:
                    log(f"   ✓ All nulls filled")
            else:
                log(f"   ✓ All missing values interpolated successfully")
        else:
            log(f"   ✓ No missing values (complete data)")

        log(f"\n✅ Combined all variables: {len(final_df):,} rows × {final_df.shape[1]} columns")
        log(f"   Date range: {final_df.index.min().strftime('%Y-%m-%d')} to {final_df.index.max().strftime('%Y-%m-%d')}")
        log(f"   Continuous: {'Yes' if len(final_df) == len(continuous_dates) else 'No (some days dropped)'}")

        return final_df, grid_info
    else:
        return pd.DataFrame(), grid_info


# ===== Process Datasets =====

def process_agcd_datasets(datasets: Dict[str, xr.Dataset],
                          land_mask: Optional[xr.DataArray] = None,
                          region_name: str = 'Australia') -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Process AGCD datasets with land-sea masking

    Returns:
        Tuple of (dataframe, ravenswood_cell, grid_info)
    """
    log("Processing AGCD datasets with land-sea masking...")

    if land_mask is not None:
        total_cells = land_mask.size
        land_cells = int(np.sum(~np.isnan(land_mask.values)))
        ocean_cells = total_cells - land_cells
        land_pct = (land_cells / total_cells) * 100

        log(f"   Grid cells: {land_cells:,} land ({land_pct:.1f}%) | {ocean_cells:,} ocean ({100 - land_pct:.1f}%)")
        log(f"   Using IPCC standard: cells with >50% land included in statistics")
    else:
        log("⚠️  WARNING: No land fraction data available")
        log("   Computing statistics over ALL grid cells")

    # Calculate tas (mean temperature) from tmax and tmin if available
    tas_ds = calculate_tas_from_tmax_tmin(datasets)
    if tas_ds is not None:
        datasets['tas'] = tas_ds

    data_dict = {}
    ravenswood_cell = None
    grid_info = None

    for var_name, ds in datasets.items():
        lat, lon = find_lat_lon(ds)
        ds = normalise_lon(ds, lon)
        da = ds[var_name]

        # Build grid info on first variable
        if grid_info is None:
            lats = ds[lat].values
            lons = ds[lon].values
            fingerprint, signature = calculate_grid_fingerprint(lats, lons)

            lat_idx = int(np.abs(lats - RAVENSWOOD_LAT).argmin())
            lon_idx = int(np.abs(lons - RAVENSWOOD_LON).argmin())

            actual_lat = float(lats[lat_idx])
            actual_lon = float(lons[lon_idx])

            lat_bounds = _calculate_cell_bounds_from_centres(lats)
            lon_bounds = _calculate_cell_bounds_from_centres(lons)

            lat_min = float(lat_bounds[lat_idx])
            lat_max = float(lat_bounds[lat_idx + 1])
            lon_min = float(lon_bounds[lon_idx])
            lon_max = float(lon_bounds[lon_idx + 1])

            grid_info = {
                "fingerprint": fingerprint,
                "signature": signature,
                "grid_shape": [len(lats), len(lons)],
                "lat_values": lats.tolist(),
                "lon_values": lons.tolist(),
                "lat_coord_name": lat,
                "lon_coord_name": lon,
                "ravenswood": {
                    "target": {"lat": RAVENSWOOD_LAT, "lon": RAVENSWOOD_LON},
                    "grid_indices": [int(lat_idx), int(lon_idx)],
                    "cell_centre": {"lat": actual_lat, "lon": actual_lon},
                    "cell_bounds": {"lat": [lat_min, lat_max], "lon": [lon_min, lon_max]},
                    "offset_from_target": {"lat": actual_lat - RAVENSWOOD_LAT, "lon": actual_lon - RAVENSWOOD_LON},
                    "cell_area_sq_deg": (lat_max - lat_min) * (lon_max - lon_min)
                }
            }

            # Add land mask info
            if land_mask is not None:
                land_cell_indices = []
                mask_array = land_mask.values

                for i_lat in range(len(lats)):
                    for i_lon in range(len(lons)):
                        if not np.isnan(mask_array[i_lat, i_lon]):
                            land_cell_indices.append([int(i_lat), int(i_lon)])

                grid_info["land_sea_mask"] = {
                    "applied": True,
                    "threshold_percent": 50.0,
                    "method": "IPCC AR6 standard",
                    "total_cells": int(land_mask.size),
                    "land_cells": len(land_cell_indices),
                    "ocean_cells": int(land_mask.size) - len(land_cell_indices),
                    "land_percentage": (len(land_cell_indices) / land_mask.size) * 100,
                    "land_cell_indices": land_cell_indices,
                    "description": "Grid cells with >50% land fraction included in regional statistics"
                }
            else:
                grid_info["land_sea_mask"] = {
                    "applied": False,
                    "description": "No land masking applied"
                }

        if ravenswood_cell is None:
            lat_actual = float(ds[lat].sel({lat: RAVENSWOOD_LAT}, method='nearest').values)
            lon_actual = float(ds[lon].sel({lon: RAVENSWOOD_LON}, method='nearest').values)

            if grid_info:
                ravenswood_cell = {
                    "centre": {"lat": round(lat_actual, 4), "lon": round(lon_actual, 4)},
                    "bounds": {
                        "lat": grid_info["ravenswood"]["cell_bounds"]["lat"],
                        "lon": grid_info["ravenswood"]["cell_bounds"]["lon"]
                    }
                }

        # Unit conversions
        if var_name in ['tas', 'tasmax', 'tasmin']:
            da = temp_to_degC(da)
            unit = "degC"
            display_name = var_name
        elif var_name == 'pr':
            da = precip_to_mm(da)
            unit = "mm_day"
            display_name = var_name
        elif var_name == 'vprp09':
            # Vapour pressure already in hPa
            unit = "hPa"
            display_name = var_name
        else:
            unit = "unknown"
            display_name = var_name

        # Regional mean WITH LAND MASKING
        log(f"   Computing regional mean for {display_name} (land-only)...")
        area_col = f"{display_name}_{region_name}_land_only_{unit}"
        area_mean_result = area_mean_land_only(da, lat, lon, land_mask)
        data_dict[area_col] = to_series(area_mean_result)
        log(f"   ✓ Regional mean complete")

        # Point extraction (unchanged)
        log(f"   Extracting {display_name} at Ravenswood...")
        point_col = f"{display_name}_Ravenswood_{unit}"
        data_dict[point_col] = to_series(extract_point(da, lat, lon, RAVENSWOOD_LAT, RAVENSWOOD_LON))
        log(f"   ✓ Point extraction complete")

        ds.close()
        del ds, da
        gc.collect()

    df = pd.DataFrame(data_dict).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    log(f"Final DataFrame: {len(df):,} rows × {df.shape[1]} columns")

    if land_mask is not None:
        log(f"✅ Regional statistics computed over LAND ONLY (IPCC >50% threshold)")

    return df, ravenswood_cell, grid_info


# ===== Main Workflow =====

def interactive_workflow():
    """Main interactive workflow"""
    log("🚀 Starting AGCD Importer...")
    log(f"Python: {sys.version.split()[0]}")
    log(f"Working directory: {Path.cwd()}")
    print()

    print("\n" + "=" * 70)
    print(" AGCD OBSERVATIONAL DATA IMPORTER")
    print("=" * 70)
    print("\n✨ Australian Gridded Climate Data (Observations)")
    print("  • Daily resolution historical records")
    print("  • IPCC-standard land-sea masking")
    print("  • Compatible with CMIP6 metrics generator")
    print("=" * 70)

    # Get AGCD data directory
    print("\n" + "=" * 70)
    print("AGCD DATA LOCATION")
    print("=" * 70)
    print("Enter path to directory containing AGCD NetCDF files")
    print("(Press Enter for default: ./agcd_data)")

    agcd_dir_input = input("> ").strip()
    if not agcd_dir_input:
        agcd_dir = BASE / "agcd_data"
    else:
        agcd_dir = Path(agcd_dir_input)

    if not agcd_dir.exists():
        print(f"\n❌ Directory not found: {agcd_dir}")
        print("\nExpected structure:")
        print("  agcd_data/")
        print("    tmax_daily_*.nc")
        print("    tmin_daily_*.nc")
        print("    precip_daily_*.nc")
        return 1

    log(f"Using AGCD directory: {agcd_dir}")

    # Select variables
    print("\n" + "=" * 70)
    print("AVAILABLE VARIABLES")
    print("=" * 70)
    for i, (key, info) in enumerate(AGCD_VARIABLES.items(), 1):
        print(f"{i}. {key:10s} | {info['unit']:15s} | {info['description']}")
    print("=" * 70)

    print("\nSelect variables (comma-separated, or 'all'):")
    print("Example: 1,2,3  or  all")

    var_choice = input("> ").strip().lower()

    if var_choice == 'all':
        selected_vars = list(AGCD_VARIABLES.keys())
    else:
        try:
            indices = [int(x.strip()) for x in var_choice.split(',')]
            var_keys = list(AGCD_VARIABLES.keys())
            selected_vars = [var_keys[i - 1] for i in indices if 1 <= i <= len(var_keys)]
        except (ValueError, IndexError):
            print("Invalid selection, using all variables")
            selected_vars = list(AGCD_VARIABLES.keys())

    print(f"\nSelected: {', '.join(selected_vars)}")

    # Year range
    print("\n" + "=" * 70)
    print("YEAR RANGE")
    print("=" * 70)
    print("Enter year range (or press Enter for all available data)")
    print("Format: start,end  (e.g., 1900,2023)")

    year_input = input("> ").strip()

    if year_input:
        try:
            start_year, end_year = map(int, year_input.split(','))
            year_range = (start_year, end_year)
            print(f"Year range: {start_year}-{end_year}")
        except ValueError:
            year_range = None
            print("Invalid format, using all available years")
    else:
        year_range = None
        print("Using all available years")

    # Bounding box (default to Australia)
    print("\n" + "=" * 70)
    print("GEOGRAPHICAL AREA")
    print("=" * 70)
    print("AGCD is Australian data only")
    print("Using full Australian domain")

    bbox = None  # AGCD is already Australia-only
    region_name = 'Australia'

    # Check for land mask
    print("\n" + "=" * 70)
    print("LAND-SEA MASKING")
    print("=" * 70)

    sftlf_path = find_local_sftlf(BASE)

    # Confirm
    print("\n" + "=" * 70)
    print("READY TO PROCESS")
    print("=" * 70)
    print(f"Source: AGCD (observational data)")
    print(f"Variables: {len(selected_vars)}")
    print(f"Directory: {agcd_dir.name}")
    if year_range:
        print(f"Years: {year_range[0]}-{year_range[1]}")
    else:
        print(f"Years: All available")
    print(f"Masking: {'Yes (IPCC >50%)' if sftlf_path else 'No'}")
    print("=" * 70)

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        return 0

    # Load land mask if available
    land_mask = None
    if sftlf_path:
        log("\nLoading land fraction data...")
        sftlf_ds = load_sftlf_from_file(sftlf_path, bbox)
        if sftlf_ds:
            lat, lon = find_lat_lon(sftlf_ds)
            sftlf_da = sftlf_ds['sftlf']
            land_mask = create_land_mask(sftlf_da, threshold=50.0)

    # Load and process datasets (sequential, no merge)
    log("\nProcessing AGCD data...")
    df, grid_info = load_and_process_agcd_sequential(
        agcd_dir, selected_vars, land_mask, region_name, year_range, bbox
    )

    if df.empty:
        log("❌ No data processed")
        return 1

    # Build ravenswood_cell from grid_info
    ravenswood_cell = None
    if grid_info and "ravenswood" in grid_info:
        ravenswood_cell = {
            "centre": grid_info["ravenswood"]["cell_centre"],
            "bounds": grid_info["ravenswood"]["cell_bounds"]
        }

    # Create output directory
    scenario_dir = ROOT_METRICS / "AGCD"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    out_parq = scenario_dir / "raw_daily.parquet"
    out_json = scenario_dir / "raw_daily.json"

    # Ensure index is named 'time' for metrics generator compatibility
    df.index.name = 'time'

    # Save parquet
    log("\nSaving data to parquet file (this may take 1-2 minutes)...")
    df.to_parquet(out_parq, engine="pyarrow", compression="zstd", index=True)
    log(f"✓ Saved: {out_parq}")
    log(f"  Size: {out_parq.stat().st_size / (1024 ** 2):.1f} MB")

    # Build metadata
    variables_dict = {}
    for col in df.columns:
        if '_Ravenswood_' in col:
            parts = col.split('_')
            var_name = parts[0]
            unit = parts[-1]
            variables_dict[var_name] = unit

    columns_map = {}
    for col in df.columns:
        if '_Ravenswood_' in col:
            var_name = col.split('_')[0]
            columns_map[var_name] = col

    # Calendar (AGCD uses standard Gregorian calendar)
    calendar = 'standard'

    meta = {
        "model": "AGCD",
        "experiment": "observations",
        "data_type": "observational",
        "source_description": "Australian Gridded Climate Data (AGCD) from Bureau of Meteorology",
        "variant": "observations",
        "grid": "0.05deg",
        "table_id": "daily",
        "calendar": calendar,
        "created": datetime.utcnow().isoformat() + "Z",
        "scenario": "AGCD",

        "land_sea_masking": {
            "applied": land_mask is not None,
            "method": "IPCC AR6 standard" if land_mask is not None else "None",
            "threshold": "50% land" if land_mask is not None else "N/A",
            "description": "Regional statistics computed over land grid cells only (>50% land fraction)" if land_mask is not None else "No land masking applied",
            "source": "sftlf (percentage_of_the_grid_cell_occupied_by_land)" if land_mask is not None else "N/A"
        },

        "source": {
            "method": "Local NetCDF files",
            "dataset": "AGCD (Australian Gridded Climate Data)",
            "provider": "Bureau of Meteorology, Australia",
            "processed": datetime.utcnow().isoformat() + "Z",
            "source_directory": str(agcd_dir),
            "variables": selected_vars,
            "year_range": list(year_range) if year_range else "all available"
        },

        "lat_coord": "lat",
        "lon_coord": "lon",

        "ravenswood_target": {
            "lat": RAVENSWOOD_LAT,
            "lon": RAVENSWOOD_LON
        },

        "ravenswood_cell": ravenswood_cell,

        "geographical_area": {
            "description": "Australian domain (AGCD coverage)"
        },

        "grid_info": grid_info,

        "time_range": [
            df.index.min().strftime('%Y-%m-%d') if hasattr(df.index.min(), 'strftime') else str(df.index.min()),
            df.index.max().strftime('%Y-%m-%d') if hasattr(df.index.max(), 'strftime') else str(df.index.max())
        ],

        "temporal_resolution": "daily",

        "date": {
            "column": "time",
            "format": "YYYY-MM-DD",
            "freq": "D"
        },

        "variables": variables_dict,

        "columns": columns_map,

        "parquet": {
            "file": out_parq.name,
            "rows": len(df),
            "columns": len(df.columns),
            "index_name": "time",
            "column_order": list(df.columns)
        }
    }

    log("Saving metadata...")
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"✓ Saved: {out_json}")

    log(f"\n✅ All files saved to: {scenario_dir}")
    log(f"   Metadata: {out_json.name}")
    log(f"   Directory: {scenario_dir}")

    print("\n" + "=" * 70)
    print("✅ AGCD IMPORT COMPLETE")
    print("=" * 70)
    print(f"\nOutput location: {scenario_dir}")
    print(f"Files created:")
    print(f"  • {out_parq.name}")
    print(f"  • {out_json.name}")
    print("\nThis scenario can now be processed by the metrics generator")
    print("alongside CMIP6 scenarios.")
    print("=" * 70 + "\n")

    return 0


def main():
    """Main entry point"""
    return interactive_workflow()


if __name__ == "__main__":
    sys.exit(main())