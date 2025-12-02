#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Configuration Utility
==============================
climateConfig.py
Last Updated: 2025-11-28 17:15 AEST - Per-location amplification factors

Interactive menu-driven utility for managing Climate Viewer configuration.

Run in PyCharm console:
    python climateConfig.py

This utility:
1. Discovers available scenarios from metricsDataFiles folder
2. Manages all configurable settings
3. Generates pre-calculated values (warming thresholds, alignment offsets)
4. Calculates per-location amplification factors from global/regional NetCDF data
5. Saves everything to climate_config.json

Location: Utilities_Scenario/climateConfig.py
Output:   ../climate_config.json (in climateData root)
Data:     ../metricsDataFiles/

Amplification Calculation:
- Global tas: netcdf_global/tas_Amon_ACCESS-CM2_ssp*.nc
- Regional tas: metricsDataFiles/SSP1-26/netcdf/*tas*.nc
- Land mask: australia_grid_coverage.kml (296 cells)
- Method: Regional warming (2015-2099) / Global warming (2015-2099)
- Calculates separate factor for each location (Australia, Ravenswood)
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Missing dependencies.  Run: pip install pandas numpy pyarrow")

# xarray is optional - only needed for amplification calculation
try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# ============================================================================
# PATHS
# ============================================================================

# This script lives in Utilities_Scenario/
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent  # climateData/
METRICS_FOLDER = BASE_DIR / "metricsDataFiles"
CONFIG_FILE = BASE_DIR / "climate_config.json"

# NetCDF paths for amplification calculation
NETCDF_GLOBAL_FOLDER = SCRIPT_DIR / "netcdf_global"
KML_PATH = BASE_DIR / "australia_grid_coverage.kml"


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def clear_screen():
    """Clear console screen (safe for PyCharm)."""
    # Check if running in a proper terminal
    if os.environ.get('TERM') or os.name == 'nt':
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        # PyCharm or other IDE - just print newlines
        print("\n" * 3)


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a subsection header."""
    print()
    print("-" * 60)
    print(f"  {title}")
    print("-" * 60)


def pause():
    """Wait for user to press Enter."""
    input("\n[Press Enter to continue]")


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default."""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def confirm(prompt: str) -> bool:
    """Get yes/no confirmation."""
    response = input(f"{prompt} [y/n]: ").strip().lower()
    return response in ('y', 'yes')


def format_value(value: Any) -> str:
    """Format a value for display."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def parse_value(value_str: str, current_value: Any) -> Any:
    """Parse string input to match type of current value."""
    if isinstance(current_value, bool):
        return value_str.lower() in ('true', 'yes', 'y', '1')
    if isinstance(current_value, int):
        return int(value_str)
    if isinstance(current_value, float):
        return float(value_str)
    if isinstance(current_value, list):
        # For lists, return the string (handled separately)
        return value_str
    return value_str


# ============================================================================
# SCENARIO DISCOVERY
# ============================================================================

def discover_scenarios() -> Dict[str, Dict]:
    """
    Discover available scenarios from metricsDataFiles folder.

    Returns:
        Dict mapping scenario name -> metadata
    """
    scenarios = {}

    if not METRICS_FOLDER.is_dir():
        return scenarios

    for folder in sorted(METRICS_FOLDER.iterdir()):
        if not folder.is_dir():
            continue

        # Skip hidden folders
        if folder.name.startswith('.') or folder.name.startswith('_'):
            continue

        # Find metrics parquet file
        parquet_file = None
        for f in folder.iterdir():
            if f.name.startswith("metrics") and f.suffix == ".parquet":
                parquet_file = f
                break

        if parquet_file is None:
            continue

        # Get basic info
        scenario_name = folder.name

        # Determine type
        if scenario_name.upper() == "AGCD":
            scen_type = "observations"
        elif scenario_name.lower() == "historical":
            scen_type = "historical"
        elif scenario_name.upper().startswith("SSP"):
            scen_type = "projection"
        else:
            scen_type = "other"

        scenarios[scenario_name] = {
            "path": str(parquet_file),
            "type": scen_type,
            "folder": str(folder)
        }

    return scenarios


def load_scenario_details(scenario_path: str) -> Dict:
    """Load detailed info about a scenario."""
    df = pd.read_parquet(scenario_path, engine="pyarrow")

    # Handle Metric vs Name column
    if "Metric" in df.columns and "Name" not in df.columns:
        df = df.rename(columns={"Metric": "Name"})

    return {
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max()),
        "locations": sorted(df["Location"].unique().tolist()),
        "types": sorted(df["Type"].unique().tolist()),
        "rows": len(df)
    }


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def get_default_config() -> Dict:
    """Return default configuration structure."""
    return {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "generator": "climateConfig.py",
            "version": "1.1"
        },
        "parameters": {
            "baseline_period_start": 1850,
            "baseline_period_end": 1900,
            "amplification_factors": {
                "Australia": 1.22,
                "Ravenswood": 1.20
            },
            "global_thresholds": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            "alignment_year": 2014
        },
        "defaults": {
            "scenarios_on": ["SSP1-26", "SSP3-70"],
            "scenarios_off": ["historical", "AGCD", "SSP2-45", "SSP5-85"],
            "start_year": 2020,
            "end_year": 2050,
            "year_buffer": 10,
            "smoothing_enabled": True,
            "smoothing_window": 20,
            "bc_enabled": True,
            "align_to_agcd": True,
            "show_15c_target": True,
            "show_time_horizons": True
        },
        "time_horizons": {
            "short_start": 2020,
            "mid_start": 2036,
            "long_start": 2039,
            "horizon_end": 2045
        },
        "scenario_info": {},
        "warming_thresholds": {},
        "alignment_offsets": {}
    }


def load_config() -> Dict:
    """Load config from JSON file, or return defaults if not found."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Ensure all sections exist
        default = get_default_config()
        for key in default:
            if key not in config:
                config[key] = default[key]
        return config
    return get_default_config()


def save_config(config: Dict):
    """Save config to JSON file."""
    config["metadata"]["generated"] = datetime.now().isoformat()

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved: {CONFIG_FILE}")


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def calculate_warming_thresholds(config: Dict) -> Dict:
    """Calculate warming thresholds from Historical raw data."""
    print("\nCalculating warming thresholds...")

    # Find Historical scenario
    scenarios = discover_scenarios()
    hist_path = None
    for name in ["historical", "Historical"]:
        if name in scenarios:
            hist_path = scenarios[name]["path"]
            break

    if hist_path is None:
        print("  ERROR: Historical scenario not found")
        return {}

    # Load raw data (not BC)
    df = pd.read_parquet(hist_path, engine="pyarrow")
    if "Metric" in df.columns and "Name" not in df.columns:
        df = df.rename(columns={"Metric": "Name"})

    print(f"  Loaded: {len(df):,} rows")

    # Get parameters
    baseline_start = config["parameters"]["baseline_period_start"]
    baseline_end = config["parameters"]["baseline_period_end"]
    thresholds = config["parameters"]["global_thresholds"]

    # Get Australia amplification factor (used for all locations)
    amp_factors = config["parameters"].get("amplification_factors", {})
    amp_factor = amp_factors.get("Australia", config["parameters"].get("amplification_factor", 1.15))
    print(f"  Using Australia amplification factor: {amp_factor}")

    # Get locations
    locations = sorted(df["Location"].unique())
    result = {}

    for location in locations:
        # Calculate pre-industrial baseline
        baseline_data = df[
            (df["Year"] >= baseline_start) &
            (df["Year"] <= baseline_end) &
            (df["Location"] == location) &
            (df["Type"] == "Temp") &
            (df["Name"] == "Average") &
            (df["Season"] == "Annual")
            ]

        if baseline_data.empty:
            print(f"  {location}: No baseline data")
            continue

        baseline = baseline_data["Value"].mean()
        year_count = baseline_data["Year"].nunique()

        # Calculate thresholds using Australia amplification
        thresh_values = {}
        for global_warming in thresholds:
            regional_warming = global_warming * amp_factor
            regional_temp = baseline + regional_warming
            thresh_values[str(global_warming)] = round(regional_temp, 3)

        result[location] = {
            "preindustrial_baseline": round(baseline, 3),
            "baseline_years": year_count,
            "amplification_factor": amp_factor,
            "thresholds": thresh_values
        }

        print(f"  {location}: baseline {baseline:.3f}°C, 1.5°C threshold = {thresh_values.get('1.5', 'N/A')}°C")

    return result


def calculate_alignment_offsets(config: Dict) -> Dict:
    """Calculate AGCD alignment offsets."""
    print("\nCalculating alignment offsets...")

    scenarios = discover_scenarios()

    if "AGCD" not in scenarios:
        print("  ERROR: AGCD scenario not found")
        return {}

    alignment_year = config["parameters"]["alignment_year"]

    # Load AGCD data
    agcd_df = pd.read_parquet(scenarios["AGCD"]["path"], engine="pyarrow")
    if "Metric" in agcd_df.columns and "Name" not in agcd_df.columns:
        agcd_df = agcd_df.rename(columns={"Metric": "Name"})

    # Get AGCD values at alignment year
    agcd_at_year = agcd_df[agcd_df["Year"] == alignment_year].copy()

    if agcd_at_year.empty:
        print(f"  ERROR: No AGCD data at year {alignment_year}")
        return {}

    # Build AGCD lookup
    agcd_lookup = {}
    for _, row in agcd_at_year.iterrows():
        key = (row["Location"], row["Type"], row["Name"], row["Season"])
        agcd_lookup[key] = row["Value"]

    print(f"  AGCD reference: {len(agcd_lookup)} metrics at {alignment_year}")

    result = {}

    # Calculate offsets for each scenario
    for scenario_name, scenario_info in scenarios.items():
        if scenario_name == "AGCD":
            continue

        # Determine alignment year for this scenario
        is_ssp = scenario_name.upper().startswith("SSP")
        scen_align_year = 2015 if is_ssp else alignment_year

        # Load scenario data
        scen_df = pd.read_parquet(scenario_info["path"], engine="pyarrow")
        if "Metric" in scen_df.columns and "Name" not in scen_df.columns:
            scen_df = scen_df.rename(columns={"Metric": "Name"})

        scen_at_year = scen_df[scen_df["Year"] == scen_align_year].copy()

        if scen_at_year.empty:
            print(f"  {scenario_name}: No data at year {scen_align_year}")
            continue

        scenario_offsets = {}
        offset_count = 0

        for _, row in scen_at_year.iterrows():
            key = (row["Location"], row["Type"], row["Name"], row["Season"])

            if key not in agcd_lookup:
                continue

            agcd_value = agcd_lookup[key]
            scen_value = row["Value"]
            offset = agcd_value - scen_value

            if abs(offset) > 0.001:
                key_str = f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"
                scenario_offsets[key_str] = round(offset, 4)
                offset_count += 1

        if scenario_offsets:
            result[scenario_name] = {
                "align_to_year": scen_align_year,
                "reference_year": alignment_year,
                "offsets": scenario_offsets
            }

        print(f"  {scenario_name}: {offset_count} offsets")

    return result


def update_scenario_info(config: Dict) -> Dict:
    """Update scenario info from discovered scenarios."""
    print("\nDiscovering scenarios...")

    scenarios = discover_scenarios()
    result = {}

    descriptions = {
        "historical": "Historical model simulation (1850-2014)",
        "Historical": "Historical model simulation (1850-2014)",
        "AGCD": "Australian Gridded Climate Data (observations)",
        "SSP1-26": "Sustainability pathway, ~1.8°C by 2100",
        "SSP1-1.9": "Very low emissions, ~1.5°C by 2100",
        "SSP2-45": "Middle of the road, ~2.7°C by 2100",
        "SSP3-70": "Regional rivalry, ~3.6°C by 2100",
        "SSP5-85": "Fossil-fuelled development, ~4.4°C by 2100"
    }

    for scenario_name, scenario_info in scenarios.items():
        details = load_scenario_details(scenario_info["path"])

        result[scenario_name] = {
            "type": scenario_info["type"],
            "years": [details["year_min"], details["year_max"]],
            "locations": details["locations"],
            "metric_types": details["types"],
            "row_count": details["rows"],
            "description": descriptions.get(scenario_name, "Climate scenario")
        }

        print(f"  {scenario_name}: {details['year_min']}-{details['year_max']}, {details['rows']:,} rows")

    return result


# ============================================================================
# AMPLIFICATION FACTOR CALCULATION
# ============================================================================

def parse_land_cells_from_kml(kml_path: Path) -> List[Tuple[int, int]]:
    """
    Parse grid cell indices from the KML file.

    Returns list of (lat_idx, lon_idx) tuples for land cells.
    """
    land_cells = []

    if not kml_path.exists():
        print(f"  [!] KML file not found: {kml_path}")
        return land_cells

    with open(kml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract cell indices from <n>Cell [lat_idx, lon_idx]</n> tags
    pattern = r"<n>Cell \[(\d+), (\d+)\]</n>"
    matches = re.findall(pattern, content)

    if not matches:
        # Try alternative pattern with <name> tag
        pattern_alt = r"<name>Cell \[(\d+), (\d+)\]</name>"
        matches = re.findall(pattern_alt, content)

    for lat_idx, lon_idx in matches:
        land_cells.append((int(lat_idx), int(lon_idx)))

    if not land_cells:
        print(f"  [!] No cell patterns found in KML: {kml_path}")
        print(f"      File size: {kml_path.stat().st_size} bytes")

    return land_cells


def find_netcdf_files() -> Dict[str, Optional[Path]]:
    """Find global and regional NetCDF files for amplification calculation."""
    result = {"global": None, "regional": None, "regional_type": None}

    # Find global tas file
    if NETCDF_GLOBAL_FOLDER.exists():
        for f in NETCDF_GLOBAL_FOLDER.glob("tas_Amon_ACCESS-CM2_ssp*.nc"):
            result["global"] = f
            break

    # Find regional tas file (in SSP1-26 netcdf folder)
    # Prefer tas (average temp), but can use tasmax if tas not available
    regional_folder = METRICS_FOLDER / "SSP1-26" / "netcdf"
    if regional_folder.exists():
        # First look for tas specifically (not tasmax, tasmin)
        for f in regional_folder.glob("*_tas_*.nc"):
            result["regional"] = f
            result["regional_type"] = "tas"
            break

        # If no tas found, look for tasmax as fallback
        if result["regional"] is None:
            for f in regional_folder.glob("*_tasmax_*.nc"):
                result["regional"] = f
                result["regional_type"] = "tasmax"
                break

    return result


def calculate_amplification_factors(config: Dict) -> Optional[Dict[str, float]]:
    """
    Calculate regional warming amplification factors from NetCDF data.

    Amplification = Regional warming / Global warming

    Calculates for:
    - Australia (all land cells, area-weighted)
    - Ravenswood (single grid cell)

    Uses:
    - Global monthly tas data (area-weighted global mean)
    - Regional daily tas data
    - 10-year averages at start and end of period

    Returns:
        Dict mapping location name to amplification factor
    """
    if not XARRAY_AVAILABLE:
        print("\n  [!] xarray not available.  Install with: pip install xarray netcdf4")
        return None

    # Location definitions - just calculate Australia (continental average)
    # Single grid cells are too noisy; use continental value for all locations
    LOCATIONS = {
        "Australia": {"type": "mask"}  # Uses land mask - 295 cells
    }

    print(f"  Calculating continental Australia amplification (295 land cells)")

    print("\nCalculating amplification factors from NetCDF data...")

    # Find NetCDF files
    nc_files = find_netcdf_files()

    if nc_files["global"] is None:
        print(f"  [!] Global tas file not found in: {NETCDF_GLOBAL_FOLDER}")
        print("      Expected: tas_Amon_ACCESS-CM2_ssp*.nc")
        return None

    if nc_files["regional"] is None:
        print(f"  [!] Regional tas file not found in: {METRICS_FOLDER / 'SSP1-26' / 'netcdf'}")
        return None

    print(f"  Global:   {nc_files['global'].name}")
    print(f"  Regional: {nc_files['regional'].name}")

    # Parse land cells from KML for Australia mask
    land_cells = parse_land_cells_from_kml(KML_PATH)
    if land_cells:
        print(f"  Land mask: {len(land_cells)} cells from KML")
    else:
        print("  [!] No land mask - Australia calculation will use all cells")

    # Analysis period
    start_year = 2015
    end_year = 2099

    try:
        # ================================================================
        # GLOBAL MEAN TEMPERATURE
        # ================================================================
        print("\n  Processing global data...")
        ds_global = xr.open_dataset(nc_files["global"])
        tas_global = ds_global["tas"]

        # Convert Kelvin to Celsius if needed
        if float(tas_global.mean()) > 200:
            tas_global = tas_global - 273.15

        # Area weights (cosine of latitude)
        lat = ds_global["lat"].values
        weights = np.cos(np.deg2rad(lat))
        weights_da = xr.DataArray(weights, dims=["lat"], coords={"lat": lat})

        # Weighted global mean
        tas_weighted = tas_global.weighted(weights_da)
        global_mean = tas_weighted.mean(dim=["lat", "lon"])

        # Extract years
        time = ds_global["time"].values
        years = np.array([int(str(t)[:4]) for t in time])

        # Calculate annual means
        global_annual = {}
        for year in range(start_year, end_year + 1):
            year_mask = years == year
            if year_mask.any():
                global_annual[year] = float(global_mean.values[year_mask].mean())

        # 10-year averages
        global_start = np.mean([global_annual[y] for y in range(start_year, start_year + 10) if y in global_annual])
        global_end = np.mean([global_annual[y] for y in range(end_year - 9, end_year + 1) if y in global_annual])
        global_warming = global_end - global_start

        ds_global.close()
        print(f"    Warming: {global_start:.2f}°C -> {global_end:.2f}°C = {global_warming:.3f}°C")

        if global_warming <= 0:
            print("  [!] No global warming detected")
            return None

        # ================================================================
        # REGIONAL DATA SETUP
        # ================================================================
        print("\n  Processing regional data...")
        ds_regional = xr.open_dataset(nc_files["regional"])

        # Use correct variable name based on what file was found
        var_name = nc_files.get("regional_type", "tas")
        if var_name == "tasmax":
            print(f"    Note: Using tasmax (no tas file found)")
        tas_regional = ds_regional[var_name]

        # Convert Kelvin to Celsius if needed
        if float(tas_regional.mean()) > 200:
            tas_regional = tas_regional - 273.15

        lat_reg = ds_regional["lat"].values
        lon_reg = ds_regional["lon"].values
        time_reg = ds_regional["time"].values
        years_reg = np.array([int(str(t)[:4]) for t in time_reg])

        # Diagnostic: show coordinate system
        lat_direction = "S→N" if lat_reg[0] < lat_reg[-1] else "N→S"
        print(f"    Grid: {len(lat_reg)} lat × {len(lon_reg)} lon, lat runs {lat_direction}")
        print(f"    Lat range: {lat_reg[0]:.2f}° to {lat_reg[-1]:.2f}°")
        print(f"    Lon range: {lon_reg[0]:.2f}° to {lon_reg[-1]:.2f}°")

        # Area weights for regional data
        weights_reg = np.cos(np.deg2rad(lat_reg))
        weights_reg_da = xr.DataArray(weights_reg, dims=["lat"], coords={"lat": lat_reg})

        # ================================================================
        # CALCULATE AMPLIFICATION FOR EACH LOCATION
        # ================================================================
        results = {}

        for loc_name, loc_info in LOCATIONS.items():
            print(f"\n  {loc_name}:")

            if loc_info["type"] == "point":
                # Single grid cell - find nearest by lat/lon
                target_lat = loc_info["lat"]
                target_lon = loc_info["lon"]

                # Find nearest grid cell
                lat_idx = int(np.abs(lat_reg - target_lat).argmin())
                lon_idx = int(np.abs(lon_reg - target_lon).argmin())

                actual_lat = lat_reg[lat_idx]
                actual_lon = lon_reg[lon_idx]
                print(f"    Grid cell: [{lat_idx}, {lon_idx}] at ({actual_lat:.2f}°, {actual_lon:.2f}°)")

                # Extract single cell time series
                cell_data = tas_regional[:, lat_idx, lon_idx]

                # Calculate annual means
                regional_annual = {}
                for year in range(start_year, end_year + 1):
                    year_mask = years_reg == year
                    if year_mask.any():
                        vals = cell_data.values[year_mask]
                        regional_annual[year] = float(np.nanmean(vals))

            elif loc_info["type"] == "cell":
                # Direct grid indices specified
                lat_idx, lon_idx = loc_info["grid_idx"]

                actual_lat = lat_reg[lat_idx]
                actual_lon = lon_reg[lon_idx]
                print(f"    Grid cell: [{lat_idx}, {lon_idx}] at ({actual_lat:.2f}°, {actual_lon:.2f}°)")

                # Extract single cell time series
                cell_data = tas_regional[:, lat_idx, lon_idx]

                # Calculate annual means
                regional_annual = {}
                for year in range(start_year, end_year + 1):
                    year_mask = years_reg == year
                    if year_mask.any():
                        vals = cell_data.values[year_mask]
                        regional_annual[year] = float(np.nanmean(vals))

            elif loc_info["type"] == "mask":
                # Use land mask for Australia
                if land_cells:
                    mask = np.zeros((len(lat_reg), len(lon_reg)), dtype=bool)
                    for lat_idx, lon_idx in land_cells:
                        if lat_idx < len(lat_reg) and lon_idx < len(lon_reg):
                            mask[lat_idx, lon_idx] = True
                    mask_da = xr.DataArray(mask, dims=["lat", "lon"],
                                           coords={"lat": lat_reg, "lon": lon_reg})
                    tas_masked = tas_regional.where(mask_da)
                    print(f"    Using {mask.sum()} land cells")
                else:
                    tas_masked = tas_regional
                    print(f"    Using all cells (no mask)")

                # Weighted mean over masked region
                tas_weighted_reg = tas_masked.weighted(weights_reg_da)
                regional_mean = tas_weighted_reg.mean(dim=["lat", "lon"], skipna=True)

                # Calculate annual means
                regional_annual = {}
                for year in range(start_year, end_year + 1):
                    year_mask = years_reg == year
                    if year_mask.any():
                        vals = regional_mean.values[year_mask]
                        regional_annual[year] = float(np.nanmean(vals))

            # 10-year averages
            regional_start = np.mean(
                [regional_annual[y] for y in range(start_year, start_year + 10) if y in regional_annual])
            regional_end = np.mean(
                [regional_annual[y] for y in range(end_year - 9, end_year + 1) if y in regional_annual])
            regional_warming = regional_end - regional_start

            # Calculate amplification
            amplification = regional_warming / global_warming
            results[loc_name] = round(amplification, 3)

            print(f"    Warming: {regional_start:.2f}°C -> {regional_end:.2f}°C = {regional_warming:.3f}°C")
            print(f"    Amplification: {amplification:.3f}")

        ds_regional.close()

        # ================================================================
        # SUMMARY
        # ================================================================
        print(f"\n  {'=' * 50}")
        print(f"  AMPLIFICATION FACTORS")
        print(f"  {'=' * 50}")
        for loc_name, amp in results.items():
            print(f"    {loc_name:<15}: {amp:.3f}")
        print(f"\n  Global warming (SSP1-2.6): {global_warming:.3f}°C")

        return results

    except Exception as e:
        print(f"  [!] Error calculating amplification: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MENU SCREENS
# ============================================================================

def menu_main(config: Dict) -> Tuple[str, Dict]:
    """Main menu."""
    while True:
        clear_screen()
        print_header("CLIMATE CONFIGURATION UTILITY")

        print(f"\nConfig file: {CONFIG_FILE}")
        print(f"Data folder: {METRICS_FOLDER}")

        scenarios = discover_scenarios()
        print(f"Scenarios found: {len(scenarios)}")

        print("\nMain Menu:")
        print("  1. Parameters (baseline period, amplification, thresholds)")
        print("  2. Defaults (scenarios, years, toggles)")
        print("  3. Time Horizons")
        print("  4. Scenario Info (view/update)")
        print("  5. Warming Thresholds (view/regenerate)")
        print("  6. Alignment Offsets (view/regenerate)")
        print("  7. Amplification Factor (view/calculate)")
        print("  8. Regenerate ALL calculated values")
        print()
        print("  S. Save configuration")
        print("  Q. Quit")

        choice = get_input("\nSelect").upper()

        if choice == "1":
            config = menu_parameters(config)
        elif choice == "2":
            config = menu_defaults(config, scenarios)
        elif choice == "3":
            config = menu_time_horizons(config)
        elif choice == "4":
            config = menu_scenario_info(config)
        elif choice == "5":
            config = menu_warming_thresholds(config)
        elif choice == "6":
            config = menu_alignment_offsets(config)
        elif choice == "7":
            config = menu_amplification(config)
        elif choice == "8":
            config = menu_regenerate_all(config)
        elif choice == "S":
            save_config(config)
            pause()
        elif choice == "Q":
            if confirm("Save before quitting?"):
                save_config(config)
            return "quit", config

    return "continue", config


def menu_parameters(config: Dict) -> Dict:
    """Parameters submenu."""
    params = config["parameters"]

    while True:
        clear_screen()
        print_header("PARAMETERS")

        items = [
            ("baseline_period_start", "Baseline period start"),
            ("baseline_period_end", "Baseline period end"),
            ("amplification_factor", "Amplification factor"),
            ("global_thresholds", "Global thresholds"),
            ("alignment_year", "Alignment year")
        ]

        for i, (key, label) in enumerate(items, 1):
            value = params.get(key, "N/A")
            print(f"  {i}. {label:<25}: {format_value(value)}")

        print()
        print("  0. Back to main menu")

        choice = get_input("\nSelect [0-5]")

        if choice == "0":
            return config

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                key, label = items[idx]
                current = params[key]

                print(f"\nCurrent: {label} = {format_value(current)}")

                if isinstance(current, list):
                    print("Enter values separated by commas")
                    new_str = get_input("New value")
                    if new_str:
                        new_value = [float(x.strip()) for x in new_str.split(",")]
                        print(f"\n  Current: {current}")
                        print(f"  New:     {new_value}")
                        if confirm("\nProceed with change?"):
                            params[key] = new_value
                            print(f"\nUpdated {label}")
                else:
                    new_str = get_input("New value")
                    if new_str:
                        new_value = parse_value(new_str, current)
                        print(f"\n  Current: {current}")
                        print(f"  New:     {new_value}")
                        if confirm("\nProceed with change?"):
                            params[key] = new_value
                            print(f"\nUpdated {label}")

                pause()
        except (ValueError, IndexError):
            pass

    return config


def menu_defaults(config: Dict, scenarios: Dict) -> Dict:
    """Defaults submenu."""
    defaults = config["defaults"]
    available = list(scenarios.keys())

    while True:
        clear_screen()
        print_header("DEFAULTS")

        print(f"\nAvailable scenarios: {', '.join(available)}")

        items = [
            ("scenarios_on", "Scenarios ON by default"),
            ("scenarios_off", "Scenarios OFF by default"),
            ("start_year", "Start year"),
            ("end_year", "End year"),
            ("year_buffer", "Year buffer"),
            ("smoothing_enabled", "Smoothing enabled"),
            ("smoothing_window", "Smoothing window (years)"),
            ("bc_enabled", "Bias correction enabled"),
            ("align_to_agcd", "Align to AGCD"),
            ("show_15c_target", "Show 1.5°C target"),
            ("show_time_horizons", "Show time horizons")
        ]

        print()
        for i, (key, label) in enumerate(items, 1):
            value = defaults.get(key, "N/A")
            print(f"  {i:2}. {label:<28}: {format_value(value)}")

        print()
        print("   0. Back to main menu")

        choice = get_input("\nSelect [0-11]")

        if choice == "0":
            return config

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                key, label = items[idx]
                current = defaults[key]

                print(f"\nCurrent: {label} = {format_value(current)}")

                if key in ("scenarios_on", "scenarios_off"):
                    print(f"\nAvailable: {', '.join(available)}")
                    print("\nOptions:")
                    print("  A. Add a scenario")
                    print("  R. Remove a scenario")
                    print("  S. Set entire list")
                    print("  C. Cancel")

                    sub = get_input("\nSelect [A/R/S/C]").upper()

                    if sub == "A":
                        scen = get_input("Scenario to add")
                        if scen and scen not in current:
                            new_value = current + [scen]
                            print(f"\n  Current: {current}")
                            print(f"  New:     {new_value}")
                            if confirm("\nProceed?"):
                                defaults[key] = new_value
                                print(f"\nAdded {scen}")
                    elif sub == "R":
                        scen = get_input("Scenario to remove")
                        if scen and scen in current:
                            new_value = [s for s in current if s != scen]
                            print(f"\n  Current: {current}")
                            print(f"  New:     {new_value}")
                            if confirm("\nProceed?"):
                                defaults[key] = new_value
                                print(f"\nRemoved {scen}")
                    elif sub == "S":
                        new_str = get_input("Enter scenarios (comma-separated)")
                        if new_str:
                            new_value = [s.strip() for s in new_str.split(",")]
                            print(f"\n  Current: {current}")
                            print(f"  New:     {new_value}")
                            if confirm("\nProceed?"):
                                defaults[key] = new_value
                                print("\nUpdated list")

                elif isinstance(current, bool):
                    new_value = not current
                    print(f"\n  Current: {format_value(current)}")
                    print(f"  New:     {format_value(new_value)}")
                    if confirm("\nToggle value?"):
                        defaults[key] = new_value
                        print(f"\nUpdated {label}")

                else:
                    new_str = get_input("New value")
                    if new_str:
                        new_value = parse_value(new_str, current)
                        print(f"\n  Current: {current}")
                        print(f"  New:     {new_value}")
                        if confirm("\nProceed with change?"):
                            defaults[key] = new_value
                            print(f"\nUpdated {label}")

                pause()
        except (ValueError, IndexError):
            pass

    return config


def menu_time_horizons(config: Dict) -> Dict:
    """Time horizons submenu."""
    horizons = config["time_horizons"]

    while True:
        clear_screen()
        print_header("TIME HORIZONS")

        items = [
            ("short_start", "Short term start"),
            ("mid_start", "Mid term start"),
            ("long_start", "Long term start"),
            ("horizon_end", "Horizon end")
        ]

        print()
        for i, (key, label) in enumerate(items, 1):
            value = horizons.get(key, "N/A")
            print(f"  {i}. {label:<20}: {value}")

        print()
        print("  0. Back to main menu")

        choice = get_input("\nSelect [0-4]")

        if choice == "0":
            return config

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                key, label = items[idx]
                current = horizons[key]

                print(f"\nCurrent: {label} = {current}")
                new_str = get_input("New value (year)")

                if new_str:
                    new_value = int(new_str)
                    print(f"\n  Current: {current}")
                    print(f"  New:     {new_value}")
                    if confirm("\nProceed with change?"):
                        horizons[key] = new_value
                        print(f"\nUpdated {label}")

                pause()
        except (ValueError, IndexError):
            pass

    return config


def menu_scenario_info(config: Dict) -> Dict:
    """Scenario info submenu."""
    clear_screen()
    print_header("SCENARIO INFO")

    info = config.get("scenario_info", {})

    if not info:
        print("\nNo scenario info stored.")
        print("Run 'Regenerate' to discover scenarios.")
    else:
        print(f"\n{'Scenario':<15} {'Type':<12} {'Years':<15} {'Rows':>10}")
        print("-" * 55)
        for name, data in sorted(info.items()):
            years = f"{data['years'][0]}-{data['years'][1]}"
            rows = f"{data['row_count']:,}"
            print(f"{name:<15} {data['type']:<12} {years:<15} {rows:>10}")

    print("\n  R. Regenerate scenario info")
    print("  0. Back to main menu")

    choice = get_input("\nSelect").upper()

    if choice == "R":
        config["scenario_info"] = update_scenario_info(config)
        pause()

    return config


def menu_warming_thresholds(config: Dict) -> Dict:
    """Warming thresholds submenu."""
    clear_screen()
    print_header("WARMING THRESHOLDS")

    thresholds = config.get("warming_thresholds", {})
    params = config["parameters"]

    print(f"\nBaseline period: {params['baseline_period_start']}-{params['baseline_period_end']}")

    # Handle both old single value and new per-location dict
    amp_factors = params.get("amplification_factors", {})
    amp_factor = amp_factors.get("Australia", params.get("amplification_factor", 1.15))
    print(f"Amplification factor (Australia): {amp_factor}")
    print(f"Global thresholds: {params['global_thresholds']}")

    if not thresholds:
        print("\nNo thresholds calculated.")
        print("Run 'Regenerate' to calculate from Historical data.")
    else:
        # Build header
        thresh_keys = params['global_thresholds']
        header = f"\n{'Location':<15} {'Baseline':>10}"
        for t in thresh_keys:
            header += f" {t:>8}°C"
        print(header)
        print("-" * (30 + 10 * len(thresh_keys)))

        for loc, data in sorted(thresholds.items()):
            row = f"{loc:<15} {data['preindustrial_baseline']:>10.2f}"
            for t in thresh_keys:
                val = data['thresholds'].get(str(t), 0)
                row += f" {val:>10.2f}"
            print(row)

    print("\n  R. Regenerate thresholds")
    print("  0. Back to main menu")

    choice = get_input("\nSelect").upper()

    if choice == "R":
        print("\nThis will recalculate thresholds from Historical raw data.")
        if confirm("Proceed?"):
            config["warming_thresholds"] = calculate_warming_thresholds(config)
        pause()

    return config


def menu_alignment_offsets(config: Dict) -> Dict:
    """Alignment offsets submenu."""
    clear_screen()
    print_header("ALIGNMENT OFFSETS")

    offsets = config.get("alignment_offsets", {})

    print(f"\nAlignment year: {config['parameters']['alignment_year']}")

    if not offsets:
        print("\nNo alignment offsets calculated.")
        print("Run 'Regenerate' to calculate from AGCD data.")
    else:
        print(f"\n{'Scenario':<15} {'Align Year':>12} {'Offsets':>10}")
        print("-" * 40)
        for scen, data in sorted(offsets.items()):
            count = len(data.get("offsets", {}))
            print(f"{scen:<15} {data['align_to_year']:>12} {count:>10}")

    print("\n  R. Regenerate offsets")
    print("  V. View offset details")
    print("  0. Back to main menu")

    choice = get_input("\nSelect").upper()

    if choice == "R":
        print("\nThis will recalculate offsets from AGCD data.")
        if confirm("Proceed?"):
            config["alignment_offsets"] = calculate_alignment_offsets(config)
        pause()
    elif choice == "V" and offsets:
        scen = get_input("Enter scenario name")
        if scen in offsets:
            print_subheader(f"Offsets for {scen}")
            for key, val in sorted(offsets[scen]["offsets"].items()):
                print(f"  {key}: {val:+.4f}")
            pause()

    return config


def menu_amplification(config: Dict) -> Dict:
    """Amplification factors submenu (view/calculate)."""
    clear_screen()
    print_header("AMPLIFICATION FACTORS")

    # Get current values - support both old single value and new per-location dict
    params = config["parameters"]
    amp_factors = params.get("amplification_factors", {})
    old_single = params.get("amplification_factor")  # Legacy single value

    print("\nCurrent values:")
    if amp_factors:
        for loc, val in sorted(amp_factors.items()):
            print(f"  {loc:<15}: {val}")
    elif old_single:
        print(f"  (legacy single): {old_single}")
    else:
        print("  None configured")

    # Show calculation metadata if available
    calc_date = config.get("metadata", {}).get("amplification_calculated")
    if calc_date:
        print(f"\nLast calculated: {calc_date[:19].replace('T', ' ')}")

    # Check for required files
    nc_files = find_netcdf_files()
    print(f"\nRequired files:")
    print(f"  Global tas:   {nc_files['global'].name if nc_files['global'] else 'NOT FOUND'}")
    if nc_files['regional']:
        reg_type = nc_files.get('regional_type', 'tas')
        print(f"  Regional {reg_type}: {nc_files['regional'].name}")
    else:
        print(f"  Regional tas: NOT FOUND")
    print(f"  Land mask:    {'Found' if KML_PATH.exists() else 'NOT FOUND'} ({KML_PATH.name})")

    files_available = nc_files["global"] and nc_files["regional"]

    print("\nOptions:")
    print("  V. View current values")
    if files_available:
        print("  R. Regenerate (calculate from NetCDF)")
    else:
        print("  R. Regenerate (UNAVAILABLE - missing files)")
    print("  M. Manual entry for location")
    print("  0. Back to main menu")

    choice = get_input("\nSelect [V/R/M/0]").upper()

    if choice == "0":
        return config

    if choice == "V":
        print_subheader("AMPLIFICATION FACTORS")
        if amp_factors:
            for loc, val in sorted(amp_factors.items()):
                print(f"\n  {loc}:")
                print(f"    Factor: {val}")
                print(f"    When global temp rises 1.0°C, {loc} warms {val:.2f}°C")
        elif old_single:
            print(f"\n  Legacy single value: {old_single}")
            print(f"  When global temp rises 1.0°C, region warms {old_single:.2f}°C")
        else:
            print("\n  No amplification factors configured")
        pause()

    elif choice == "R" and files_available:
        print_subheader("CALCULATING FROM NETCDF DATA")
        new_factors = calculate_amplification_factors(config)

        if new_factors:
            print("\n  Current vs Calculated:")
            for loc, new_val in sorted(new_factors.items()):
                old_val = amp_factors.get(loc, old_single if old_single else "N/A")
                print(f"    {loc:<15}: {old_val} -> {new_val}")

            if confirm("\nUpdate amplification factors?"):
                params["amplification_factors"] = new_factors
                # Remove legacy single value if present
                if "amplification_factor" in params:
                    del params["amplification_factor"]
                config["metadata"]["amplification_calculated"] = datetime.now().isoformat()
                print("\n  [+] Updated amplification factors")
                print("\n  [!] Remember to regenerate warming thresholds (option 5)")
        pause()

    elif choice == "R" and not files_available:
        print("\n  [!] Cannot calculate - missing NetCDF files")
        print(f"\n  Expected locations:")
        print(f"    Global:   {NETCDF_GLOBAL_FOLDER}/tas_Amon_ACCESS-CM2_ssp*.nc")
        print(f"    Regional: {METRICS_FOLDER}/SSP1-26/netcdf/*tas*.nc")
        pause()

    elif choice == "M":
        # Manual entry for a specific location
        print("\nAvailable locations: Australia, Ravenswood")
        loc_name = get_input("Enter location name")
        if loc_name:
            current_val = amp_factors.get(loc_name, "not set")
            new_str = get_input(f"Enter amplification factor for {loc_name} (current: {current_val})")
            if new_str:
                try:
                    new_value = float(new_str)
                    print(f"\n  Current: {current_val}")
                    print(f"  New:     {new_value}")
                    if confirm("\nProceed with change?"):
                        if "amplification_factors" not in params:
                            params["amplification_factors"] = {}
                        params["amplification_factors"][loc_name] = new_value
                        print(f"\n  [+] Updated {loc_name} amplification factor")
                        print("\n  [!] Remember to regenerate warming thresholds (option 5)")
                except ValueError:
                    print("\n  [!] Invalid number")
        pause()

    return config


def menu_regenerate_all(config: Dict) -> Tuple[str, Dict]:
    """Regenerate all calculated values."""
    clear_screen()
    print_header("REGENERATE ALL")

    # Check if amplification can be calculated
    nc_files = find_netcdf_files()
    can_calc_amp = nc_files["global"] and nc_files["regional"]

    print("\nThis will recalculate:")
    print("  - Scenario info (from metricsDataFiles)")
    if can_calc_amp:
        print("  - Amplification factors (from NetCDF data)")
    else:
        print("  - Amplification factors (SKIPPED - missing NetCDF files)")
    print("  - Warming thresholds (from Historical raw data)")
    print("  - Alignment offsets (from AGCD data)")

    if not confirm("\nProceed with regeneration?"):
        return config

    print()
    config["scenario_info"] = update_scenario_info(config)

    # Calculate amplification factors if NetCDF files available
    if can_calc_amp:
        amp_factors = calculate_amplification_factors(config)
        if amp_factors:
            config["parameters"]["amplification_factors"] = amp_factors
            # Remove legacy single value if present
            if "amplification_factor" in config["parameters"]:
                del config["parameters"]["amplification_factor"]
            config["metadata"]["amplification_calculated"] = datetime.now().isoformat()

    config["warming_thresholds"] = calculate_warming_thresholds(config)
    config["alignment_offsets"] = calculate_alignment_offsets(config)

    print("\n" + "=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)

    print(f"\nScenarios: {len(config['scenario_info'])}")
    amp_factors = config["parameters"].get("amplification_factors", {})
    if amp_factors:
        print(f"Amplification factors: {len(amp_factors)} locations")
        for loc, val in sorted(amp_factors.items()):
            print(f"  - {loc}: {val}")
    print(f"Locations with thresholds: {len(config['warming_thresholds'])}")
    print(f"Scenarios with offsets: {len(config['alignment_offsets'])}")

    if confirm("\nSave configuration now?"):
        save_config(config)

    pause()
    return config


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("Loading configuration...")
    config = load_config()

    while True:
        result, config = menu_main(config)
        if result == "quit":
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()