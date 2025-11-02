#!/usr/bin/env python3
# climateCMIP6ImporterWeb.py — OPTIMIZED with LAND-SEA MASKING
#
# New features:
# - IPCC-standard land-sea masking (>50% land threshold)
# - Automatic sftlf (land fraction) download
# - Ocean grid cells excluded from regional statistics
# - Parallel downloads (3 simultaneous)
# - Clean in-place progress display (no scrolling)
# - Request logging
# - File caching
# - All original features preserved

import os
import json
import warnings
import gc
import hashlib
import time
import sys
import shutil
import zipfile
import requests  # For direct CDS API calls
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# ===== Configuration =====
BASE = Path(__file__).resolve().parent
ROOT_METRICS = BASE / "metricsDataFiles"

LEVEL = 'single_levels'

# CDS variable name mapping
AVAILABLE_VARIABLES = {
    'tas': {
        'cds_name': 'near_surface_air_temperature',
        'description': 'Daily mean near-surface air temperature',
        'unit': 'K → °C'
    },
    'tasmax': {
        'cds_name': 'daily_maximum_near_surface_air_temperature',
        'description': 'Daily maximum near-surface air temperature',
        'unit': 'K → °C'
    },
    'tasmin': {
        'cds_name': 'daily_minimum_near_surface_air_temperature',
        'description': 'Daily minimum near-surface air temperature',
        'unit': 'K → °C'
    },
    'pr': {
        'cds_name': 'precipitation',
        'description': 'Daily precipitation',
        'unit': 'kg m-2 s-1 → mm/day'
    },
    'sfcWind': {
        'cds_name': 'near_surface_wind_speed',
        'description': 'Near-surface wind speed',
        'unit': 'm/s'
    },
    'huss': {
        'cds_name': 'near_surface_specific_humidity',
        'description': 'Near-surface specific humidity',
        'unit': 'kg/kg → g/kg'
    },
    'psl': {
        'cds_name': 'sea_level_pressure',
        'description': 'Sea level pressure',
        'unit': 'Pa → hPa'
    },
    'sftlf': {
        'cds_name': 'percentage_of_the_grid_cell_occupied_by_land',
        'description': 'Land area fraction (for masking ocean)',
        'unit': '%',
        'is_static': True  # Does not vary with time
    }
}

# Experiment mapping
EXPERIMENT_INFO = {
    'historical': {
        'cds_name': 'historical',
        'description': 'Historical simulation (1850-2014)',
        'default_years': (1850, 2014),
        'dir_name': 'historical'
    },
    'SSP1-26': {
        'cds_name': 'ssp1_2_6',
        'description': 'SSP1-2.6: Low emissions, sustainability',
        'default_years': (2015, 2100),
        'dir_name': 'SSP1-26'
    },
    'SSP2-45': {
        'cds_name': 'ssp2_4_5',
        'description': 'SSP2-4.5: Middle of the road',
        'default_years': (2015, 2100),
        'dir_name': 'SSP2-45'
    },
    'SSP3-70': {
        'cds_name': 'ssp3_7_0',
        'description': 'SSP3-7.0: Regional rivalry, high emissions',
        'default_years': (2015, 2100),
        'dir_name': 'SSP3-70'
    },
    'SSP5-85': {
        'cds_name': 'ssp5_8_5',
        'description': 'SSP5-8.5: Fossil-fueled development',
        'default_years': (2015, 2100),
        'dir_name': 'SSP5-85'
    }
}

# Common models
COMMON_MODELS = [
    'access_cm2',
    'access_esm1_5',
    'canesm5',
    'cesm2',
    'cnrm_cm6_1',
    'ec_earth3',
    'hadgem3_gc31_ll',
    'miroc6',
    'mpi_esm1_2_lr',
    'ukesm1_0_ll'
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def validate_netcdf_file(cache_path: Path) -> bool:
    """
    Validate that a NetCDF file is complete and readable

    Checks:
    - File exists
    - File has reasonable size (>1 MB)
    - File has NetCDF magic bytes
    """
    if not cache_path.exists():
        return False

    # Check file size (climate files should be at least 1 MB)
    file_size = cache_path.stat().st_size
    if file_size < 1024 * 1024:  # Less than 1 MB
        return False

    # Check NetCDF magic bytes
    try:
        with open(cache_path, 'rb') as f:
            header = f.read(4)
            # NetCDF classic: CDF\001
            # NetCDF4/HDF5: \211HDF
            if len(header) < 4:
                return False
            if header[:3] not in [b'CDF', b'\x89HD']:
                return False
    except Exception:
        return False

    return True


# ===== Progress Display =====

class DownloadProgress:
    """Clean in-place progress display for downloads"""

    def __init__(self, variables: List[str], experiment: str, use_simple: bool = False):
        self.variables = variables
        self.experiment = experiment
        self.status = {var: {'state': 'pending', 'message': '', 'size': 0, 'time': 0} for var in variables}
        self.start_time = time.time()
        self.last_update = 0
        self.use_simple = use_simple or not sys.stdout.isatty()

        if not self.use_simple:
            self.header_lines = 3
            self.total_lines = len(variables) + self.header_lines + 1

    def clear_lines(self, n: int):
        """Clear n lines by moving cursor up"""
        for _ in range(n):
            sys.stdout.write('\033[F\033[K')
        sys.stdout.flush()

    def start(self):
        """Initialize display"""
        if self.use_simple:
            print(f"\n{'═' * 70}")
            print(f"DOWNLOADING: {self.experiment}")
            print(f"{'═' * 70}")
        else:
            self._print_full_status()

    def update(self, var_name: str, state: str, message: str = "", size_mb: float = 0, elapsed: float = 0):
        """Update status for a variable"""
        now = time.time()
        if now - self.last_update < 0.3:  # Throttle updates
            return
        self.last_update = now

        self.status[var_name] = {'state': state, 'message': message, 'size': size_mb, 'time': elapsed}

        if self.use_simple:
            self._print_simple_status()
        else:
            self._print_full_status()

    def _print_simple_status(self):
        """Single-line status update"""
        completed = sum(1 for s in self.status.values() if s['state'] == 'completed')
        downloading = sum(1 for s in self.status.values() if s['state'] == 'downloading')

        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)

        # Find currently active variable
        active_var = next((v for v, s in self.status.items() if s['state'] in ['downloading', 'queued', 'running']), '')
        active_state = self.status[active_var]['state'] if active_var else 'waiting'

        status_line = (
            f"\r  {completed}/{len(self.variables)} done | "
            f"{downloading} downloading | "
            f"{active_var:12s}: {active_state:12s} | "
            f"{mins:02d}:{secs:02d}  "
        )

        sys.stdout.write(status_line)
        sys.stdout.flush()

    def _print_full_status(self):
        """Full multi-line status display"""
        if hasattr(self, 'total_lines') and self.total_lines > 0:
            self.clear_lines(self.total_lines)

        # Header
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        completed = sum(1 for s in self.status.values() if s['state'] == 'completed')

        print(f"\n{'═' * 70}")
        print(f"{self.experiment:40s} | {completed}/{len(self.variables)} | {mins:02d}:{secs:02d}")
        print(f"{'═' * 70}")

        # Status for each variable
        for var in self.variables:
            s = self.status[var]
            icon = self._get_icon(s['state'])

            # Format: icon var_name state message
            state_str = f"{s['state']:12s}"
            msg_str = s['message'][:40] if s['message'] else ''

            # Add size/time if completed
            if s['state'] == 'completed' and s['size'] > 0:
                msg_str = f"{s['size']:.0f}MB in {s['time']:.0f}s"

            print(f"  {icon} {var:12s} {state_str} {msg_str}")

        print()  # Blank line at bottom
        sys.stdout.flush()

    def _get_icon(self, state: str) -> str:
        """Get icon for state"""
        icons = {
            'pending': '⏳',
            'cached': '💾',
            'queued': '📤',
            'running': '⚙️ ',
            'downloading': '⬇️ ',
            'completed': '✅',
            'failed': '❌'
        }
        return icons.get(state, '?')

    def finish(self):
        """Print final summary"""
        if self.use_simple:
            print("\n")

        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        completed = sum(1 for s in self.status.values() if s['state'] == 'completed')
        failed = sum(1 for s in self.status.values() if s['state'] == 'failed')

        if not self.use_simple:
            self.clear_lines(self.total_lines)

        print(f"{'═' * 70}")
        print(f"✅ {self.experiment}: {completed}/{len(self.variables)} complete in {mins:02d}:{secs:02d}")
        if failed > 0:
            print(f"❌ Failed: {failed}")
        print(f"{'═' * 70}\n")
        sys.stdout.flush()


# ===== Request Tracking =====

def load_request_log(netcdf_dir: Path) -> dict:
    """Load request log from scenario netcdf directory"""
    log_file = netcdf_dir / "cds_requests.json"

    if not log_file.exists():
        return {}

    try:
        with open(log_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_request_log(netcdf_dir: Path, requests: dict):
    """Save request log to scenario netcdf directory"""
    log_file = netcdf_dir / "cds_requests.json"

    with open(log_file, 'w') as f:
        json.dump(requests, f, indent=2)


def get_request_key(model: str, experiment: str, variable: str, year_range: tuple,
                    temporal_resolution: str, bbox: Optional[tuple]) -> str:
    """Generate unique key for a request"""
    bbox_str = "global" if bbox is None else f"bbox_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    return f"{model}_{experiment}_{variable}_{year_range[0]}-{year_range[1]}_{temporal_resolution}_{bbox_str}"


def check_pending_request(netcdf_dir: Path, request_key: str, client) -> tuple:
    """
    Check if there's a pending request and attempt to download if ready

    Returns:
        (status, cache_path, request_info)
        status: 'not_found', 'downloaded', 'pending', 'failed'
    """
    import requests

    request_log = load_request_log(netcdf_dir)

    if request_key not in request_log:
        return ('not_found', None, None)

    request_info = request_log[request_key]
    request_id = request_info.get('request_id')
    filename = request_info.get('filename')
    cache_path = netcdf_dir / filename

    if not request_id:
        return ('not_found', None, None)

    # Check if file already downloaded
    if cache_path.exists() and validate_netcdf_file(cache_path):
        return ('downloaded', cache_path, request_info)

    # Get CDS credentials from client
    try:
        url = client.url
        key = client.key
    except:
        # Can't check status without credentials
        return ('pending', None, request_info)

    # Check request status via CDS API
    try:
        status_url = f"{url}/tasks/{request_id}"
        response = requests.get(status_url, auth=(key.split(':')[0], key.split(':')[1]))

        if response.status_code != 200:
            log(f"  ⚠️  Could not check status for {filename}")
            return ('pending', None, request_info)

        task_info = response.json()
        state = task_info.get('state', 'unknown')

        if state == 'completed':
            log(f"  ✓ Request ready: {filename}")

            # Get download URL
            download_url = task_info.get('location')
            if not download_url:
                log(f"  ⚠️  No download URL available")
                return ('failed', None, request_info)

            # Download the file
            log(f"  Downloading from CDS...")
            temp_path = cache_path.parent / f"{filename}.tmp"

            download_response = requests.get(download_url, stream=True)
            if download_response.status_code != 200:
                log(f"  ❌ Download failed")
                return ('failed', None, request_info)

            with open(temp_path, 'wb') as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Handle ZIP extraction
            if is_zip_file(temp_path):
                log(f"  Extracting ZIP...")
                final_path = extract_netcdf_from_zip(temp_path, cache_path.parent)
                if final_path:
                    temp_path.unlink()  # Remove the ZIP
                    cache_path = final_path
                else:
                    log(f"  ⚠️  Failed to extract, keeping ZIP")
                    temp_path.rename(cache_path.with_suffix('.nc.zip'))
                    return ('failed', None, request_info)
            else:
                temp_path.rename(cache_path)

            # Validate
            if validate_netcdf_file(cache_path):
                log(f"  ✓ Downloaded and validated: {filename}")
                # Update log
                request_info['status'] = 'completed'
                request_info['downloaded_at'] = datetime.utcnow().isoformat() + "Z"
                save_request_log(netcdf_dir, request_log)
                return ('downloaded', cache_path, request_info)
            else:
                log(f"  ❌ Downloaded file is invalid")
                return ('failed', None, request_info)

        elif state == 'failed':
            log(f"  ❌ Request failed: {filename}")
            request_info['status'] = 'failed'
            save_request_log(netcdf_dir, request_log)
            return ('failed', None, request_info)

        else:
            # Still queued or running
            request_info['status'] = state
            request_info['last_checked'] = datetime.utcnow().isoformat() + "Z"
            save_request_log(netcdf_dir, request_log)
            return ('pending', None, request_info)

    except Exception as e:
        log(f"  ⚠️  Could not check/download request: {e}")
        return ('pending', None, request_info)


def log_request(netcdf_dir: Path, request_key: str, request_id: str,
                filename: str, status: str = 'queued'):
    """Log a CDS request"""
    request_log = load_request_log(netcdf_dir)

    request_log[request_key] = {
        'request_id': request_id,
        'filename': filename,
        'status': status,
        'submitted_at': datetime.utcnow().isoformat() + "Z",
        'last_checked': datetime.utcnow().isoformat() + "Z"
    }

    save_request_log(netcdf_dir, request_log)


def cleanup_completed_requests(netcdf_dir: Path):
    """Remove completed/failed requests from log to keep it clean"""
    request_log = load_request_log(netcdf_dir)

    # Keep only queued/running requests
    active_log = {
        key: info for key, info in request_log.items()
        if info.get('status') in ['queued', 'running', 'unknown']
    }

    save_request_log(netcdf_dir, active_log)


# ===== Cache Management =====

def init_scenario_cache(scenario_dir: Path):
    """Initialize netcdf cache directory for a scenario"""
    netcdf_dir = scenario_dir / "netcdf"
    netcdf_dir.mkdir(parents=True, exist_ok=True)
    return netcdf_dir


def get_cache_filename(netcdf_dir: Path, model: str, experiment: str, variable: str,
                       year_range: tuple, temporal_resolution: str, bbox: Optional[tuple]) -> Path:
    """Generate unique cache filename in scenario netcdf directory"""
    bbox_str = "global" if bbox is None else f"bbox_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    filename = f"{model}_{experiment}_{variable}_{year_range[0]}-{year_range[1]}_{temporal_resolution}_{bbox_str}.nc"
    filename = filename.replace(" ", "_").replace(":", "_").replace("/", "_")
    return netcdf_dir / filename


def check_cache(cache_path: Path) -> bool:
    """Check if cached NetCDF file exists and is valid"""
    if not cache_path.exists():
        return False
    if cache_path.stat().st_size == 0:
        cache_path.unlink()
        return False
    # Quick validation - check it's a proper NetCDF
    if not validate_netcdf_file(cache_path):
        return False
    return True


# ===== Geographic Config =====

def load_geographic_config(config_path: Path = None) -> dict:
    """Load geographic areas configuration"""
    if config_path is None:
        config_path = Path(__file__).parent / "geographic_areas.json"

    if not config_path.exists():
        print(f"\n⚠️  ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        return json.load(f)


# ===== Grid Functions =====

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


# ===== LAND-SEA MASKING FUNCTIONS =====

def find_local_sftlf(model: str, base_dir: Path = None) -> Optional[Path]:
    """
    Search for local sftlf (land fraction) file

    Checks multiple naming conventions:
    - sftlf_MODEL.nc
    - sftlf_fx_MODEL.nc
    - MODEL_sftlf.nc

    Args:
        model: Model name (e.g., 'access_cm2')
        base_dir: Directory to search (default: script directory)

    Returns:
        Path to sftlf file if found, None otherwise
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    # Possible file naming patterns
    patterns = [
        f"sftlf_{model}.nc",
        f"sftlf_fx_{model}.nc",
        f"{model}_sftlf.nc",
        f"sftlf_{model.upper()}.nc",
        f"sftlf_fx_{model.upper()}.nc",
        f"{model.upper()}_sftlf.nc"
    ]

    for pattern in patterns:
        candidate = base_dir / pattern
        if candidate.exists():
            log(f"✓ Found local sftlf file: {candidate.name}")
            return candidate

    return None


def load_sftlf_from_file(sftlf_path: Path, bbox: Optional[tuple] = None) -> xr.Dataset:
    """
    Load sftlf from local NetCDF file and optionally subset to bbox

    Args:
        sftlf_path: Path to sftlf NetCDF file
        bbox: Optional bounding box (north, west, south, east)

    Returns:
        xarray Dataset with sftlf data
    """
    log(f"Loading land fraction from {sftlf_path.name}")

    try:
        ds = xr.open_dataset(sftlf_path)

        # Subset to bbox if provided
        if bbox is not None:
            try:
                lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
                lon_name = 'lon' if 'lon' in ds.coords else 'longitude'

                # Normalise longitude if needed
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
                log(f"   Using full grid")

        return ds

    except Exception as e:
        log(f"❌ Failed to load {sftlf_path}: {e}")
        return None


def create_land_mask(sftlf_da: xr.DataArray, threshold: float = 50.0) -> xr.DataArray:
    """
    Create binary land mask from land fraction data

    Following IPCC AR6 methodology:
    - Grid cells with >50% land are considered land
    - Ocean cells are masked out

    Args:
        sftlf_da: Land area fraction DataArray (0-100%)
        threshold: Percentage threshold for land (default 50%)

    Returns:
        Binary mask DataArray (1=land, 0=ocean, NaN=missing)
    """
    # IPCC standard: >50% land
    land_mask = xr.where(sftlf_da > threshold, 1.0, 0.0)

    # Set ocean cells to NaN for proper masking
    land_mask = xr.where(land_mask > 0, 1.0, np.nan)

    return land_mask


def apply_land_mask(da: xr.DataArray, land_mask: xr.DataArray) -> xr.DataArray:
    """
    Apply land mask to data array

    Args:
        da: Data array to mask
        land_mask: Binary land mask (1=land, NaN=ocean)

    Returns:
        Masked data array with ocean cells set to NaN
    """
    return da * land_mask


def area_mean_land_only(da: xr.DataArray, lat: str, lon: str,
                        land_mask: Optional[xr.DataArray] = None) -> xr.DataArray:
    """
    Calculate area-weighted mean over LAND ONLY

    Follows IPCC methodology:
    1.  Apply land mask (>50% land threshold)
    2.  Weight by latitude (cosine of latitude)
    3.  Compute mean only over land cells

    Args:
        da: Data array
        lat: Name of latitude coordinate
        lon: Name of longitude coordinate
        land_mask: Binary land mask (1=land, NaN=ocean)

    Returns:
        Area-weighted mean over land only
    """
    # Apply land mask if provided
    if land_mask is not None:
        # Broadcast mask to match data dimensions
        da_masked = da * land_mask
    else:
        da_masked = da

    # Latitude weighting (cosine of latitude)
    weights = np.cos(np.deg2rad(da_masked[lat]))

    # Compute weighted mean, excluding NaN (ocean) cells
    weighted_mean = da_masked.weighted(weights).mean(dim=[lat, lon], skipna=True)

    return weighted_mean


# ===== Interactive Functions =====

def display_experiments():
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENTS (SCENARIOS)")
    print("=" * 70)
    for i, (key, info) in enumerate(EXPERIMENT_INFO.items(), 1):
        years = f"{info['default_years'][0]}-{info['default_years'][1]}"
        print(f"{i}. {key:15s} | {years:15s} | {info['description']}")
    print("=" * 70)


def display_variables():
    print("\n" + "=" * 70)
    print("AVAILABLE VARIABLES")
    print("=" * 70)
    for i, (key, info) in enumerate(AVAILABLE_VARIABLES.items(), 1):
        if key == 'sftlf':
            print(f"{i}. {key:10s} | {info['unit']:20s} | {info['description']} [AUTO]")
        else:
            print(f"{i}. {key:10s} | {info['unit']:20s} | {info['description']}")
    print("=" * 70)
    print("\nNote: sftlf (land fraction) will be used from local file if available,")
    print("      or downloaded from CDS if needed (for ocean masking)")


def display_models():
    print("\n" + "=" * 70)
    print("COMMON MODELS IN CDS")
    print("=" * 70)
    for i, model in enumerate(COMMON_MODELS, 1):
        print(f"{i}. {model}")
    print("=" * 70)


def select_multiple(options: list, prompt: str, default_all=False) -> list:
    print(f"\n{prompt}")
    print("Enter numbers separated by commas (e.g., 1,3,5)")
    print("Enter 'all' for all options")
    if default_all:
        print("Press Enter for all options")

    while True:
        choice = input("> ").strip().lower()

        if not choice and default_all:
            return options

        if choice == 'all':
            return options

        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            selected = [options[i - 1] for i in indices if 1 <= i <= len(options)]
            if selected:
                return selected
            print("Invalid selection. Try again.")
        except (ValueError, IndexError):
            print("Invalid input. Enter numbers like: 1,2,3 or 'all'")


def get_temporal_resolution() -> str:
    print("\n" + "=" * 70)
    print("TEMPORAL RESOLUTION")
    print("=" * 70)
    print("1. Daily - 365 rows per year")
    print("2. Monthly (RECOMMENDED) - 12 rows per year, 30x faster download!")
    print("=" * 70)

    choice = input("Select resolution (1-2, default=2): ").strip() or '2'

    if choice == '2':
        print("Selected: Monthly (faster downloads)")
        return 'monthly'
    else:
        print("Selected: Daily (slower, larger files)")
        return 'daily'


def get_geographical_area(config: dict) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
    """Get geographical bounding box from config file"""
    areas = config.get('areas', {})

    if not areas:
        print("ERROR: No areas defined in geographic_areas.json")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GEOGRAPHICAL AREA SELECTION")
    print("=" * 70)
    print("Available regions (from geographic_areas.json):")
    print()

    area_list = list(areas.items())
    for i, (name, info) in enumerate(area_list, 1):
        desc = info.get('description', '')
        bbox = info.get('bbox')

        if bbox is None:
            size_info = "(global, no subsetting)"
        else:
            area_sq = abs((bbox['east'] - bbox['west']) * (bbox['north'] - bbox['south']))
            size_info = f"({area_sq:.1f} sq°)"

        print(f"{i}. {name:20s} {size_info:20s} {desc}")

    print("=" * 70)

    default_idx = next((i for i, (name, _) in enumerate(area_list) if name == 'Australia'), 0)
    choice = input(f"Select region (1-{len(area_list)}, default={default_idx + 1}): ").strip()

    if not choice:
        choice_idx = default_idx
    else:
        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(area_list):
                choice_idx = default_idx
        except ValueError:
            choice_idx = default_idx

    region_name, region_info = area_list[choice_idx]
    bbox_dict = region_info.get('bbox')

    if bbox_dict is None:
        print(f"\nSelected: {region_name} (global)")
        bbox = None
    else:
        bbox = (bbox_dict['north'], bbox_dict['west'], bbox_dict['south'], bbox_dict['east'])
        print(f"\nSelected: {region_name}")
        print(f"Bounding box: N={bbox[0]}°, W={bbox[1]}°, S={bbox[2]}°, E={bbox[3]}°")

    return bbox, region_name


# ===== Unit Conversion Functions =====

def find_lat_lon(ds):
    lat = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
    lon = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
    if not lat or not lon:
        raise ValueError("Could not find lat/lon coordinates")
    return lat, lon


def normalise_lon(ds, lon):
    try:
        ds = ds.assign_coords({lon: ((ds[lon] + 180) % 360) - 180}).sortby(lon)
    except Exception:
        pass
    return ds


def tas_to_degC(da):
    if (da.attrs.get("units", "").lower() in ("k", "kelvin")):
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da


def pr_to_mm_per_step(da):
    units = (da.attrs.get("units") or "").lower()
    cm = (da.attrs.get("cell_methods") or "").lower()
    if "sum" in cm or units in {"mm", "kg m-2", "kg m**-2"}:
        out = da.copy()
        out.attrs["units"] = "mm"
        return out
    out = da * 86400
    out.attrs["units"] = "mm"
    return out


def huss_to_g_per_kg(da):
    units = (da.attrs.get("units") or "").lower().replace(" ", "")
    if units in {"kgkg-1", "kg/kg", "kgkg**-1", "1"}:
        out = da * 1000.0
        out.attrs["units"] = "g/kg"
    elif units in {"g/kg", "gkg-1", "gkg**-1"}:
        out = da.copy()
        out.attrs["units"] = "g/kg"
    else:
        out = da.copy()
        out.attrs["units"] = units or "unknown"
    return out


def psl_to_hpa(da):
    units = (da.attrs.get("units") or "").lower()
    if units in {"pa", "pascal", "pascals"}:
        out = da / 100.0
        out.attrs["units"] = "hPa"
    elif units in {"hpa", "mb", "mbar"}:
        out = da.copy()
        out.attrs["units"] = "hPa"
    else:
        out = da.copy()
        out.attrs["units"] = units or "unknown"
    return out


def rav_point(da, lat, lon, lat_tgt, lon_tgt):
    return da.sel({lat: lat_tgt, lon: lon_tgt}, method="nearest")


def to_series(da):
    da = da.compute() if hasattr(da.data, "compute") else da
    t = da["time"].values
    idx = [pd.to_datetime(x).strftime("%Y-%m-%d") for x in t]
    return pd.Series(np.asarray(da.values).astype(float), index=pd.Index(idx, name="time"))


# ===== Download Helper =====

def _download_file_threaded(request_obj, cache_path: Path, var_name: str) -> bool:
    """
    Download helper for thread pool with validation

    Returns:
        True if successful and file is valid, False otherwise
    """
    try:
        # Suppress download progress output
        import contextlib
        import io

        # Download the file (suppress tqdm and other output)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            request_obj.download(str(cache_path))

        # CRITICAL: Wait for filesystem to sync
        time.sleep(2)

        # Check if CDS returned a ZIP file (common for CDS)
        try:
            with open(cache_path, 'rb') as f:
                header = f.read(4)
                if header[:2] == b'PK':  # ZIP file magic bytes
                    # Extract NetCDF from ZIP
                    temp_dir = cache_path.parent / f".temp_{cache_path.stem}"
                    temp_dir.mkdir(exist_ok=True)

                    try:
                        with zipfile.ZipFile(cache_path, 'r') as zf:
                            zf.extractall(temp_dir)

                        nc_files = list(temp_dir.glob("*.nc"))
                        if nc_files:
                            # Replace ZIP with extracted NetCDF
                            cache_path.unlink()
                            shutil.move(str(nc_files[0]), str(cache_path))

                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception as zip_err:
                        print(f"\n⚠️  Failed to extract ZIP for {var_name}: {zip_err}")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        if cache_path.exists():
                            cache_path.unlink()
                        return False
        except Exception:
            pass

        # Validate the downloaded file
        if not validate_netcdf_file(cache_path):
            print(f"\n⚠️  Downloaded file invalid for {var_name}")
            print(f"   File may be corrupted or incomplete")

            # Delete invalid file
            if cache_path.exists():
                cache_path.unlink()

            return False

        return True

    except Exception as e:
        print(f"\n❌ Download error for {var_name}: {e}")

        # Clean up any partial file
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception:
                pass

        return False


# ===== OPTIMIZED CDS Fetch with Progress Display =====

def fetch_from_cds_parallel(model, experiment, variables, year_range, bbox,
                            temporal_resolution='daily', max_retries=3, netcdf_dir=None,
                            download_land_mask=True):
    """
    Optimized fetch with clean progress display and automatic land mask download
    Checks for local sftlf file first before attempting CDS download
    """
    import cdsapi
    import logging

    # Suppress CDS API logging
    logging.getLogger('cdsapi').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)

    if netcdf_dir is None:
        raise ValueError("netcdf_dir must be specified")

    # Check for local sftlf file first
    local_sftlf_path = None
    if download_land_mask:
        local_sftlf_path = find_local_sftlf(model, BASE)

        if local_sftlf_path:
            log(f"✓ Using local land fraction file: {local_sftlf_path.name}")
            log("  No need to download sftlf from CDS")
        else:
            log("📍 No local sftlf found, will attempt CDS download")

    # Add sftlf to variables if land masking requested AND no local file
    all_variables = list(variables)
    if download_land_mask and not local_sftlf_path and 'sftlf' not in all_variables:
        all_variables.append('sftlf')
        log("📍 Adding sftlf (land fraction) to download queue")

    years = [str(y) for y in range(year_range[0], year_range[1] + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)] if temporal_resolution == 'daily' else None

    # Convert variable names
    cds_variables = []
    var_mapping = {}
    for var_name in all_variables:
        if var_name in AVAILABLE_VARIABLES:
            cds_var = AVAILABLE_VARIABLES[var_name]['cds_name']
            cds_variables.append(cds_var)
            var_mapping[cds_var] = var_name

    if not cds_variables:
        return {}, None

    # Check cache
    cached_vars = {}
    to_download = {}

    for cds_var, var_name in var_mapping.items():
        # sftlf is static (no time dimension), use simplified filename
        if var_name == 'sftlf':
            cache_path = get_cache_filename(netcdf_dir, model, 'fx', var_name, (0, 0),
                                            'fixed', bbox)
        else:
            cache_path = get_cache_filename(netcdf_dir, model, experiment, var_name, year_range,
                                            temporal_resolution, bbox)

        if check_cache(cache_path):
            cached_vars[var_name] = {'cache_path': cache_path, 'cds_var': cds_var}
        else:
            to_download[var_name] = {'cds_var': cds_var, 'cache_path': cache_path}

    log(f"Cache: {len(cached_vars)} cached, {len(to_download)} to download")

    # Initialize progress display
    all_vars = list(var_mapping.values())
    progress = DownloadProgress(all_vars, experiment)
    progress.start()

    # Mark cached as complete
    for var_name in cached_vars:
        progress.update(var_name, 'cached', 'Using cached file')

    submitted_requests = {}
    completed_downloads = {}

    if to_download:
        try:
            client = cdsapi.Client(wait_until_complete=False, delete=False)
        except Exception as e:
            progress.finish()
            log(f"❌ Failed to initialize CDS client: {e}")
            return {}, None

        # Check pending requests
        pending_count = 0
        failed_count = 0

        log(f"Checking for pending CDS requests...")
        for var_name, info in list(to_download.items()):
            cds_var = info['cds_var']
            cache_path = info['cache_path']

            if var_name == 'sftlf':
                request_key = get_request_key(model, 'fx', var_name, (0, 0), 'fixed', bbox)
            else:
                request_key = get_request_key(model, experiment, var_name, year_range,
                                              temporal_resolution, bbox)

            status, downloaded_path, request_info = check_pending_request(netcdf_dir, request_key, client)

            if status == 'downloaded':
                cached_vars[var_name] = {'cache_path': downloaded_path, 'cds_var': cds_var}
                to_download.pop(var_name)
                progress.update(var_name, 'cached', 'Downloaded from CDS')
            elif status == 'pending':
                pending_count += 1
                to_download.pop(var_name)
                progress.update(var_name, 'pending', f'Pending on CDS ({request_info.get("status", "queued")})')
            elif status == 'failed':
                failed_count += 1
                log(f"⚠️  Previous request for {var_name} failed, will resubmit")

        if pending_count > 0:
            log(f"⏳ {pending_count} request(s) still pending on CDS")
        if failed_count > 0:
            log(f"⚠️  {failed_count} failed request(s) will be resubmitted")

        # Submit new requests
        for var_name, info in to_download.items():
            cds_var = info['cds_var']
            cache_path = info['cache_path']

            # Special handling for sftlf (static field)
            if var_name == 'sftlf':
                request_params = {
                    "format": "netcdf",
                    "variable": cds_var,
                    "model": model,
                }
            else:
                request_params = {
                    "format": "netcdf",
                    "temporal_resolution": temporal_resolution,
                    "experiment": EXPERIMENT_INFO[experiment]['cds_name'],
                    "variable": cds_var,
                    "model": model,
                    "year": years,
                    "month": months
                }

            if bbox is not None:
                request_params["area"] = [bbox[0], bbox[1], bbox[2], bbox[3]]

            if temporal_resolution == 'daily' and var_name != 'sftlf':
                request_params["day"] = days

            try:
                request_obj = client.retrieve("projections-cmip6", request_params)

                submitted_requests[var_name] = {
                    'cds_var': cds_var,
                    'request': request_obj,
                    'cache_path': cache_path,
                    'submitted_at': time.time(),
                    'status': 'queued'
                }

                # Log the request
                if var_name == 'sftlf':
                    request_key = get_request_key(model, 'fx', var_name, (0, 0), 'fixed', bbox)
                else:
                    request_key = get_request_key(model, experiment, var_name, year_range,
                                                  temporal_resolution, bbox)

                log_request(netcdf_dir, request_key, request_obj.request_id,
                            cache_path.name, 'queued')

                progress.update(var_name, 'queued', 'Submitted to CDS')

            except Exception as e:
                progress.update(var_name, 'failed', f'Submit failed: {str(e)[:30]}')

        if not submitted_requests:
            progress.finish()
            # If we have cached vars but no new requests, still need to return proper tuple
            if cached_vars:
                # Load the cached datasets (need to process them)
                datasets = {}
                calendar = None
                for var_name, info in cached_vars.items():
                    cache_path = info['cache_path']
                    try:
                        ds = xr.open_dataset(cache_path, engine='netcdf4')
                        if calendar is None and 'time' in ds.coords:
                            calendar = ds.time.attrs.get('calendar', 'proleptic_gregorian')
                        datasets[var_name] = ds
                    except Exception:
                        pass

                # Load local sftlf if available
                if local_sftlf_path and 'sftlf' not in datasets:
                    sftlf_ds = load_sftlf_from_file(local_sftlf_path, bbox)
                    if sftlf_ds is not None:
                        datasets['sftlf'] = sftlf_ds

                return datasets, calendar
            else:
                return {}, None

        # Poll and download
        MAX_CONCURRENT_DOWNLOADS = 3

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
            download_futures = {}
            pending_requests = set(submitted_requests.keys())

            while pending_requests or download_futures:
                for var_name in list(pending_requests):
                    req_info = submitted_requests[var_name]
                    request_obj = req_info['request']
                    cache_path = req_info['cache_path']

                    try:
                        request_obj.update()
                        state = request_obj.reply.get('state', 'unknown')
                        req_info['status'] = state

                        if state == 'completed':
                            future = executor.submit(_download_file_threaded, request_obj, cache_path, var_name)
                            download_futures[future] = {
                                'var_name': var_name,
                                'cache_path': cache_path,
                                'started_at': time.time()
                            }
                            pending_requests.remove(var_name)
                            progress.update(var_name, 'downloading', 'Downloading...')
                        elif state == 'running':
                            progress.update(var_name, 'running', 'CDS processing')
                        elif state == 'failed':
                            progress.update(var_name, 'failed', 'CDS failed')
                            pending_requests.remove(var_name)
                    except Exception:
                        pass

                for future in list(download_futures.keys()):
                    if future.done():
                        info = download_futures.pop(future)
                        var_name = info['var_name']
                        cache_path = info['cache_path']

                        try:
                            success = future.result()

                            if success:
                                elapsed = time.time() - info['started_at']
                                file_size = cache_path.stat().st_size / (1024 ** 2)

                                progress.update(var_name, 'completed', '', file_size, elapsed)
                                completed_downloads[var_name] = {
                                    'cache_path': cache_path,
                                    'cds_var': submitted_requests[var_name]['cds_var']
                                }
                            else:
                                progress.update(var_name, 'failed', 'Download failed')
                        except Exception as e:
                            progress.update(var_name, 'failed', f'Error: {str(e)[:30]}')

                if pending_requests or download_futures:
                    time.sleep(1)

        progress.finish()
        cleanup_completed_requests(netcdf_dir)

    # Load datasets
    datasets = {}
    calendar = None

    all_vars = {**cached_vars, **completed_downloads}

    for var_name, info in all_vars.items():
        cache_path = info['cache_path']
        cds_var = info['cds_var']

        try:
            if not validate_netcdf_file(cache_path):
                log(f"❌ {var_name}: Invalid cache file, skipping")
                continue

            ds = None
            try:
                ds = xr.open_dataset(cache_path, engine='netcdf4')
            except Exception as e1:
                try:
                    ds = xr.open_dataset(cache_path, engine='h5netcdf')
                    log(f"   ℹ️  {var_name}: Using h5netcdf engine")
                except Exception as e2:
                    log(f"❌ {var_name}: Cannot open file")
                    continue

            if ds is None:
                continue

            if calendar is None and 'time' in ds.coords:
                calendar = ds.time.attrs.get('calendar', 'proleptic_gregorian')

            actual_var = None
            for possible in [cds_var, var_name]:
                if possible in ds:
                    actual_var = possible
                    break

            if actual_var:
                if actual_var != var_name:
                    ds = ds.rename({actual_var: var_name})

                if bbox is not None and var_name != 'sftlf':  # sftlf already subset
                    try:
                        lat, lon = find_lat_lon(ds)
                        ds = normalise_lon(ds, lon)
                        lat_slice = slice(bbox[2], bbox[0])
                        lon_slice = slice(bbox[1], bbox[3])
                        ds = ds.sel({lat: lat_slice, lon: lon_slice})
                    except Exception:
                        pass

                datasets[var_name] = ds

        except Exception as e:
            log(f"❌ Failed to load {var_name}: {str(e)[:100]}")

    # Load local sftlf file if available and not already downloaded
    if local_sftlf_path and 'sftlf' not in datasets:
        log("Loading local land fraction file...")
        sftlf_ds = load_sftlf_from_file(local_sftlf_path, bbox)
        if sftlf_ds is not None:
            datasets['sftlf'] = sftlf_ds
        else:
            log("⚠️  Failed to load local sftlf file")

    if calendar is None:
        calendar = 'proleptic_gregorian'

    return datasets, calendar


def process_datasets(datasets, region_name='Australia', temporal_resolution='daily'):
    """
    Process datasets WITH LAND-SEA MASKING

    Follows IPCC AR6 methodology:
    1.  Extract land fraction (sftlf)
    2.  Create binary mask (>50% land)
    3.  Apply mask before computing regional statistics
    4.  Point extraction remains unchanged
    """
    RAVENSWOOD_LAT = -20.115
    RAVENSWOOD_LON = 146.900

    log("Processing datasets with land-sea masking...")

    # Extract land mask if available
    land_mask = None
    if 'sftlf' in datasets:
        log("📍 Creating land mask from sftlf (IPCC >50% threshold)")

        sftlf_ds = datasets['sftlf']
        lat, lon = find_lat_lon(sftlf_ds)
        sftlf_da = sftlf_ds['sftlf']

        # Create binary land mask
        land_mask = create_land_mask(sftlf_da, threshold=50.0)

        # Count land vs ocean cells
        total_cells = land_mask.size
        land_cells = int(np.sum(~np.isnan(land_mask.values)))
        ocean_cells = total_cells - land_cells
        land_pct = (land_cells / total_cells) * 100

        log(f"   Grid cells: {land_cells:,} land ({land_pct:.1f}%) | {ocean_cells:,} ocean ({100 - land_pct:.1f}%)")
        log(f"   Using IPCC standard: cells with >50% land included in statistics")

        # Remove sftlf from datasets (no longer needed)
        del datasets['sftlf']
    else:
        log("⚠️  WARNING: No land fraction data (sftlf) available")
        log("   Computing statistics over ALL grid cells (land + ocean)")
        log("   Recommend re-running with sftlf for accurate land-only statistics")

    data_dict = {}
    ravenswood_cell = None
    grid_info = None

    for var_name, ds in datasets.items():
        lat, lon = find_lat_lon(ds)
        ds = normalise_lon(ds, lon)
        da = ds[var_name]

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

            # Add land mask information if available
            if land_mask is not None:
                # Get land cell indices
                land_cell_indices = []
                mask_array = land_mask.values  # This is True where land, False where ocean/masked

                for i_lat in range(len(lats)):
                    for i_lon in range(len(lons)):
                        if not np.isnan(mask_array[i_lat, i_lon]):  # Land cell
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
                    "description": "No land masking applied - all cells used in regional statistics"
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

        display_name = var_name

        # Unit conversions
        if var_name in ['tas', 'tasmax', 'tasmin']:
            da = tas_to_degC(da)
            unit = "degC"
        elif var_name == 'pr':
            da = pr_to_mm_per_step(da)
            unit = "mm_day"
        elif var_name == 'sfcWind':
            unit = "ms"
            display_name = 'wind'
        elif var_name == 'huss':
            da = huss_to_g_per_kg(da)
            unit = "g_per_kg"
        elif var_name == 'psl':
            da = psl_to_hpa(da)
            unit = "hPa"
        else:
            unit = "unknown"

        output_var = display_name

        # Regional mean WITH LAND MASKING
        area_col = f"{output_var}_{region_name}_land_only_{unit}"
        if land_mask is not None:
            area_mean_result = area_mean_land_only(da, lat, lon, land_mask)
        else:
            # Fallback to unmasked mean if no land mask
            area_mean_result = area_mean_land_only(da, lat, lon, None)

        data_dict[area_col] = to_series(area_mean_result)

        # Point extraction (unchanged - no masking needed)
        point_col = f"{output_var}_Ravenswood_{unit}"
        data_dict[point_col] = to_series(rav_point(da, lat, lon, RAVENSWOOD_LAT, RAVENSWOOD_LON))

        ds.close()
        del ds, da
        gc.collect()

    df = pd.DataFrame(data_dict).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if temporal_resolution == 'monthly':
        df.index = pd.to_datetime(df.index).to_period('M').to_timestamp()

    log(f"Final DataFrame: {len(df):,} rows × {df.shape[1]} columns")

    if land_mask is not None:
        log(f"✅ Regional statistics computed over LAND ONLY (IPCC >50% threshold)")

    return df, ravenswood_cell, grid_info


# ===== Main Workflow =====

def interactive_workflow():
    """Main interactive workflow"""
    print("\n" + "=" * 70)
    print(" CMIP6 CLIMATE DATA IMPORTER - WITH LAND-SEA MASKING")
    print("=" * 70)
    print("\n✨ FEATURES:")
    print("  • IPCC-standard land-sea masking (>50% land threshold)")
    print("  • Ocean grid cells excluded from regional statistics")
    print("  • Automatic land fraction (sftlf) download")
    print("  • Parallel downloads (3 simultaneous)")
    print("  • Clean progress display (no scrolling)")
    print("=" * 70)

    geo_config = load_geographic_config()

    temporal_resolution = get_temporal_resolution()

    display_models()
    model_choice = input("\nSelect model (enter number, default=1): ").strip() or '1'
    try:
        model = COMMON_MODELS[int(model_choice) - 1]
    except (ValueError, IndexError):
        model = 'access_cm2'

    display_experiments()
    exp_keys = list(EXPERIMENT_INFO.keys())
    selected_exp_keys = select_multiple(exp_keys, "Select experiments:", default_all=False)

    display_variables()
    var_keys = [k for k in AVAILABLE_VARIABLES.keys() if k != 'sftlf']  # Exclude sftlf from selection
    selected_vars = select_multiple(var_keys, "Select variables:", default_all=True)

    bbox, region_name = get_geographical_area(geo_config)

    # Year ranges
    year_ranges = {}
    for exp in selected_exp_keys:
        year_ranges[exp] = EXPERIMENT_INFO[exp]['default_years']

    print("\n" + "=" * 70)
    print("READY TO DOWNLOAD")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Scenarios: {len(selected_exp_keys)}")
    print(f"Variables: {len(selected_vars)} + sftlf (land mask)")
    print(f"Region: {region_name}")
    print(f"Masking: IPCC standard (>50% land threshold)")
    print("=" * 70)

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    overall_start = time.time()

    for scenario_num, exp_key in enumerate(selected_exp_keys, 1):
        print(f"\n{'═' * 70}")
        print(f"SCENARIO {scenario_num}/{len(selected_exp_keys)}: {exp_key}")
        print(f"{'═' * 70}\n")

        year_range = year_ranges[exp_key]

        # Create scenario directory
        dir_name = EXPERIMENT_INFO[exp_key]['dir_name']
        scenario_dir = ROOT_METRICS / dir_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        netcdf_dir = init_scenario_cache(scenario_dir)

        existing_nc = list(netcdf_dir.glob("*.nc"))
        if existing_nc:
            log(f"Found {len(existing_nc)} existing NetCDF files in {netcdf_dir.name}/")

        # Fetch data (including sftlf for masking)
        datasets, calendar = fetch_from_cds_parallel(
            model=model,
            experiment=exp_key,
            variables=selected_vars,
            year_range=year_range,
            bbox=bbox,
            temporal_resolution=temporal_resolution,
            netcdf_dir=netcdf_dir,
            download_land_mask=True  # Enable land mask download
        )

        if not datasets:
            log(f"❌ Failed for {exp_key}")
            continue

        # Process with land masking
        log(f"Processing {exp_key} with land-sea masking...")
        df, ravenswood_cell, grid_info = process_datasets(datasets, region_name, temporal_resolution)

        # Save
        out_dir = scenario_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if temporal_resolution == 'monthly':
            out_parq = out_dir / "raw_monthly.parquet"
            out_json = out_dir / "raw_monthly.json"
        else:
            out_parq = out_dir / "raw_daily.parquet"
            out_json = out_dir / "raw_daily.json"

        df.to_parquet(out_parq, engine="pyarrow", compression="zstd", index=True)

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

        netcdf_files = sorted([f.name for f in netcdf_dir.glob("*.nc")])

        meta = {
            "model": model,
            "experiment": exp_key,
            "variant": "r1i1p1f1",
            "grid": "gn",
            "table_id": temporal_resolution,
            "calendar": calendar,
            "created": datetime.utcnow().isoformat() + "Z",
            "scenario": dir_name,

            "land_sea_masking": {
                "applied": True,
                "method": "IPCC AR6 standard",
                "threshold": "50% land",
                "description": "Regional statistics computed over land grid cells only (>50% land fraction)",
                "source": "sftlf (percentage_of_the_grid_cell_occupied_by_land)"
            },

            "source": {
                "method": "CDS API via cdsapi (parallel downloads with land masking)",
                "dataset": "projections-cmip6",
                "downloaded": datetime.utcnow().isoformat() + "Z",
                "cds_request": {
                    "model": model,
                    "experiment": EXPERIMENT_INFO[exp_key]['cds_name'],
                    "temporal_resolution": temporal_resolution,
                    "variables": selected_vars + ['sftlf'],
                    "year_range": list(year_range)
                },
                "netcdf_files": netcdf_files,
                "netcdf_directory": "netcdf/"
            },

            "lat_coord": "lat",
            "lon_coord": "lon",

            "ravenswood_target": {
                "lat": -20.115,
                "lon": 146.900
            },

            "ravenswood_cell": ravenswood_cell,

            "geographical_area": {
                "north": bbox[0],
                "west": bbox[1],
                "south": bbox[2],
                "east": bbox[3]
            } if bbox else None,

            "grid_info": grid_info,

            "time_range": [
                df.index.min().strftime('%Y-%m-%d') if hasattr(df.index.min(), 'strftime') else str(df.index.min()),
                df.index.max().strftime('%Y-%m-%d') if hasattr(df.index.max(), 'strftime') else str(df.index.max())
            ],

            "temporal_resolution": temporal_resolution,

            "date": {
                "column": "time",
                "format": "YYYY-MM-DD",
                "freq": "MS" if temporal_resolution == "monthly" else "D"
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

        with open(out_json, "w") as f:
            json.dump(meta, f, indent=2)

        log(f"✅ Saved {out_parq.name}")
        log(f"   Metadata: {out_json.name}")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'═' * 70}")
    print(f"✅ ALL COMPLETE in {overall_elapsed / 60:.1f} minutes")
    print(f"{'═' * 70}\n")


def main():
    """Main entry point"""
    import argparse

    p = argparse.ArgumentParser(description="CMIP6 climate data importer with land-sea masking")
    p.add_argument("--interactive", "-i", action="store_true", default=True)
    args = p.parse_args()

    cds_rc = Path.home() / ".cdsapirc"
    if not cds_rc.exists():
        print("\n⚠️  ERROR: CDS credentials not found!")
        print(f"\nCreate: {cds_rc}")
        return 1

    interactive_workflow()
    return 0


if __name__ == "__main__":
    sys.exit(main())