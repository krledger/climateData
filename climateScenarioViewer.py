#!/usr/bin/env python3
"""
Climate Metrics Viewer (Streamlit)
A streamlined single-file app for browsing climate metrics with charts and tables.
Supports various metric types including Temperature, Rain, Wind and Humidity.
Now includes 1.5°C warming analysis tab with visual dashboard.
Modified to include time horizon controls and chart shading options.

IMPROVEMENTS:
- Fixed smoothing effect on delta calculations
- Added continuity for historical-to-future transitions
- Enhanced dashboard with emphasis on changes
- Added reference year column to dashboard
"""

import os
import sys
import re
from typing import Sequence, Tuple, Literal, Dict, Optional
from io import BytesIO

import pandas as pd
import numpy as np

try:
    import folium
    from folium import plugins as folium_plugins
    from streamlit_folium import st_folium

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# ============================== CONFIG ========================================
MODE = "metrics"
PORT = 8501

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
FOLDER = os.path.join(BASE_DIR, "metricsDataFiles")

GROUP_KEYS = ["Location", "Type", "Name", "Season", "Data Type"]
IDX_COLS_ANNUAL = ["Year"]
IDX_COLS_SEASONAL = ["Year", "Season"]

# 1.5°C Analysis Configuration
BASELINE_PERIOD = (1850, 1900)
WARMING_TARGET = 1.5  # degrees Celsius
AUSTRALIA_BOUNDS = {'lat': (-45, -10), 'lon': (110, 155)}

# Dashboard Configuration
REFERENCE_YEAR = 2020  # Reference year for dashboard comparisons

# TEMPORARY: Hardcoded baseline until historical data available
# Based on IPCC AR6: pre-industrial (1850-1900) global mean ~13.7°C
# Australian baseline approximately 21.5°C for continental average
USE_HARDCODED_BASELINE = True
HARDCODED_GLOBAL_BASELINE = 13.7  # degrees Celsius (global mean)
HARDCODED_AUSTRALIA_BASELINE = 21.5  # degrees Celsius (Australia mean)


# ============================ HELPERS =========================================
def resolve_folder() -> str:
    """Validate metrics folder exists."""
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
    """Convert string to URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def parse_data_type(series: pd.Series) -> pd.DataFrame:
    """Extract Type, Name, Location from 'Data Type' column."""
    return series.str.extract(
        r"^(?P<Type>[^ ]+) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$"
    )


# ============================ UI COMPONENTS ===================================
def multi_selector(
        container,
        label: str,
        options,
        default=None,
        widget: Literal["checkbox", "toggle"] = "checkbox",
        columns: int = 1,
        key_prefix: str = "sel",
        namespace: str = None
):
    """Render multi-selection UI with checkboxes or toggles."""
    opts = dedupe_preserve_order(options or [])
    default = set(default or [])
    selected = []

    container.markdown(f"**{label}**")
    cols = container.columns(columns)
    widget_fn = container.checkbox if widget == "checkbox" else container.toggle

    # Add hash of options to ensure unique keys even with same namespace
    import hashlib
    opts_hash = hashlib.md5(str(sorted(opts)).encode()).hexdigest()[:8]

    for i, opt in enumerate(opts):
        key = f"{namespace + '_' if namespace else ''}{key_prefix}_{i}_{opts_hash}_{slugify(opt)}"
        with cols[i % columns]:
            if widget_fn(str(opt), value=(opt in default), key=key):
                selected.append(opt)
    return selected


# ============================ DATA TRANSFORMS =================================
def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist with proper types."""
    required = {
        "Year": "Int64",
        "Season": str,
        "Data Type": str,
        "Value": float,
        "Location": str,
        "Name": str,
        "Type": str,
        "Scenario": str
    }

    df = df.copy()

    # Add missing columns
    if "Season" not in df.columns:
        df["Season"] = "Annual"

    for col, dtype in required.items():
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
    """Calculate deltas by subtracting base scenario values.

    For historical data, use the first available base scenario value as offset
    since historical periods don't overlap with future scenarios.
    """
    join_keys = ["Year", "Season", "Data Type", "Location", "Type", "Name"]
    base_values = base[join_keys + ["Value"]].rename(columns={"Value": "BaseValue"})

    merged = view.merge(base_values, on=join_keys, how="left")

    # For rows without a match (e.g. historical data), use first base value per group
    group_keys = ["Season", "Data Type", "Location", "Type", "Name"]

    # Get first year base values for each group
    first_base = base.sort_values("Year").groupby(group_keys, as_index=False).first()
    first_base = first_base[group_keys + ["Value"]].rename(columns={"Value": "FirstBaseValue"})

    # Merge first base values
    merged = merged.merge(first_base, on=group_keys, how="left")

    # Use matched base value if available, otherwise use first base value
    merged["BaseValue"] = merged["BaseValue"].fillna(merged["FirstBaseValue"])

    # Calculate delta
    merged["Value"] = merged["Value"] - merged["BaseValue"]

    return merged.drop(columns=["BaseValue", "FirstBaseValue"])


def apply_baseline_from_start(view: pd.DataFrame) -> pd.DataFrame:
    """Subtract first year's value as baseline per group so each starts at zero."""
    if view.empty:
        return view

    keys = ["Scenario"] + GROUP_KEYS

    # Sort by year to ensure consistent first value selection
    view = view.sort_values("Year").copy()

    # Get first year value for each group
    first_values = view.groupby(keys, as_index=False).first()
    first_values = first_values[keys + ["Value"]].rename(columns={"Value": "Baseline"})

    # Merge baseline values back
    result = view.merge(first_values, on=keys, how="left")

    # Subtract baseline to make first value zero
    result["Value"] = result["Value"] - result["Baseline"]

    return result.drop(columns=["Baseline"])


def apply_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling mean smoothing per group."""
    if window <= 1 or df.empty:
        return df

    # Ensure odd window
    window = window if window % 2 == 1 else window + 1
    min_periods = max(1, window // 2)

    group_cols = ["Scenario"] + GROUP_KEYS

    smoothed_groups = []
    for _, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("Year").copy()
        group["Value"] = group["Value"].rolling(
            window, center=True, min_periods=min_periods
        ).mean()
        smoothed_groups.append(group)

    result = pd.concat(smoothed_groups, ignore_index=True)
    return result.dropna(subset=["Value"])


# ============================ DATA LOADING ====================================
def discover_scenarios(base_folder: str) -> list[Tuple[str, str, str]]:
    """Find all scenario folders with metrics parquet files."""
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
    """Load and prepare a single metrics parquet file."""
    try:
        df = pd.read_parquet(path, engine="pyarrow")

        # Parse Data Type column
        parts = parse_data_type(df["Data Type"])
        df = pd.concat([df, parts], axis=1)

        # Apply schema
        df = ensure_schema(df)

        return df.dropna(subset=["Year"])
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")


# ============================ 1.5°C ANALYSIS ==================================
def calculate_baseline_temperature(df: pd.DataFrame, location: str = None) -> float:
    """Calculate baseline global or location-specific mean temperature for 1850-1900."""
    baseline_data = df[
        (df["Year"] >= BASELINE_PERIOD[0]) &
        (df["Year"] <= BASELINE_PERIOD[1]) &
        (df["Type"] == "Temp") &
        (df["Name"] == "Average") &
        (df["Season"] == "Annual")
        ]

    if location:
        baseline_data = baseline_data[baseline_data["Location"] == location]

    if baseline_data.empty:
        return np.nan

    return baseline_data["Value"].mean()


def find_15C_warming_year(
        df: pd.DataFrame,
        baseline_temp: float,
        location: str = None
) -> Tuple[Optional[int], Optional[float], pd.DataFrame]:
    """Find the year when warming reaches 1.5°C above baseline.

    Returns:
        - Year when 1.5°C reached (or None)
        - Warming level at that year (or max warming if not reached)
        - DataFrame with annual temperature trajectory
    """
    # Get annual average temperatures
    annual_temps = df[
        (df["Type"] == "Temp") &
        (df["Name"] == "Average") &
        (df["Season"] == "Annual")
        ]

    if location:
        annual_temps = annual_temps[annual_temps["Location"] == location]

    annual_temps = annual_temps.groupby("Year")["Value"].mean().reset_index()
    annual_temps["Warming"] = annual_temps["Value"] - baseline_temp
    annual_temps = annual_temps.sort_values("Year")

    # Find first year reaching 1.5°C
    warming_years = annual_temps[annual_temps["Warming"] >= WARMING_TARGET]

    if not warming_years.empty:
        year = int(warming_years.iloc[0]["Year"])
        warming = warming_years.iloc[0]["Warming"]
        return year, warming, annual_temps

    # If not reached, return max warming
    if not annual_temps.empty:
        max_idx = annual_temps["Warming"].idxmax()
        max_warming = annual_temps.loc[max_idx, "Warming"]
        return None, max_warming, annual_temps

    return None, None, pd.DataFrame()


def extract_australian_conditions(
        df: pd.DataFrame,
        year: int,
        start_year: int
) -> Dict[str, Dict[str, float]]:
    """Extract Australian climate conditions for a specific year with changes from start year."""
    conditions = {}

    # Get data for target year and start year
    year_data = df[
        (df["Year"] == year) &
        (df["Location"] == "Australia")
        ]

    start_data = df[
        (df["Year"] == start_year) &
        (df["Location"] == "Australia")
        ]

    # Temperature metrics
    temp_metrics = [
        ("Temp", "Average", "tas"),
        ("Temp", "5-Day Avg Max", "5day_avg_max"),
        ("Temp", "Days>37", "days_over_37")
    ]

    # Precipitation metrics
    precip_metrics = [
        ("Rain", "CDD", "cdd"),
        ("Rain", "Max 5-Day", "max_5day"),
        ("Rain", "Total", "total"),
        ("Rain", "R20", "r20")
    ]

    # Wind metrics
    wind_metrics = [
        ("Wind", "95th Percentile", "wind_95p"),
        ("Wind", "Max Day", "wind_max")
    ]

    all_metrics = temp_metrics + precip_metrics + wind_metrics

    for metric_type, name, key in all_metrics:
        # Get year data
        year_metric = year_data[
            (year_data["Type"] == metric_type) &
            (year_data["Name"] == name)
            ]

        # Get start year data
        start_metric = start_data[
            (start_data["Type"] == metric_type) &
            (start_data["Name"] == name)
            ]

        if not year_metric.empty:
            # Calculate for all seasons
            for season in ["Annual", "DJF", "MAM", "JJA", "SON"]:
                season_year = year_metric[year_metric["Season"] == season]
                season_start = start_metric[start_metric["Season"] == season]

                if not season_year.empty:
                    year_val = float(season_year["Value"].mean())
                    start_val = float(season_start["Value"].mean()) if not season_start.empty else np.nan
                    change = year_val - start_val if not np.isnan(start_val) else np.nan

                    conditions[f"{key}_{season.lower()}"] = {
                        "value": year_val,
                        "change": change,
                        "label": f"{name} ({season})"
                    }

    return conditions


def analyze_15C_warming_from_data(
        scenarios: list[Tuple[str, str, str]],
        load_func,
        selected_location: str = "Australia"
) -> Dict:
    """Analyze when scenarios reach 1.5°C warming."""
    results = {
        "metadata": {
            "baseline_period": BASELINE_PERIOD,
            "warming_target": WARMING_TARGET,
            "using_hardcoded_baseline": USE_HARDCODED_BASELINE,
            "location": selected_location
        },
        "scenarios": {}
    }

    # Determine baseline temperature
    if USE_HARDCODED_BASELINE:
        baseline_temp = HARDCODED_AUSTRALIA_BASELINE if selected_location == "Australia" else HARDCODED_GLOBAL_BASELINE
        results["metadata"]["baseline_source"] = "Hardcoded (IPCC AR6 estimate)"
    else:
        # Try to load from historical data
        baseline_temp = None
        for label, _, path in scenarios:
            df = load_func(path, os.path.getmtime(path))
            baseline_temp = calculate_baseline_temperature(df, selected_location)
            if not np.isnan(baseline_temp):
                results["metadata"]["baseline_source"] = f"Calculated from {label}"
                break

        if baseline_temp is None or np.isnan(baseline_temp):
            return {
                "error": "Could not calculate baseline temperature.  Set USE_HARDCODED_BASELINE = True or add historical data."
            }

    results["metadata"]["baseline_temperature"] = baseline_temp

    # Analyze each scenario
    for label, _, path in scenarios:
        df = load_func(path, os.path.getmtime(path))

        # Get start year from data
        start_year = int(df["Year"].min())

        warming_year, warming_level, trajectory = find_15C_warming_year(df, baseline_temp, selected_location)

        results["scenarios"][label] = {
            "reaches_15C": warming_year is not None,
            "year": warming_year,
            "warming_level": warming_level,
            "trajectory": trajectory,
            "start_year": start_year
        }

        if warming_year:
            conditions = extract_australian_conditions(df, warming_year, start_year)
            results["scenarios"][label]["conditions_summary"] = conditions
            results["scenarios"][label]["years_until_15C"] = warming_year - 2025

    return results


def create_timeline_dataframe(results: Dict) -> pd.DataFrame:
    """Create a timeline dataframe showing when each scenario reaches 1.5°C."""
    timeline_data = []

    for scenario, data in results.get("scenarios", {}).items():
        if data.get("reaches_15C"):
            timeline_data.append({
                "Scenario": scenario,
                "Year": data["year"],
                "Warming (°C)": round(data["warming_level"], 2),
                "Years Until": data.get("years_until_15C", "N/A")
            })

    if not timeline_data:
        return pd.DataFrame()

    return pd.DataFrame(timeline_data).sort_values("Year")


def create_dashboard_image(results: Dict, selected_location: str) -> BytesIO:
    """Create a visual dashboard image summarising climate change impacts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return None

    # Create figure with custom size
    fig = plt.figure(figsize=(16, 10), facecolor='#F5F5F5')

    # Define colours
    bg_color = '#FFFFFF'
    header_color = '#2C5F7C'
    warm_color = '#E74C3C'
    cool_color = '#3498DB'
    precip_color = '#27AE60'
    wind_color = '#8E44AD'

    # Main title
    fig.suptitle(f'Climate Change Summary: {selected_location}',
                 fontsize=24, fontweight='bold', color=header_color, y=0.98)

    # Get baseline info
    baseline_temp = results["metadata"]["baseline_temperature"]
    baseline_period = results["metadata"]["baseline_period"]

    # Create subplots
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3,
                          left=0.08, right=0.92, top=0.92, bottom=0.08)

    # Title box
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7, f'Baseline: {baseline_period[0]}-{baseline_period[1]} ({baseline_temp:.1f}°C)',
                  ha='center', fontsize=14, style='italic', color=header_color)

    # Collect data from scenarios
    timeline_data = []
    temp_changes = []
    precip_changes = []

    for scenario, data in results["scenarios"].items():
        if data.get("reaches_15C"):
            timeline_data.append({
                'scenario': scenario,
                'year': data['year'],
                'warming': data['warming_level']
            })

            if "conditions_summary" in data:
                cond = data["conditions_summary"]

                # Temperature changes (Annual only)
                if "tas_annual" in cond and not np.isnan(cond["tas_annual"]["change"]):
                    temp_changes.append({
                        'scenario': scenario,
                        'tas': cond["tas_annual"]["change"],
                        'days37': cond.get("days_over_37_annual", {}).get("change", 0)
                    })

                # Precipitation changes
                if "total_annual" in cond and not np.isnan(cond["total_annual"]["change"]):
                    precip_changes.append({
                        'scenario': scenario,
                        'total': cond["total_annual"]["change"],
                        'cdd': cond.get("cdd_annual", {}).get("change", 0)
                    })

    # Plot 1: When 1.5°C warming is reached
    ax1 = fig.add_subplot(gs[1, :])
    if timeline_data:
        scenarios = [d['scenario'] for d in timeline_data]
        years = [d['year'] for d in timeline_data]
        warmings = [d['warming'] for d in timeline_data]

        colors_bars = [warm_color if w >= 1.5 else cool_color for w in warmings]
        bars = ax1.barh(scenarios, years, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add year labels on bars
        for i, (bar, year) in enumerate(zip(bars, years)):
            ax1.text(year + 1, i, f'{year}', va='center', fontsize=11, fontweight='bold')

        ax1.set_xlabel('Year', fontsize=13, fontweight='bold')
        ax1.set_title('When Does Each Scenario Reach 1.5°C Warming?',
                      fontsize=15, fontweight='bold', pad=15, color=header_color)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
    else:
        ax1.text(0.5, 0.5, 'No scenarios reach 1.5°C warming',
                 ha='center', va='center', fontsize=14, style='italic')
        ax1.axis('off')

    # Plot 2: Temperature changes
    ax2 = fig.add_subplot(gs[2, 0])
    if temp_changes:
        scenarios = [d['scenario'] for d in temp_changes]
        tas = [d['tas'] for d in temp_changes]

        bars = ax2.bar(range(len(scenarios)), tas, color=warm_color, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Change (°C)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Temperature Change', fontsize=13, fontweight='bold',
                      color=warm_color, pad=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, tas)):
            ax2.text(i, val + 0.05, f'+{val:.1f}°C', ha='center', fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, style='italic')
        ax2.axis('off')

    # Plot 3: Hot days
    ax3 = fig.add_subplot(gs[2, 1])
    if temp_changes:
        scenarios = [d['scenario'] for d in temp_changes]
        days37 = [d['days37'] for d in temp_changes]

        bars = ax3.bar(range(len(scenarios)), days37, color='#FF6B6B', alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(scenarios)))
        ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax3.set_ylabel('Change (days)', fontsize=11, fontweight='bold')
        ax3.set_title('Days Over 37°C Change', fontsize=13, fontweight='bold',
                      color='#FF6B6B', pad=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, days37)):
            sign = '+' if val >= 0 else ''
            ax3.text(i, val + 0.5, f'{sign}{val:.0f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, style='italic')
        ax3.axis('off')

    # Plot 4: Precipitation changes
    ax4 = fig.add_subplot(gs[2, 2])
    if precip_changes:
        scenarios = [d['scenario'] for d in precip_changes]
        total = [d['total'] for d in precip_changes]

        colors_bars = [precip_color if t >= 0 else '#E67E22' for t in total]
        bars = ax4.bar(range(len(scenarios)), total, color=colors_bars, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax4.set_xticks(range(len(scenarios)))
        ax4.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax4.set_ylabel('Change (mm)', fontsize=11, fontweight='bold')
        ax4.set_title('Total Precipitation Change', fontsize=13, fontweight='bold',
                      color=precip_color, pad=10)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_axisbelow(True)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, total)):
            sign = '+' if val >= 0 else ''
            ax4.text(i, val + (5 if val >= 0 else -5), f'{sign}{val:.0f} mm',
                     ha='center', fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, style='italic')
        ax4.axis('off')

    # Plot 5: Consecutive dry days
    ax5 = fig.add_subplot(gs[3, 0])
    if precip_changes:
        scenarios = [d['scenario'] for d in precip_changes]
        cdd = [d['cdd'] for d in precip_changes]

        bars = ax5.bar(range(len(scenarios)), cdd, color='#E67E22', alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(scenarios)))
        ax5.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax5.set_ylabel('Change (days)', fontsize=11, fontweight='bold')
        ax5.set_title('Consecutive Dry Days Change', fontsize=13, fontweight='bold',
                      color='#E67E22', pad=10)
        ax5.grid(axis='y', alpha=0.3, linestyle='--')
        ax5.set_axisbelow(True)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, cdd)):
            sign = '+' if val >= 0 else ''
            ax5.text(i, val + 0.5, f'{sign}{val:.0f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, style='italic')
        ax5.axis('off')

    # Add disclaimer text
    ax_footer = fig.add_subplot(gs[3, 1:])
    ax_footer.axis('off')
    disclaimer = (
        'Changes shown are relative to the start year of the scenario data.\n'
        'All metrics shown are annual averages when 1.5°C warming is reached.'
    )
    ax_footer.text(0.5, 0.5, disclaimer, ha='center', va='center',
                   fontsize=10, style='italic', color='#7F8C8D',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                             edgecolor='#BDC3C7', linewidth=2))

    # Save to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return buf


# ============================ MAP FUNCTIONS ===================================
def create_region_map(kml_path: str, ravenswood_kml_path: str = None, center_location: tuple = (-26.5, 133.5),
                      zoom: int = 4):
    """Create an interactive Folium map with KML overlay showing grid coverage.

    Args:
        kml_path: Path to the main KML file
        ravenswood_kml_path: Path to Ravenswood-specific KML file (optional)
        center_location: (lat, lon) tuple for map center
        zoom: Initial zoom level

    Returns:
        Tuple of (Folium map object or None, status_message, placemark_count)
    """
    if not FOLIUM_AVAILABLE:
        return None, "Folium not available", 0

    if not os.path.exists(kml_path):
        return None, "KML file not found", 0

    # Create base map centered on Australia with OpenTopoMap (shows roads, cities, towns)
    m = folium.Map(
        location=center_location,
        zoom_start=zoom,
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a>',
        control_scale=True
    )

    placemark_count = 0
    status_msg = "Success"

    # Try to parse and display main KML
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(kml_path)
        root = tree.getroot()

        # Handle KML namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        # Find all Placemarks with coordinates
        placemarks = root.findall('.//kml:Placemark', ns)

        placemark_count = len(placemarks)

        for placemark in placemarks:
            name_elem = placemark.find('.//kml:name', ns)
            name = name_elem.text if name_elem is not None else 'Grid Cell'

            # Check if this is Ravenswood cell in main KML
            is_ravenswood = 'ravenswood' in name.lower()

            # Try to find coordinates in various formats
            coords_elem = placemark.find('.//kml:coordinates', ns)

            if coords_elem is not None and coords_elem.text:
                coords_text = coords_elem.text.strip()
                coords_list = []

                # Parse coordinates (format: lon,lat,alt lon,lat,alt ...)
                for coord in coords_text.split():
                    parts = coord.split(',')
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            coords_list.append([lat, lon])
                        except (ValueError, IndexError):
                            continue

                # Add polygon with different styling for Ravenswood
                if len(coords_list) > 2:
                    if is_ravenswood:
                        # Highlight Ravenswood with orange/red fill (very transparent)
                        folium.Polygon(
                            locations=coords_list,
                            popup=f"<b>{name}</b><br>(Selected Location)",
                            color='#E74C3C',
                            fill=True,
                            fillColor='#E74C3C',
                            fillOpacity=0.1,
                            weight=2
                        ).add_to(m)
                    else:
                        # Regular grid cells - grey lines only
                        folium.Polygon(
                            locations=coords_list,
                            popup=name,
                            color='lightgrey',
                            fill=False,
                            weight=1
                        ).add_to(m)
                elif len(coords_list) == 1:
                    # Single point
                    color = '#E74C3C' if is_ravenswood else 'grey'
                    folium.CircleMarker(
                        location=coords_list[0],
                        popup=name,
                        radius=5 if is_ravenswood else 3,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7 if is_ravenswood else 0.5
                    ).add_to(m)

    except Exception as e:
        status_msg = f"Partial load: {str(e)[:100]}"

    # Load and display Ravenswood-specific KML if provided
    ravenswood_loaded = False
    if ravenswood_kml_path and os.path.exists(ravenswood_kml_path):
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(ravenswood_kml_path)
            root = tree.getroot()

            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            placemarks = root.findall('.//kml:Placemark', ns)

            for placemark in placemarks:
                name_elem = placemark.find('.//kml:name', ns)
                name = name_elem.text if name_elem is not None else 'Ravenswood'

                coords_elem = placemark.find('.//kml:coordinates', ns)

                if coords_elem is not None and coords_elem.text:
                    coords_text = coords_elem.text.strip()
                    coords_list = []

                    for coord in coords_text.split():
                        parts = coord.split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                coords_list.append([lat, lon])
                            except (ValueError, IndexError):
                                continue

                    if len(coords_list) > 2:
                        # Highlight Ravenswood with orange/red fill (very transparent)
                        folium.Polygon(
                            locations=coords_list,
                            popup=f"<b>{name}</b><br>(Selected Location)",
                            color='#E74C3C',
                            fill=True,
                            fillColor='#E74C3C',
                            fillOpacity=0.1,
                            weight=2
                        ).add_to(m)
                        ravenswood_loaded = True
                        placemark_count += 1

        except Exception as e:
            if "Success" in status_msg:
                status_msg = f"Main KML loaded, Ravenswood KML issue: {str(e)[:50]}"

    return m, status_msg, placemark_count


# ============================ MAIN APP ========================================
def run_metrics_viewer() -> None:
    """Main Streamlit application."""
    try:
        import altair as alt
        import streamlit as st
    except ImportError:
        sys.exit("Missing dependencies.  Run: pip install streamlit pandas pyarrow altair")

    # ============================= DISCLAIMER ================================
    if "disclaimer_accepted" not in st.session_state:
        st.session_state.disclaimer_accepted = False

    st.set_page_config(page_title="Climate Metrics Viewer", layout="wide")

    if not st.session_state.disclaimer_accepted:
        # Centre the disclaimer content manually
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.title("🌡️ Climate Metrics Viewer")
            st.markdown("---")

            st.subheader("Data Disclaimer")
            st.markdown("""
By selecting "Accept," you acknowledge that the climate data provided through this viewer has not been independently verified and that you use any information displayed entirely at your own risk.  This tool is intended for general informational and educational purposes only and should not be used as the sole basis for any decision-making.  Climate projections and historical data may contain uncertainties, errors or limitations.  You are solely responsible for verifying any data before use and for determining whether this information is appropriate for your specific needs and requirements.  The creators and providers of this viewer make no representations or warranties of any kind, express or implied, regarding the accuracy, completeness, timeliness or reliability of the data presented.  Under no circumstances shall the creators or providers be liable for any decisions made or actions taken based on information from this viewer.
            """)

            st.markdown("---")

            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("Accept", key="accept_disclaimer", type="primary", width="stretch"):
                    st.session_state.disclaimer_accepted = True
                    st.rerun()

        st.stop()

    # ============================= DATA SETUP ================================
    @st.cache_data(show_spinner=False)
    def load_metrics_cached(path: str, mtime: float) -> pd.DataFrame:
        """Cached metrics loader."""
        _ = mtime  # Force cache invalidation on file changes
        return load_metrics_file(path)

    @st.cache_data(show_spinner=False)
    def calculate_preindustrial_baseline(
            all_data: pd.DataFrame,
            locations: list
    ) -> dict:
        """Calculate preindustrial baseline temperatures once at startup."""
        baseline_temps = {}

        historical_data = all_data[
            (all_data["Year"].between(1850, 1900)) &
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

    @st.cache_data(show_spinner=False)
    def load_minimal_metadata(
            pairs: Sequence[Tuple[str, str, float]]
    ) -> pd.DataFrame:
        """Load minimal columns for building filters."""
        frames = []
        for label, path, mtime in pairs:
            _ = mtime
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

    # Discover scenarios
    base_folder = resolve_folder()
    scenarios = discover_scenarios(base_folder)

    if not scenarios:
        st.error(f"❌ No scenarios found in: {base_folder}")
        st.stop()

    labels = [lbl for lbl, _, _ in scenarios]
    label_to_path = {lbl: path for lbl, _, path in scenarios}
    BASE_LABEL = "SSP1-26" if "SSP1-26" in labels else labels[0]

    # Load minimal metadata for filters
    metadata_tuples = [
        (lbl, label_to_path[lbl], os.path.getmtime(label_to_path[lbl]))
        for lbl in labels
    ]
    metadata = load_minimal_metadata(metadata_tuples)

    # ============================= TABS ======================================
    tab1, tab2 = st.tabs(["🌡️ Climate Metrics", "📊 Dashboard"])

    # ============================= TAB 1: METRICS VIEWER =====================
    with tab1:
        # ============================= SIDEBAR ===================================
        ns = "metrics"

        st.sidebar.title("⚙️ Filters")

        # Location filter
        with st.sidebar.expander("📍 Locations", expanded=True):
            all_locations = sorted(metadata["Location"].dropna().unique())
            default_locs = ["Ravenswood"] if "Ravenswood" in all_locations else all_locations[:1]

            loc_sel = multi_selector(
                st, "Select locations", all_locations,
                default=default_locs, columns=1, namespace=ns, key_prefix="loc"
            )

        if not loc_sel:
            st.sidebar.warning("⚠️ Select at least one location")
            st.stop()

        # Scenario filter
        with st.sidebar.expander("🌍 Scenarios", expanded=True):
            # Default to base scenario plus historical if available
            default_scen = [BASE_LABEL]
            if "historical" in [lbl.lower() for lbl in labels]:
                hist_label = next(lbl for lbl in labels if lbl.lower() == "historical")
                default_scen.append(hist_label)
            elif len(labels) > 1:
                default_scen.append(labels[1])

            scen_sel = multi_selector(
                st, "Select scenarios", labels,
                default=default_scen, columns=1, namespace=ns, key_prefix="scen"
            )

        if not scen_sel:
            st.sidebar.warning("⚠️ Select at least one scenario")
            st.stop()

        # Display options
        with st.sidebar.expander("🎨 Display Options", expanded=True):
            mode = st.radio(
                "Display Mode",
                ["Values", "Baseline (start year)", f"Deltas vs {BASE_LABEL}"],
                index=0, key=f"{ns}_mode"
            )

            smooth = st.toggle("Smooth Values", value=False, key=f"{ns}_smooth")
            if smooth:
                smooth_win = st.slider(
                    "Smoothing Window (years)",
                    3, 21, step=2, value=9, key=f"{ns}_smooth_win"
                )
            else:
                smooth_win = 1

            table_interval = st.radio(
                "Table Interval (years)",
                [1, 2, 5, 10],
                index=1, horizontal=True, key=f"{ns}_table_int"
            )

            # Chart shading options
            show_15c_shading = st.toggle(
                "1.5°C above preindustrial",
                value=False,
                key=f"{ns}_show_15c"
            )
            show_horizon_shading = st.toggle(
                "Time horizon shading",
                value=False,
                key=f"{ns}_show_horizons"
            )

        # Year range with time horizons
        with st.sidebar.expander("📆 Year Range", expanded=True):
            yr_min, yr_max = int(metadata["Year"].min()), int(metadata["Year"].max())

            # Find first non-historical year (typically 2015)
            non_historical_scenarios = [s for s in labels if "historical" not in s.lower()]
            if non_historical_scenarios:
                # Get minimum year from non-historical scenarios
                non_hist_data = metadata[metadata["Scenario"].isin(non_historical_scenarios)]
                default_start_year = int(non_hist_data["Year"].min()) if not non_hist_data.empty else yr_min
            else:
                default_start_year = yr_min

            # Initialise session state if not exists
            if f"{ns}_y0" not in st.session_state:
                st.session_state[f"{ns}_y0"] = default_start_year
            if f"{ns}_y1" not in st.session_state:
                st.session_state[f"{ns}_y1"] = yr_max

            # Year inputs
            col_start, col_end = st.columns(2)
            with col_start:
                y0 = st.number_input(
                    "Start year",
                    min_value=yr_min,
                    max_value=yr_max,
                    value=st.session_state[f"{ns}_y0"],
                    step=1,
                    key=f"{ns}_year_start"
                )
            with col_end:
                y1 = st.number_input(
                    "End year",
                    min_value=yr_min,
                    max_value=yr_max,
                    value=st.session_state[f"{ns}_y1"],
                    step=1,
                    key=f"{ns}_year_end"
                )

            # Update session state
            st.session_state[f"{ns}_y0"] = y0
            st.session_state[f"{ns}_y1"] = y1

            # Validate range
            if y0 > y1:
                st.error("⚠️ Start year must be before end year")
                y0, y1 = y1, y0

            # Time horizons with connected range sliders
            st.markdown("**Time horizons:**")

            # Short to Mid period (Blue in chart)
            short_mid = st.slider(
                "🟦 Short → Mid",
                min_value=y0,
                max_value=y1,
                value=(min(max(2020, y0), y1), min(max(2036, y0), y1)),
                step=1,
                key=f"{ns}_short_mid",
                help="Short-term period (blue shading on chart)"
            )
            short_start = short_mid[0]
            mid_start = short_mid[1]

            # Mid to Long period (Orange in chart)
            mid_long = st.slider(
                "🟧 Mid → Long",
                min_value=y0,
                max_value=y1,
                value=(mid_start, min(max(2039, y0), y1)),
                step=1,
                key=f"{ns}_mid_long",
                help="Medium-term period (orange shading on chart)"
            )
            mid_start = mid_long[0]
            long_start = mid_long[1]

            # Long to End period (Red in chart)
            long_end = st.slider(
                "🟥 Long → End",
                min_value=y0,
                max_value=y1,
                value=(long_start, min(2045, y1)),
                step=1,
                key=f"{ns}_long_end",
                help="Long-term period (red shading on chart)"
            )
            long_start = long_end[0]
            horizon_end = long_end[1]

        # Metric type filter
        with st.sidebar.expander("📊 Metric Type", expanded=False):
            preferred_types = ["Temp", "Rain", "Wind", "Humidity"]
            all_types = list(metadata["Type"].unique())
            type_options = [t for t in preferred_types if t in all_types] + \
                           [t for t in sorted(all_types) if t not in preferred_types]
            default_type = "Temp" if "Temp" in type_options else type_options[0]

            type_sel = st.radio(
                "Type", type_options,
                index=type_options.index(default_type),
                horizontal=True, key=f"{ns}_type"
            )

        # Metric name filter (depends on type and location)
        with st.sidebar.expander("📈 Metric Names", expanded=False):
            filtered_meta = metadata[
                (metadata["Location"].isin(loc_sel)) &
                (metadata["Type"] == type_sel)
                ]
            name_options = sorted(filtered_meta["Name"].dropna().unique())

            # Smart sorting based on type
            if type_sel == "Temp":
                priority = ["Average", "Max", "Max Day", "5-Day Avg Max", "Avg Max"]
                name_options = sorted(
                    name_options,
                    key=lambda n: (priority.index(n) if n in priority else 99, n)
                )
                default_names = ["Average"] if "Average" in name_options else name_options[:1]
            elif type_sel == "Humidity":
                priority = ["Average RH", "Average VPD"]
                name_options = sorted(
                    name_options,
                    key=lambda n: (priority.index(n) if n in priority else 99, n)
                )
                default_names = name_options[:1]
            else:
                default_names = name_options[:1]

            name_sel = multi_selector(
                st, "Select metrics", name_options,
                default=default_names, widget="toggle",
                columns=1, namespace=ns, key_prefix="name"
            )

        if not name_sel:
            st.sidebar.warning("⚠️ Select at least one metric")
            st.stop()

        # Season filter
        with st.sidebar.expander("📅 Seasons", expanded=False):
            seasons_all = ["Annual", "DJF", "MAM", "JJA", "SON"]
            available_seasons = [s for s in seasons_all if s in metadata["Season"].unique()]
            default_seasons = ["Annual"] if "Annual" in available_seasons else available_seasons

            season_sel = multi_selector(
                st, "Select seasons", available_seasons,
                default=default_seasons, columns=1,
                namespace=ns, key_prefix="season"
            )

        if not season_sel:
            st.sidebar.warning("⚠️ Select at least one season")
            st.stop()

        # Help section
        with st.sidebar.expander("ℹ️ Help"):
            st.markdown("""
            **Smoothing**: Applies a rolling average to reduce noise in the data.

            **Baseline**: Shows change from the first year in the selected range.

            **Deltas**: Shows difference from the reference scenario (SSP1-26).

            **Table Interval**: Controls row spacing in the values table.

            **Time Horizon Shading**: Shades chart regions matching sidebar colours (🟦 blue, 🟧 orange, 🟥 red) for annual data only.

            **1.5°C above preindustrial**: Shows horizontal red dashed line at 1.5°C above 1850-1900 baseline for temperature charts (hover over line for details).

            **Historical Gap**: Visual gaps between historical and future scenarios reflect actual temporal discontinuity in the data source.
            """)

        # ============================= MAIN CONTENT ==============================
        st.title("🌡️ Climate Metrics Viewer")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📍 Locations", len(loc_sel))
        with col2:
            st.metric("🌍 Scenarios", len(scen_sel))
        with col3:
            st.metric("📅 Year Range", f"{y0}–{y1}")
        with col4:
            st.metric("📊 Mode", mode.split("(")[0].strip())

        st.caption(
            f"Locations: {', '.join(loc_sel)} • "
            f"Scenarios: {', '.join(scen_sel)} • "
            f"Type: {type_sel} • "
            f"Metrics: {', '.join(name_sel)}"
        )

        # Load full data for selected scenarios
        with st.spinner("Loading data..."):
            dfs = []
            for label in scen_sel:
                path = label_to_path[label]
                df = load_metrics_cached(path, os.path.getmtime(path))
                df["Scenario"] = label
                dfs.append(df)

            df_all = pd.concat(dfs, ignore_index=True)

            # Load base scenario for delta calculation
            base_path = label_to_path[BASE_LABEL]
            base_df = load_metrics_cached(base_path, os.path.getmtime(base_path))
            base_df["Scenario"] = BASE_LABEL

            # Pre-calculate preindustrial baseline for all locations (once at startup)
            preindustrial_baseline = calculate_preindustrial_baseline(df_all, loc_sel)

        # Filter data
        mask = (
                df_all["Year"].between(y0, y1) &
                df_all["Location"].isin(loc_sel) &
                df_all["Season"].isin(season_sel) &
                (df_all["Type"] == type_sel) &
                df_all["Name"].isin(name_sel)
        )
        view = df_all[mask].copy()

        # Apply transformations
        use_baseline = mode.startswith("Baseline")
        apply_delta = mode.startswith("Deltas")

        # FIXED: Apply transformations before smoothing for deltas
        if apply_delta:
            # Filter base_df to same criteria
            base_mask = (
                    base_df["Year"].between(y0, y1) &
                    base_df["Location"].isin(loc_sel) &
                    base_df["Season"].isin(season_sel) &
                    (base_df["Type"] == type_sel) &
                    base_df["Name"].isin(name_sel)
            )
            base_filtered = base_df[base_mask].copy()

            # Apply deltas first
            view = apply_deltas_vs_base(view, base_filtered)

            # Then apply smoothing if requested
            if smooth:
                view = apply_smoothing(view, smooth_win)
        elif use_baseline:
            # For baseline mode, apply smoothing first to preserve first values
            if smooth:
                view = apply_smoothing(view, smooth_win)
            view = apply_baseline_from_start(view)
        else:
            # For raw values, just apply smoothing
            if smooth:
                view = apply_smoothing(view, smooth_win)

        view = view.dropna(subset=["Value"])

        # Check for empty results
        if view.empty:
            st.warning("⚠️ No data matches your current filters.  Try:")
            st.markdown("""
            - Expanding the year range
            - Selecting additional scenarios or locations
            - Choosing different metric names
            """)
            st.stop()

        # Determine index columns for pivot
        use_seasonal = len(season_sel) > 1 or (
                len(season_sel) == 1 and season_sel[0] != "Annual"
        )
        idx_cols = IDX_COLS_SEASONAL if use_seasonal else IDX_COLS_ANNUAL

        # Calculate 1.5°C reference line for temperature charts using pre-calculated baseline
        warming_reference = None
        if type_sel == "Temp" and "Average" in name_sel and show_15c_shading:
            if preindustrial_baseline and not use_baseline and not apply_delta:
                # Use pre-calculated baseline temperatures
                warming_reference = {loc: temp + 1.5 for loc, temp in preindustrial_baseline.items()}

        # ============================= SUMMARY STATISTICS ========================
        with st.expander("📊 Summary Statistics", expanded=False):
            summary = view.groupby(["Scenario", "Name"])["Value"].agg([
                ("Mean", "mean"),
                ("Min", "min"),
                ("Max", "max"),
                ("Change", lambda x: x.max() - x.min())
            ]).round(2)
            st.dataframe(summary, width="stretch")

        # ============================= CHARTS =====================================
        st.subheader("📈 Visualisation")

        # Create separate chart for each location
        for location in loc_sel:
            st.markdown(f"### 📍 {location}")

            location_data = view[view["Location"] == location].copy()

            if location_data.empty:
                st.info(f"No data available for {location}")
                continue

            plot_data = location_data[idx_cols + ["Data Type", "Value", "Location", "Name", "Scenario"]].copy()
            plot_data = plot_data.rename(columns={"Data Type": "Metric"})

            # Prepare X-axis
            if idx_cols == IDX_COLS_ANNUAL:
                plot_data["X"] = plot_data["Year"].astype(int)
                x_encoding = alt.X(
                    "X:Q",
                    title="Year",
                    axis=alt.Axis(format="d", labelFontSize=12, titleFontSize=14),
                    scale=alt.Scale(domain=[int(y0), int(y1)])
                )
            else:
                season_order = ["DJF", "MAM", "JJA", "SON"]
                plot_data["Season"] = pd.Categorical(
                    plot_data["Season"], categories=season_order, ordered=True
                )
                plot_data = plot_data.sort_values(["Year", "Season"])
                plot_data["X"] = (
                        plot_data["Year"].astype(int).astype(str) +
                        "-" +
                        plot_data["Season"].astype(str)
                )
                x_encoding = alt.X(
                    "X:N",
                    title="Year–Season",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=14, labelAngle=-45),
                    sort=list(plot_data["X"].unique())
                )

            # Interactive selection
            selection = alt.selection_point(fields=["Metric", "Scenario"], bind="legend")

            # Build chart layers
            chart_layers = []

            # Add time horizon shading if enabled (only for annual data)
            if show_horizon_shading and idx_cols == IDX_COLS_ANNUAL:
                # Short period (short_start to mid_start)
                if short_start < mid_start and short_start >= y0 and mid_start <= y1:
                    short_rect = alt.Chart(pd.DataFrame({
                        'start': [max(short_start, y0)],
                        'end': [min(mid_start, y1)]
                    })).mark_rect(opacity=0.15, color='#3498DB').encode(
                        x=alt.X('start:Q', scale=alt.Scale(domain=[int(y0), int(y1)])),
                        x2=alt.X2('end:Q'),
                        tooltip=alt.value(f'Short period: {short_start}–{mid_start}')
                    )
                    chart_layers.append(short_rect)

                # Mid period (mid_start to long_start)
                if mid_start < long_start and mid_start >= y0 and long_start <= y1:
                    mid_rect = alt.Chart(pd.DataFrame({
                        'start': [max(mid_start, y0)],
                        'end': [min(long_start, y1)]
                    })).mark_rect(opacity=0.15, color='#F39C12').encode(
                        x=alt.X('start:Q', scale=alt.Scale(domain=[int(y0), int(y1)])),
                        x2=alt.X2('end:Q'),
                        tooltip=alt.value(f'Mid period: {mid_start}–{long_start}')
                    )
                    chart_layers.append(mid_rect)

                # Long period (long_start to horizon_end)
                if long_start < horizon_end and long_start >= y0 and horizon_end <= y1:
                    long_rect = alt.Chart(pd.DataFrame({
                        'start': [max(long_start, y0)],
                        'end': [min(horizon_end, y1)]
                    })).mark_rect(opacity=0.15, color='#E74C3C').encode(
                        x=alt.X('start:Q', scale=alt.Scale(domain=[int(y0), int(y1)])),
                        x2=alt.X2('end:Q'),
                        tooltip=alt.value(f'Long period: {long_start}–{horizon_end}')
                    )
                    chart_layers.append(long_rect)

            # Main line chart
            chart = (
                alt.Chart(plot_data)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=x_encoding,
                    y=alt.Y(
                        "Value:Q",
                        title="Value",
                        scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=14)
                    ),
                    color=alt.Color(
                        "Scenario:N",
                        title="Scenario",
                        legend=alt.Legend(
                            labelFontSize=12,
                            titleFontSize=14,
                            labelLimit=400
                        )
                    ),
                    strokeDash=alt.StrokeDash(
                        "Metric:N",
                        title="Metric",
                        legend=alt.Legend(
                            labelFontSize=12,
                            titleFontSize=14,
                            labelLimit=300
                        )
                    ),
                    tooltip=[
                        alt.Tooltip("Scenario:N", title="Scenario"),
                        alt.Tooltip("Metric:N", title="Metric"),
                        alt.Tooltip("Value:Q", format=",.2f", title="Value"),
                        alt.Tooltip("X:N", title="Period"),
                        alt.Tooltip("Location:N", title="Location"),
                        alt.Tooltip("Name:N", title="Name"),
                    ],
                    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
                )
                .add_params(selection)
            )

            chart_layers.append(chart)

            # Add 1.5°C warming reference line if available for this location
            if warming_reference and location in warming_reference:
                reference_temp = warming_reference[location]
                reference_line = (
                    alt.Chart(pd.DataFrame({"temp": [reference_temp]}))
                    .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2)
                    .encode(
                        y=alt.Y("temp:Q"),
                        tooltip=alt.value(f"1.5°C warming target ({reference_temp:.2f}°C)")
                    )
                )
                chart_layers.append(reference_line)

            # Combine all layers
            combined_chart = alt.layer(*chart_layers).properties(height=450).interactive()
            st.altair_chart(combined_chart, use_container_width=True)

        # ============================= TABLE =====================================
        st.subheader("📋 Data Table")

        # Apply table interval filtering
        table_view = view.copy()
        if table_interval > 1:
            anchor = int(table_view["Year"].min())
            table_view = table_view[
                ((table_view["Year"].astype(int) - anchor) % table_interval) == 0
                ]

        # Create pivot table
        if len(scen_sel) > 1:
            table = table_view.pivot_table(
                index=idx_cols,
                columns=["Data Type", "Scenario"],
                values="Value",
                aggfunc="first"
            ).sort_index()
        else:
            table = table_view.pivot_table(
                index=idx_cols,
                columns="Data Type",
                values="Value",
                aggfunc="first"
            ).sort_index()

        st.dataframe(table, width="stretch", height=400)

        # Download button
        csv = table.to_csv()
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"climate_metrics_{type_sel}_{y0}-{y1}.csv",
            mime="text/csv",
            key=f"{ns}_download"
        )

    # ============================= TAB 2: DASHBOARD ==========================
    with tab2:
        st.title("📊 Climate Change Dashboard")

        # Get non-historical scenarios
        dashboard_scenarios = [s for s in scen_sel if "historical" not in s.lower()]

        if not dashboard_scenarios:
            st.warning("⚠️ Please select at least one future scenario (excluding historical) to view dashboard.")
            st.stop()

        # Location selector for dashboard
        dashboard_location = st.selectbox(
            "Select location for dashboard",
            loc_sel,
            key="dashboard_location"
        )

        # Display region map if KML available
        kml_path = os.path.join(BASE_DIR, "Australia_grid_coverage.kml")
        ravenswood_kml_path = os.path.join(BASE_DIR, "Ravenswood_grid_cell.kml")

        if FOLIUM_AVAILABLE and os.path.exists(kml_path):
            with st.expander("🗺️ Region Map", expanded=False):
                st.markdown("**Grid coverage area for climate metrics**")
                region_map, status_msg, placemark_count = create_region_map(
                    kml_path,
                    ravenswood_kml_path if os.path.exists(ravenswood_kml_path) else None
                )
                if region_map:
                    if placemark_count > 0:
                        st.caption(f"✓ Displaying {placemark_count} grid cells from KML file")
                    elif "Partial" in status_msg:
                        st.warning(f"⚠️ {status_msg}")
                    st_folium(region_map, width=1100, height=600)
                else:
                    st.info(f"Map unavailable: {status_msg}")
        elif not FOLIUM_AVAILABLE:
            st.info("📦 Install folium and streamlit-folium to view region map: `pip install folium streamlit-folium`")

        # Dashboard metrics configuration
        dashboard_metrics = {
            "🌡️ Temperature": [
                ("Temp", "Average", "°C", "🌡️"),
                ("Temp", "5-Day Avg Max", "°C", "☀️"),
                ("Temp", "Days>=37", "days", "🔥")
            ],
            "🌧️ Precipitation": [
                ("Rain", "Max 5-Day", "mm", "🌧️"),
                ("Rain", "R20", "days", "💧"),
                ("Rain", "Total", "mm", "☔")
            ],
            "🏜️ Drought": [
                ("Rain", "CDD", "days", "🏜️")
            ],
            "💨 Wind": [
                ("Wind", "95th Percentile", "m/s", "💨"),
                ("Wind", "Average", "m/s", "🌬️")
            ]
        }

        # Helper function to find metric name with variations
        def find_metric_name(data, metric_type, preferred_name):
            """Find actual metric name in data, trying variations if exact match not found"""
            available = data[data["Type"] == metric_type]["Name"].unique()

            # Try exact match first
            if preferred_name in available:
                return preferred_name

            # Try common variations for days metrics
            if "Days" in preferred_name or "days" in preferred_name:
                variations = [
                    preferred_name,
                    preferred_name.replace(">=", ">"),
                    preferred_name.replace(">", ">="),
                    preferred_name.replace(">", " >"),
                    preferred_name.replace(">=", " >="),
                    preferred_name.replace(" ", ""),
                    # Try with spaces
                    preferred_name.replace("Days", "Days "),
                    preferred_name.replace("Tx", "Tx "),
                ]
                for var in variations:
                    if var in available:
                        return var

                # Try case-insensitive partial match
                preferred_lower = preferred_name.lower().replace(" ", "")
                for avail in available:
                    avail_lower = avail.lower().replace(" ", "")
                    if "37" in preferred_lower and "37" in avail_lower and (
                            "day" in avail_lower or "tx" in avail_lower):
                        return avail

            # Return original if no match found
            return preferred_name

        # Calculate changes for each scenario and metric
        with st.spinner("Calculating dashboard metrics..."):
            # Filter data for dashboard location
            dashboard_data = df_all[
                (df_all["Location"] == dashboard_location) &
                (df_all["Season"] == "Annual") &
                (df_all["Scenario"].isin(dashboard_scenarios))
                ].copy()

            if dashboard_data.empty:
                st.error("❌ No data available for selected location and scenarios.")
                st.stop()

            # Calculate baseline (start year) values for each metric/scenario
            # Use the start year from sidebar settings
            baseline_year = y0

            # Calculate pre-industrial baselines for ALL metrics from HISTORICAL scenario (1850-1900)
            preindustrial_baselines = {}

            # Find historical scenario
            historical_scenario = None
            for label in labels:
                if "historical" in label.lower():
                    historical_scenario = label
                    break

            if historical_scenario:
                historical_path = label_to_path[historical_scenario]
                historical_df = load_metrics_cached(historical_path, os.path.getmtime(historical_path))

                historical_data = historical_df[
                    (historical_df["Location"] == dashboard_location) &
                    (historical_df["Season"] == "Annual") &
                    (historical_df["Year"].between(BASELINE_PERIOD[0], BASELINE_PERIOD[1]))
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
                                preindustrial_baselines[key] = subset["Value"].mean()
                    baseline_source = f"Historical scenario ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]})"
                else:
                    baseline_source = "Historical scenario (no 1850-1900 data)"
            else:
                # If no historical scenario, try to use earliest available data from all scenarios
                earliest_year = int(df_all["Year"].min())
                earliest_data = df_all[
                    (df_all["Location"] == dashboard_location) &
                    (df_all["Season"] == "Annual") &
                    (df_all["Year"] == earliest_year)
                    ]

                for metric_type in earliest_data["Type"].unique():
                    for metric_name in earliest_data[earliest_data["Type"] == metric_type]["Name"].unique():
                        key = (metric_type, metric_name)
                        subset = earliest_data[
                            (earliest_data["Type"] == metric_type) &
                            (earliest_data["Name"] == metric_name)
                            ]
                        if not subset.empty:
                            preindustrial_baselines[key] = subset["Value"].mean()

                baseline_source = f"Earliest available ({earliest_year})"

            # Function to get 5-year average centered on target year
            def get_5year_average(data, metric_type, metric_name, scenario, center_year):
                """Get 5-year average centered on target year (center_year-2 to center_year+2)
                Falls back to available data if full 5-year window not available"""
                start_year = center_year - 2
                end_year = center_year + 2

                subset = data[
                    (data["Type"] == metric_type) &
                    (data["Name"] == metric_name) &
                    (data["Scenario"] == scenario) &
                    (data["Year"].between(start_year, end_year))
                    ]

                if not subset.empty:
                    return subset["Value"].mean()

                # If no data in 5-year window, try just the center year
                single_year = data[
                    (data["Type"] == metric_type) &
                    (data["Name"] == metric_name) &
                    (data["Scenario"] == scenario) &
                    (data["Year"] == center_year)
                    ]

                if not single_year.empty:
                    return single_year["Value"].mean()

                # If still no data, try closest available year within +/- 5 years
                nearby = data[
                    (data["Type"] == metric_type) &
                    (data["Name"] == metric_name) &
                    (data["Scenario"] == scenario) &
                    (data["Year"].between(center_year - 5, center_year + 5))
                    ]

                if not nearby.empty:
                    # Use the year closest to center_year
                    nearby = nearby.copy()
                    nearby["distance"] = abs(nearby["Year"] - center_year)
                    closest = nearby.nsmallest(min(5, len(nearby)), "distance")
                    return closest["Value"].mean()

                return None

            # Function to calculate both changes using 5-year averages
            def calculate_changes(data, metric_type, metric_name, scenario, from_year, to_year):
                baseline = get_5year_average(data, metric_type, metric_name, scenario, from_year)
                target = get_5year_average(data, metric_type, metric_name, scenario, to_year)

                change_from_start = None
                change_from_preindustrial = None

                if baseline is not None and target is not None:
                    change_from_start = target - baseline

                # Calculate change from pre-industrial baseline
                key = (metric_type, metric_name)
                if target is not None and key in preindustrial_baselines:
                    change_from_preindustrial = target - preindustrial_baselines[key]

                return change_from_start, change_from_preindustrial, target

        # Display dashboard for each scenario
        for scenario in dashboard_scenarios:
            st.markdown(f"---")
            st.subheader(f"🌍 {scenario}")

            # Create four columns for reference year plus three time horizons
            st.markdown("### 📅 Time Horizons")
            col_ref, col_short, col_mid, col_long = st.columns(4)

            with col_ref:
                st.markdown(f"<h2 style='text-align: center; color: #808080;'>{REFERENCE_YEAR}</h2>",
                            unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-weight: bold; margin-top: -10px;'>REFERENCE</p>",
                            unsafe_allow_html=True)
            with col_short:
                st.markdown(f"<h2 style='text-align: center; color: #3498DB;'>{mid_start}</h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-weight: bold; margin-top: -10px;'>SHORT TERM</p>",
                            unsafe_allow_html=True)
            with col_mid:
                st.markdown(f"<h2 style='text-align: center; color: #F39C12;'>{long_start}</h2>",
                            unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-weight: bold; margin-top: -10px;'>MEDIUM TERM</p>",
                            unsafe_allow_html=True)
            with col_long:
                st.markdown(f"<h2 style='text-align: center; color: #E74C3C;'>{horizon_end}</h2>",
                            unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-weight: bold; margin-top: -10px;'>LONG TERM</p>",
                            unsafe_allow_html=True)

            st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)

            # Display metrics by category
            for category, metrics in dashboard_metrics.items():
                st.markdown(f"<h3 style='margin: 12px 0 8px 0;'>{category}</h3>", unsafe_allow_html=True)

                for metric_type, metric_name, unit, icon in metrics:
                    # Find actual metric name in data
                    actual_metric_name = find_metric_name(dashboard_data, metric_type, metric_name)
                    display_name = actual_metric_name

                    # Create columns for each time horizon (including reference)
                    col_ref, col_short, col_mid, col_long = st.columns(4)

                    # Calculate changes for each horizon using actual metric name
                    change_ref, pi_change_ref, value_ref = calculate_changes(
                        dashboard_data, metric_type, actual_metric_name, scenario, baseline_year, REFERENCE_YEAR
                    )
                    change_short, pi_change_short, value_short = calculate_changes(
                        dashboard_data, metric_type, actual_metric_name, scenario, baseline_year, mid_start
                    )
                    change_mid, pi_change_mid, value_mid = calculate_changes(
                        dashboard_data, metric_type, actual_metric_name, scenario, baseline_year, long_start
                    )
                    change_long, pi_change_long, value_long = calculate_changes(
                        dashboard_data, metric_type, actual_metric_name, scenario, baseline_year, horizon_end
                    )

                    # Determine delta colour based on metric type
                    if metric_type == "Rain" and metric_name in ["CDD"]:
                        delta_color = "inverse"
                    else:
                        delta_color = "normal"

                    # Display reference year column
                    with col_ref:
                        if change_ref is not None:
                            color = '#00c853' if change_ref < 0 and delta_color == 'inverse' else '#d32f2f' if change_ref > 0 and delta_color == 'inverse' else '#00c853' if change_ref > 0 else '#d32f2f'

                            display_html = (
                                f"<div style='text-align: center; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background: #fafafa;'>"
                                f"<p style='font-size: 24px; margin: 0 0 8px 0; color: #333; font-weight: bold;'>{display_name}</p>"
                                f"<p style='font-size: 24px; font-weight: bold; color: {color}; margin: 3px 0;'>{change_ref:+.1f} {unit}</p>"
                                f"<p style='font-size: 11px; color: #999; margin: 0;'>from {baseline_year}</p>"
                            )

                            if pi_change_ref is not None:
                                pi_color = '#00c853' if pi_change_ref < 0 and delta_color == 'inverse' else '#d32f2f' if pi_change_ref > 0 and delta_color == 'inverse' else '#00c853' if pi_change_ref > 0 else '#d32f2f'
                                display_html += (
                                    f"<hr style='border: 0; border-top: 1px dashed #ccc; margin: 8px 0;'>"
                                    f"<p style='font-size: 24px; font-weight: bold; color: {pi_color}; margin: 3px 0;'>{pi_change_ref:+.1f} {unit}</p>"
                                    f"<p style='font-size: 11px; color: #999; margin: 0;'>Pre-Industrial</p>"
                                )

                            display_html += f"<p style='font-size: 12px; color: #666; margin: 8px 0 0 0;'>Total: {value_ref:.1f} {unit}</p></div>"
                            st.markdown(display_html, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<div style='text-align: center; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px;'>"
                                f"<p style='font-size: 24px; color: #333; font-weight: bold; margin: 0;'>{display_name}</p>"
                                f"<p style='font-size: 16px; color: #999; margin: 8px 0;'>N/A</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    # Display short term column
                    with col_short:
                        if change_short is not None:
                            color = '#00c853' if change_short < 0 and delta_color == 'inverse' else '#d32f2f' if change_short > 0 and delta_color == 'inverse' else '#00c853' if change_short > 0 else '#d32f2f'

                            display_html = (
                                f"<div style='text-align: center; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background: #fafafa;'>"
                                f"<p style='font-size: 24px; margin: 0 0 8px 0; color: #333; font-weight: bold;'>{display_name}</p>"
                                f"<p style='font-size: 24px; font-weight: bold; color: {color}; margin: 3px 0;'>{change_short:+.1f} {unit}</p>"
                                f"<p style='font-size: 11px; color: #999; margin: 0;'>from {baseline_year}</p>"
                            )

                            if pi_change_short is not None:
                                pi_color = '#00c853' if pi_change_short < 0 and delta_color == 'inverse' else '#d32f2f' if pi_change_short > 0 and delta_color == 'inverse' else '#00c853' if pi_change_short > 0 else '#d32f2f'
                                display_html += (
                                    f"<hr style='border: 0; border-top: 1px dashed #ccc; margin: 8px 0;'>"
                                    f"<p style='font-size: 24px; font-weight: bold; color: {pi_color}; margin: 3px 0;'>{pi_change_short:+.1f} {unit}</p>"
                                    f"<p style='font-size: 11px; color: #999; margin: 0;'>Pre-Industrial</p>"
                                )

                            display_html += f"<p style='font-size: 12px; color: #666; margin: 8px 0 0 0;'>Total: {value_short:.1f} {unit}</p></div>"
                            st.markdown(display_html, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<div style='text-align: center; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px;'>"
                                f"<p style='font-size: 24px; color: #333; font-weight: bold; margin: 0;'>{display_name}</p>"
                                f"<p style='font-size: 16px; color: #999; margin: 8px 0;'>N/A</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    # Display mid term column
                    with col_mid:
                        if change_mid is not None:
                            color = '#00c853' if change_mid < 0 and delta_color == 'inverse' else '#d32f2f' if change_mid > 0 and delta_color == 'inverse' else '#00c853' if change_mid > 0 else '#d32f2f'

                            display_html = (
                                f"<div style='text-align: center; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background: #fafafa;'>"
                                f"<p style='font-size: 24px; margin: 0 0 8px 0; color: #333; font-weight: bold;'>{display_name}</p>"
                                f"<p style='font-size: 24px; font-weight: bold; color: {color}; margin: 3px 0;'>{change_mid:+.1f} {unit}</p>"
                                f"<p style='font-size: 11px; color: #999; margin: 0;'>from {baseline_year}</p>"
                            )

                            if pi_change_mid is not None:
                                pi_color = '#00c853' if pi_change_mid < 0 and delta_color == 'inverse' else '#d32f2f' if pi_change_mid > 0 and delta_color == 'inverse' else '#00c853' if pi_change_mid > 0 else '#d32f2f'
                                display_html += (
                                    f"<hr style='border: 0; border-top: 1px dashed #ccc; margin: 8px 0;'>"
                                    f"<p style='font-size: 24px; font-weight: bold; color: {pi_color}; margin: 3px 0;'>{pi_change_mid:+.1f} {unit}</p>"
                                    f"<p style='font-size: 11px; color: #999; margin: 0;'>Pre-Industrial</p>"
                                )

                            display_html += f"<p style='font-size: 12px; color: #666; margin: 8px 0 0 0;'>Total: {value_mid:.1f} {unit}</p></div>"
                            st.markdown(display_html, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<div style='text-align: center; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px;'>"
                                f"<p style='font-size: 24px; color: #333; font-weight: bold; margin: 0;'>{display_name}</p>"
                                f"<p style='font-size: 16px; color: #999; margin: 8px 0;'>N/A</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    # Display long term column
                    with col_long:
                        if change_long is not None:
                            color = '#00c853' if change_long < 0 and delta_color == 'inverse' else '#d32f2f' if change_long > 0 and delta_color == 'inverse' else '#00c853' if change_long > 0 else '#d32f2f'

                            display_html = (
                                f"<div style='text-align: center; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background: #fafafa;'>"
                                f"<p style='font-size: 24px; margin: 0 0 8px 0; color: #333; font-weight: bold;'>{display_name}</p>"
                                f"<p style='font-size: 24px; font-weight: bold; color: {color}; margin: 3px 0;'>{change_long:+.1f} {unit}</p>"
                                f"<p style='font-size: 11px; color: #999; margin: 0;'>from {baseline_year}</p>"
                            )

                            if pi_change_long is not None:
                                pi_color = '#00c853' if pi_change_long < 0 and delta_color == 'inverse' else '#d32f2f' if pi_change_long > 0 and delta_color == 'inverse' else '#00c853' if pi_change_long > 0 else '#d32f2f'
                                display_html += (
                                    f"<hr style='border: 0; border-top: 1px dashed #ccc; margin: 8px 0;'>"
                                    f"<p style='font-size: 24px; font-weight: bold; color: {pi_color}; margin: 3px 0;'>{pi_change_long:+.1f} {unit}</p>"
                                    f"<p style='font-size: 11px; color: #999; margin: 0;'>Pre-Industrial</p>"
                                )

                            display_html += f"<p style='font-size: 12px; color: #666; margin: 8px 0 0 0;'>Total: {value_long:.1f} {unit}</p></div>"
                            st.markdown(display_html, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<div style='text-align: center; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px;'>"
                                f"<p style='font-size: 24px; color: #333; font-weight: bold; margin: 0;'>{display_name}</p>"
                                f"<p style='font-size: 16px; color: #999; margin: 8px 0;'>N/A</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)


# ================================== MAIN ======================================
if __name__ == "__main__":
    if MODE.lower() == "metrics":
        run_metrics_viewer()
    else:
        sys.exit('Set MODE = "metrics"')