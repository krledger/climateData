#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Viewer Constants and Configuration
===========================================
Last Updated: 2025-11-25 19:00 AEST

All configuration constants and settings for the Climate Metrics Viewer.
Consolidated from former config.py and constants.py modules.
"""

import os
import pandas as pd
import numpy as np

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

MODE = "metrics"
PORT = 8502

# ============================================================================
# EMOJI CONSTANTS (Unicode escapes - won't corrupt during file edits)
# ============================================================================

EMOJI = {
    # Navigation & UI
    "thermometer": "\U0001F321\uFE0F",  # Thermometer
    "chart": "\U0001F4CA",               # Bar chart
    "globe": "\U0001F30D",               # Globe/Earth
    "book": "\U0001F4D6",                # Open book
    "pin": "\U0001F4CD",                 # Pin/location
    "calendar": "\U0001F4C5",            # Calendar
    "map": "\U0001F5FA\uFE0F",           # World map
    "lightbulb": "\U0001F4A1",           # Light bulb
    "warning": "\u26A0\uFE0F",           # Warning sign
    "page": "\U0001F4C4",                # Page/document
    "search": "\U0001F50D",              # Magnifying glass
    "display": "\U0001F5A5\uFE0F",       # Desktop computer
    "graph": "\U0001F4C8",               # Chart increasing
    "wrench": "\U0001F527",              # Wrench/tool
    "info": "\u2139\uFE0F",              # Info
    # Status indicators
    "check": "\u2705",                   # Green check
    "x_mark": "\u274C",                  # Red X
    "green_circle": "\U0001F7E2",        # Green circle
    "yellow_circle": "\U0001F7E1",       # Yellow circle
    "red_circle": "\U0001F534",          # Red circle
    # Additional
    "wave": "\U0001F30A",                # Wave (for overview)
    "scroll": "\U0001F4DC",              # Scroll/document
    "link": "\U0001F517",                # Link
    "books": "\U0001F4DA",               # Stack of books
    "gear": "\u2699\uFE0F",              # Gear/settings
}

# ============================================================================
# FILE PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
FOLDER = os.path.join(BASE_DIR, "metricsDataFiles")

# ============================================================================
# DATA SCHEMA
# ============================================================================

GROUP_KEYS = ["Location", "Type", "Name", "Season", "Data Type"]
IDX_COLS_ANNUAL = ["Year"]
IDX_COLS_SEASONAL = ["Year", "Season"]

REQUIRED_COLUMNS = {
    "Year": "Int64",
    "Season": str,
    "Data Type": str,
    "Value": float,
    "Location": str,
    "Name": str,
    "Type": str,
    "Scenario": str
}

# ============================================================================
# CLIMATE PARAMETERS
# ============================================================================

# Baseline period for pre-industrial temperature calculations
BASELINE_PERIOD = (1850, 1900)

# Reference year for climate projections (default baseline for comparisons)
REFERENCE_YEAR = 2020

# Hardcoded baselines for fallback when historical data unavailable
HARDCODED_GLOBAL_BASELINE = 13.9  # deg C - IPCC AR6 global mean temperature 1850-1900
HARDCODED_AUSTRALIA_BASELINE = 21.7  # deg C - Australia mean temperature 1850-1900

# Global warming target for 1.5 deg C shading (degrees Celsius)
WARMING_TARGET = 1.5

# Regional warming amplification factor
# Regional warming is typically greater than global mean
# Australia/Ravenswood experiences ~1.15x global warming
AMPLIFICATION_FACTOR = 1.15

# ============================================================================
# TIME HORIZONS
# ============================================================================

TIME_HORIZONS = {
    'short_start_default': 2020,
    'mid_start_default': 2036,
    'long_start_default': 2039,
    'horizon_end_default': 2045
}

# ============================================================================
# SEASONS
# ============================================================================

SEASONS = {
    'all': ['Annual', 'DJF', 'MAM', 'JJA', 'SON'],
    'names': {
        'DJF': 'Summer (Dec-Feb)',
        'MAM': 'Autumn (Mar-May)',
        'JJA': 'Winter (Jun-Aug)',
        'SON': 'Spring (Sep-Nov)',
        'Annual': 'Annual'
    }
}

# ============================================================================
# METRIC PRIORITIES
# ============================================================================

# Metric display priorities (order in sidebar selector)
# Lower index = higher priority (appears first)
METRIC_PRIORITIES = {
    'Rain': [
        'Total',  # Raw total - default selection
        'Total (BC)',  # Bias-corrected total - second priority
        'Max Day',  # Extreme rainfall events
        'Max Day (BC)',
        'Max 5-Day',  # Multi-day accumulation
        'Max 5-Day (BC)',
        'R10mm',  # Wet day frequency
        'R10mm (BC)',
        'R20mm',  # Heavy rainfall days
        'R20mm (BC)',
        'Min Day',  # Less critical metrics
        'Min Day (BC)',
        'Min 5-Day',
        'Min 5-Day (BC)',
        'CDD',  # Consecutive dry days - last priority
        'CDD (BC)'
    ],
    'Temp': [
        'Average',
        'Max Day',
        'Avg Max',
        '5-Day Avg Max',
        'Min Day',
        'Avg Min',
        '5-Day Avg Min',
        'Days >37C 3hr',
        'Days >40C 3hr'
    ],
    'Wind': [
        'Average',
        '95th Percentile',
        'Max Day'
    ],
    'Humidity': [
        'Average'
    ],
    'VPD': [
        'Average'
    ]
}

# ============================================================================
# DASHBOARD METRICS
# ============================================================================

# Dashboard metrics configuration
# Format: (metric_type, metric_name, unit, icon, key)
DASHBOARD_METRICS = {
    'Temp': [
        ('Temp', 'Average', 'C', 'thermometer', 'temp_avg'),
        ('Temp', 'Max Day', 'C', 'thermometer', 'temp_max'),
        ('Temp', 'Days >37C 3hr', 'days', 'thermometer', 'temp_heat_days')
    ],
    'Rain': [
        ('Rain', 'Total', 'mm', 'wave', 'rain_total'),
        ('Rain', 'Max Day', 'mm', 'wave', 'rain_max'),
        ('Rain', 'CDD', 'days', 'warning', 'rain_cdd')
    ],
    'Wind': [
        ('Wind', 'Average', 'm/s', 'wave', 'wind_avg'),
        ('Wind', 'Max Day', 'm/s', 'wave', 'wind_max')
    ],
    'Humidity': [
        ('Humidity', 'Average', '%', 'wave', 'humidity_avg')
    ],
    'VPD': [
        ('VPD', 'Average', 'kPa', 'wave', 'vpd_avg')
    ]
}

# ============================================================================
# SCENARIO DESCRIPTIONS
# ============================================================================

SCENARIO_DESCRIPTIONS = {
    'SSP1-26': 'Sustainability pathway, net zero by 2075, ~1.8C warming',
    'SSP1-1.9': 'Very low emissions, net zero by 2050, ~1.5C warming',
    'SSP2-45': 'Middle-of-the-road, ~2.7C warming',
    'SSP3-70': 'Regional rivalry, ~3.6C warming',
    'SSP5-85': 'Fossil-fuelled development, ~4.4C warming',
    'historical': 'Historical model simulation (1850-2014)',
    'Historical': 'Historical model simulation (1850-2014)',
    'AGCD': 'Observed climate data from Australian Bureau of Meteorology'
}

# ============================================================================
# FEATURE FLAGS
# ============================================================================

try:
    import folium
    from streamlit_folium import st_folium

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_scenario_description(scenario: str) -> str:
    """
    Get description for a scenario.

    Args:
        scenario: Scenario name

    Returns:
        Description string
    """
    # Try exact match first
    if scenario in SCENARIO_DESCRIPTIONS:
        return SCENARIO_DESCRIPTIONS[scenario]

    # Try case-insensitive match
    for key, value in SCENARIO_DESCRIPTIONS.items():
        if key.lower() == scenario.lower():
            return value

    # Try partial match for SSP scenarios
    scenario_upper = scenario.upper()
    for key in SCENARIO_DESCRIPTIONS:
        if key.upper() in scenario_upper or scenario_upper in key.upper():
            return SCENARIO_DESCRIPTIONS[key]

    return "Climate scenario"


def should_use_inverse_delta(metric_type: str, metric_name: str) -> bool:
    """
    Determine if a metric should use inverse delta colouring.

    For most metrics, increases (positive deltas) are shown in red (bad).
    For some metrics like rainfall, increases might be good.

    Args:
        metric_type: Type of metric (Temp, Rain, Wind, Humidity)
        metric_name: Name of metric

    Returns:
        True if delta colours should be inverted (green=increase, red=decrease)
    """
    metric_name_lower = metric_name.lower()

    # CDD (Consecutive Dry Days) - increase is BAD, use normal colouring
    if 'cdd' in metric_name_lower or 'dry days' in metric_name_lower:
        return False

    # For rainfall totals - increase is generally GOOD in this region
    if metric_type == 'Rain':
        if 'total' in metric_name_lower:
            return True
        # Heavy rainfall/max could indicate flooding, use normal
        return False

    # Temperature/Wind/Humidity increases generally BAD, use normal colouring
    return False


# ============================================================================
# BIAS CORRECTION FACTORS
# ============================================================================
# Loaded from bias_correction_factors.json (generated by climateBiasFactors.py)
# Run: python Scenario_processing_code/climateBiasFactors.py to regenerate

import json as _json

def _load_bias_corrections():
    """Load bias correction factors from JSON file."""
    bias_file = os.path.join(BASE_DIR, "bias_correction_factors.json")
    if os.path.exists(bias_file):
        with open(bias_file, "r", encoding="utf-8") as f:
            data = _json.load(f)
            return data.get("corrections", {})
    else:
        # Return empty dict if file not found - no corrections applied
        print(f"[!] Bias correction file not found: {bias_file}")
        return {}

BIAS_CORRECTION_FACTORS = _load_bias_corrections()


# ============================================================================
# BIAS CORRECTION APPLICATION FUNCTION
# ============================================================================

def apply_bias_correction(value, scenario, location, season, metric_type, metric_name):
    """
    Apply bias correction to a metric value.

    Args:
        value: Raw metric value
        scenario: Scenario name (e.g. 'historical', 'SSP1-26')
        location: Location name (e.g. 'Australia', 'Ravenswood')
        season: Season name (e.g. 'DJF', 'Annual')
        metric_type: Metric type (e.g. 'Temp', 'Rain')
        metric_name: Metric name (e.g. 'Average', 'Total')

    Returns:
        Corrected value if BC applied, otherwise raw value
    """
    if value is None or pd.isna(value):
        return value

    # Only apply BC to Historical and SSP scenarios (not AGCD)
    if scenario.upper() == 'AGCD':
        return value

    # Check if this is a Historical or SSP scenario
    scenario_upper = scenario.upper()
    is_correctable = (scenario_upper == 'HISTORICAL' or
                     (scenario_upper.startswith('SSP') and any(c.isdigit() for c in scenario)))

    if not is_correctable:
        return value

    # Build metric key from type and name
    metric_key = f"{metric_type}_{metric_name}".replace(' ', '_').replace('(', '').replace(')', '')

    # Get correction factor
    try:
        bc_data = BIAS_CORRECTION_FACTORS[location][season][metric_key]
        correction = bc_data['value']
        correction_type = bc_data['type']

        if correction_type == 'additive':
            return value + correction
        elif correction_type == 'multiplicative':
            return value * correction
        else:
            return value

    except (KeyError, TypeError):
        # No correction available for this metric
        return value