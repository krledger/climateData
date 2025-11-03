"""
Climate Viewer Constants
Centralized configuration for all hardcoded values, colors, metrics, and display settings.
"""

# ============================================================================
# BASELINE AND WARMING TARGETS (IPCC AR6 Standards)
# ============================================================================

# Pre-industrial baseline period (IPCC AR6 standard: 1850-1900)
# Used to calculate regional baseline temperatures from historical data
BASELINE_PERIOD = (1850, 1900)

# Global warming target (IPCC Paris Agreement target)
WARMING_TARGET = 1.5  # °C

# CRITICAL: Regional warming amplification factor (land warms faster than ocean)
# This is the SINGLE SOURCE OF TRUTH for the amplification factor
# Regional warming = Global warming × AMPLIFICATION_FACTOR
# Based on analysis: Regional warming / Global warming ratio for mid-latitude land
# DO NOT hardcode this value anywhere - always import from here
AMPLIFICATION_FACTOR = 1.149

# Reference year for dashboard comparisons
REFERENCE_YEAR = 2020

# Hardcoded baseline temperatures (°C) - FALLBACK VALUES ONLY
# These are used ONLY when historical data (1850-1900) is unavailable
# Primary method: Calculate from historical scenario data for the specific location
# Fallback method: Use these hardcoded values if calculation fails
HARDCODED_GLOBAL_BASELINE = 13.7    # °C (IPCC AR6 global mean)
HARDCODED_AUSTRALIA_BASELINE = 21.5  # °C (approximate Australia mean)

# Use hardcoded baseline or calculate from data
USE_HARDCODED_BASELINE = True


# ============================================================================
# DISPLAY COLORS
# ============================================================================

COLORS = {
    # Background and structural
    'background': '#F5F5F5',
    'panel_background': '#FFFFFF',
    'border': '#e0e0e0',
    
    # Header and text
    'header': '#2C5F7C',
    'text_primary': '#333',
    'text_secondary': '#666',
    'text_muted': '#999',
    'text_light': '#7F8C8D',
    
    # Temperature metrics
    'warm': '#E74C3C',
    'hot': '#FF6B6B',
    'cool': '#3498DB',
    
    # Precipitation metrics
    'precip': '#27AE60',
    'precip_negative': '#E67E22',
    'drought': '#E67E22',
    
    # Wind metrics
    'wind': '#8E44AD',
    'wind_secondary': '#9B59B6',
    
    # Time horizons (match slider colors)
    'horizon_short': '#3498DB',    # Blue
    'horizon_mid': '#F39C12',      # Orange
    'horizon_long': '#E74C3C',     # Red
    
    # Delta indicators
    'positive': '#00c853',
    'negative': '#d32f2f',
}


# ============================================================================
# DASHBOARD METRICS CONFIGURATION
# ============================================================================

DASHBOARD_METRICS = {
    "🌡️ Temperature": [
        ("Temp", "Average", "°C", "🌡️", "tas"),
        ("Temp", "5-Day Avg Max", "°C", "☀️", "5day_avg_max"),
        ("Temp", "Days>=37", "days", "🔥", "days37"),
    ],
    "🌧️ Precipitation": [
        ("Rain", "Max 5-Day", "mm", "🌊", "max_5day"),
        ("Rain", "R20", "days", "💧", "r20"),
        ("Rain", "Total", "mm", "☔", "total"),
    ],
    "🏜️ Drought": [
        ("Rain", "CDD", "days", "🏜️", "cdd"),
    ],
    "💨 Wind": [
        ("Wind", "95th Percentile", "m/s", "💨", "wind_95p"),
        ("Wind", "Average", "m/s", "🌬️", "wind_avg"),
    ],
}


# ============================================================================
# ANALYSIS METRICS (for extract_conditions functions)
# ============================================================================

ANALYSIS_METRICS = {
    'temperature': [
        ("Temp", "Average", "tas", "°C"),
        ("Temp", "5-Day Avg Max", "5day_avg_max", "°C"),
        ("Temp", "Days>37", "days_over_37", "days"),
        ("Temp", "Days>=37", "days_over_37", "days"),  # Alternative name
    ],
    'precipitation': [
        ("Rain", "CDD", "cdd", "days"),
        ("Rain", "Max 5-Day", "max_5day", "mm"),
        ("Rain", "Total", "total", "mm"),
        ("Rain", "R20", "r20", "days"),
    ],
    'wind': [
        ("Wind", "95th Percentile", "wind_95p", "m/s"),
        ("Wind", "Max Day", "wind_max", "m/s"),
        ("Wind", "Average", "wind_avg", "m/s"),
    ],
}

# Flattened list for iteration
ALL_ANALYSIS_METRICS = (
    ANALYSIS_METRICS['temperature'] + 
    ANALYSIS_METRICS['precipitation'] + 
    ANALYSIS_METRICS['wind']
)


# ============================================================================
# METRIC PRIORITIES FOR UI ORDERING
# ============================================================================

METRIC_PRIORITIES = {
    'Temp': ["Average", "Max", "Max Day", "5-Day Avg Max", "Avg Max"],
    'Humidity': ["Average RH", "Average VPD"],
}


# ============================================================================
# DELTA COLOR LOGIC
# ============================================================================

# Metrics where increase is bad (inverse coloring)
INVERSE_DELTA_METRICS = {
    ("Rain", "CDD"),  # Consecutive dry days
}


# ============================================================================
# TIME HORIZON DEFAULTS
# ============================================================================

TIME_HORIZONS = {
    'short_start_default': 2020,
    'mid_start_default': 2036,
    'long_start_default': 2039,
    'horizon_end_default': 2045,
}


# ============================================================================
# GEOGRAPHY
# ============================================================================

AUSTRALIA_BOUNDS = {
    'lat': (-45, -10),
    'lon': (110, 155),
}

# Default map center and zoom
MAP_DEFAULTS = {
    'center': (-26.5, 133.5),
    'zoom': 4,
}


# ============================================================================
# SEASONS
# ============================================================================

SEASONS = {
    'all': ["Annual", "DJF", "MAM", "JJA", "SON"],
    'order': ["DJF", "MAM", "JJA", "SON"],  # For sorting (excluding Annual)
}


# ============================================================================
# DATA DISPLAY
# ============================================================================

# Table interval options (years)
TABLE_INTERVALS = [1, 2, 5, 10]

# Smoothing window defaults
SMOOTHING = {
    'min': 3,
    'max': 21,
    'step': 2,
    'default': 9,
}


# ============================================================================
# CHART DIMENSIONS
# ============================================================================

CHART_CONFIG = {
    'height': 450,
    'point_size': 50,
    'line_width': 2.5,
    'opacity_selected': 1.0,
    'opacity_unselected': 0.2,
    'shading_opacity': 0.15,
}


# ============================================================================
# DASHBOARD DISPLAY
# ============================================================================

DASHBOARD_DISPLAY = {
    'card_style': """
        text-align: center;
        border: 2px solid {border_color};
        border-radius: 8px;
        padding: 10px;
        background: {bg_color};
    """,
    'metric_font_size': '24px',
    'delta_font_size': '24px',
    'label_font_size': '11px',
    'total_font_size': '12px',
}


# ============================================================================
# SCENARIO DESCRIPTIONS
# ============================================================================

SCENARIO_DESCRIPTIONS = {
    'SSP1-26': 'Paris Agreement pathway (strong mitigation)',
    'SSP1-1.9': 'Very strong mitigation (1.5°C target)',
    'SSP2-45': 'Middle-of-road pathway (moderate action)',
    'SSP3-70': 'Middle pathway with overshoot',
    'SSP5-85': 'High emissions pathway (limited action)',
}


# ============================================================================
# FILE PATTERNS
# ============================================================================

FILE_PATTERNS = {
    'metrics': 'metrics*.parquet',
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_scenario_description(scenario_name: str) -> str:
    """Get human-readable description for a scenario."""
    for key, desc in SCENARIO_DESCRIPTIONS.items():
        if key in scenario_name:
            return desc
    return "Future projection"


def should_use_inverse_delta(metric_type: str, metric_name: str) -> bool:
    """Check if metric should use inverse delta coloring (increase = bad)."""
    return (metric_type, metric_name) in INVERSE_DELTA_METRICS


def get_regional_warming_at_global_target(global_target: float = WARMING_TARGET) -> float:
    """Calculate regional warming when global reaches target (e.g., 1.5°C)."""
    return global_target * AMPLIFICATION_FACTOR
