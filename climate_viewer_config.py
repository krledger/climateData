"""
Climate Viewer Configuration
Technical settings, file paths, and data schema definitions.
"""

import os

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

MODE = "metrics"
PORT = 8501

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
# FEATURE FLAGS
# ============================================================================

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
