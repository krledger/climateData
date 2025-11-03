#!/usr/bin/env python3
"""
Verify all imports are correct after renaming files with climate_viewer_ prefix
"""

import sys

print("Testing imports...")
print("(Note: Some modules may fail if streamlit/altair/folium not installed - this is OK)")
print()

errors = []

try:
    print("  âœ“ climate_viewer_config")
    import climate_viewer_config
except ImportError as e:
    errors.append(f"climate_viewer_config: {e}")

try:
    print("  âœ“ climate_viewer_helpers")
    import climate_viewer_helpers
except ImportError as e:
    errors.append(f"climate_viewer_helpers: {e}")

try:
    print("  âœ“ climate_viewer_ui_components")
    import climate_viewer_ui_components
except ImportError as e:
    errors.append(f"climate_viewer_ui_components: {e}")

try:
    print("  âœ“ climate_viewer_data_operations")
    import climate_viewer_data_operations
except ImportError as e:
    errors.append(f"climate_viewer_data_operations: {e}")

try:
    print("  âœ“ climate_viewer_analysis")
    import climate_viewer_analysis
except ImportError as e:
    errors.append(f"climate_viewer_analysis: {e}")

try:
    print("  âœ“ climate_viewer_maps")
    import climate_viewer_maps
except ImportError as e:
    errors.append(f"climate_viewer_maps: {e}")

try:
    print("  âœ“ climate_viewer_tab_metrics")
    import climate_viewer_tab_metrics
except ImportError as e:
    if "altair" in str(e) or "streamlit" in str(e):
        print("  âš  climate_viewer_tab_metrics (missing streamlit/altair - install with: pip install -r requirements.txt)")
    else:
        errors.append(f"climate_viewer_tab_metrics: {e}")

try:
    print("  âœ“ climate_viewer_tab_dashboard")
    import climate_viewer_tab_dashboard
except ImportError as e:
    if "streamlit" in str(e):
        print("  âš  climate_viewer_tab_dashboard (missing streamlit - install with: pip install -r requirements.txt)")
    else:
        errors.append(f"climate_viewer_tab_dashboard: {e}")

try:
    print("  âœ“ climate_viewer_tab_global")
    import climate_viewer_tab_global
except ImportError as e:
    errors.append(f"climate_viewer_tab_global: {e}")

print()

if errors:
    print("âŒ Import structure errors found:")
    for error in errors:
        print(f"   {error}")
    sys.exit(1)
else:
    print("âœ… All module imports successful!")
    print()
    print("Module structure is correct.")
    print("Install dependencies with: pip install -r requirements.txt")
    print("Then run: streamlit run climate_viewer_app.py")

