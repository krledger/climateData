# -*- coding: utf-8 -*-
"""
Climate Viewer Maps Module
===========================
climate_viewer_maps.py
Last Updated: 2025-12-02 15:00 AEST - Polygons only, taller map, optimised

Features:
- Displays grid cell polygons for ALL selected locations
- Auto-discovers KML files by naming convention
- Each location displayed in a distinct colour
- Uses OpenStreetMap tiles (roads, towns, cities visible)

KML File Naming Convention:
- {location}_grid_coverage.kml  (e.g., australia_grid_coverage.kml)
- {location}_grid_cell.kml      (e.g., ravenswood_grid_cell.kml)

Adding a new location (e.g., Queensland):
1. Run metrics generator for Queensland
2. Run generator_kml.py to create queensland_grid_coverage.kml
3. Map viewer automatically discovers and displays it
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

from climate_viewer_constants import FOLIUM_AVAILABLE

if FOLIUM_AVAILABLE:
    import folium

# Map height in pixels - tall enough to show full Australia
MAP_HEIGHT = 550

# Distinct colours for locations - extendable
LOCATION_COLOURS = {
    "australia": {"color": "#3498DB", "fill": "#3498DB", "name": "Blue"},
    "ravenswood": {"color": "#E74C3C", "fill": "#E74C3C", "name": "Red"},
    "queensland": {"color": "#27AE60", "fill": "#27AE60", "name": "Green"},
    "new_south_wales": {"color": "#9B59B6", "fill": "#9B59B6", "name": "Purple"},
    "victoria": {"color": "#F39C12", "fill": "#F39C12", "name": "Orange"},
    "southeast_australia": {"color": "#1ABC9C", "fill": "#1ABC9C", "name": "Teal"},
    "tasmania": {"color": "#E91E63", "fill": "#E91E63", "name": "Pink"},
    "south_australia": {"color": "#795548", "fill": "#795548", "name": "Brown"},
    "western_australia": {"color": "#607D8B", "fill": "#607D8B", "name": "Grey-Blue"},
    "northern_territory": {"color": "#FF5722", "fill": "#FF5722", "name": "Deep Orange"},
}

DEFAULT_COLOUR = {"color": "#95A5A6", "fill": "#95A5A6", "name": "Grey"}


def get_location_colour(location: str) -> dict:
    """Get colour scheme for a location (case-insensitive, handles underscores/spaces)."""
    key = location.lower().replace(" ", "_")
    return LOCATION_COLOURS.get(key, DEFAULT_COLOUR)


def discover_kml_files(base_dir: Path) -> dict:
    """
    Discover all KML files in the base directory.

    Returns dict mapping location names to KML file paths.
    """
    kml_files = {}

    if not base_dir.exists():
        return kml_files

    for kml_path in base_dir.glob("*.kml"):
        filename = kml_path.stem.lower()

        # Extract location name from filename
        if filename.endswith("_grid_coverage"):
            location = filename.replace("_grid_coverage", "")
        elif filename.endswith("_grid_cell"):
            location = filename.replace("_grid_cell", "")
        else:
            location = filename

        # Normalise location name
        location_key = location.replace("_", " ").title()

        # Special cases
        if location == "australia":
            location_key = "Australia"
        elif location == "ravenswood":
            location_key = "Ravenswood"

        kml_files[location_key] = kml_path

    return kml_files


def parse_kml_polygons(kml_path: Path) -> list:
    """
    Parse polygon coordinates from a KML file.

    Returns list of dicts with 'name', 'coords', 'description'.
    Only returns polygons (3+ coordinates), skips points and boundary boxes.
    """
    if not kml_path or not kml_path.exists():
        return []

    polygons = []

    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()

        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        placemarks = root.findall('.//kml:Placemark', ns)

        for placemark in placemarks:
            name_elem = placemark.find('.//kml:name', ns)
            name = name_elem.text if name_elem is not None else 'Grid Cell'

            # Skip boundary/region boxes (typically named "Australia Region", "Download Boundary", etc.)
            name_lower = name.lower()
            if any(skip in name_lower for skip in ['boundary', 'region', 'extent', 'download']):
                continue

            # Skip point markers (Target Point, Cell Centre, etc.)
            if any(skip in name_lower for skip in ['target', 'centre', 'center', 'point']):
                continue

            desc_elem = placemark.find('.//kml:description', ns)
            description = desc_elem.text if desc_elem is not None else ''

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

                # Only include valid polygons (3+ points)
                if len(coords_list) >= 3:
                    polygons.append({
                        'name': name,
                        'coords': coords_list,
                        'description': description
                    })

    except Exception:
        pass

    return polygons


def create_region_map(kml_path: str, ravenswood_kml_path: str = None,
                      center_location: tuple = (-26.5, 133.5), zoom: int = 4):
    """
    Original single-KML map function (backwards compatible).

    For multi-location support, use create_multi_location_map() instead.
    """
    if not FOLIUM_AVAILABLE:
        return None, "Folium not available", 0

    if not os.path.exists(kml_path):
        return None, "KML file not found", 0

    m = folium.Map(
        location=center_location,
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )

    placemark_count = 0
    status_msg = "Success"

    try:
        polygons = parse_kml_polygons(Path(kml_path))
        placemark_count = len(polygons)

        for poly in polygons:
            coords = poly['coords']
            name = poly['name']
            is_ravenswood = 'ravenswood' in name.lower()

            if is_ravenswood:
                folium.Polygon(
                    locations=coords,
                    popup=f"<b>{name}</b>",
                    color='#E74C3C',
                    fill=True,
                    fillColor='#E74C3C',
                    fillOpacity=0.3,
                    weight=2
                ).add_to(m)
            else:
                folium.Polygon(
                    locations=coords,
                    popup=name,
                    color='grey',
                    fillColor='grey',
                    fillOpacity=0.2,
                    weight=1
                ).add_to(m)

    except Exception as e:
        status_msg = f"Partial load: {str(e)[:100]}"

    # Load Ravenswood overlay if provided
    if ravenswood_kml_path and os.path.exists(ravenswood_kml_path):
        try:
            rv_polygons = parse_kml_polygons(Path(ravenswood_kml_path))

            for poly in rv_polygons:
                coords = poly['coords']
                name = poly['name']

                folium.Polygon(
                    locations=coords,
                    popup=f"<b>{name}</b>",
                    color='#E74C3C',
                    fill=True,
                    fillColor='#E74C3C',
                    fillOpacity=0.3,
                    weight=2
                ).add_to(m)
                placemark_count += 1

        except Exception as e:
            if "Success" in status_msg:
                status_msg = f"Main KML loaded, Ravenswood KML issue: {str(e)[:50]}"

    return m, status_msg, placemark_count


def create_multi_location_map(locations: list, base_dir: Path = None) -> tuple:
    """
    Create a map showing grid cells for multiple locations.

    Args:
        locations: List of location names to display
        base_dir: Directory containing KML files (defaults to module directory)

    Returns:
        Tuple of (folium.Map, dict of loaded locations info)

    Each location is displayed in a distinct colour.
    KML files are auto-discovered by naming convention.
    """
    if not FOLIUM_AVAILABLE:
        return None, {"error": "Folium not available"}

    if base_dir is None:
        base_dir = Path(__file__).parent

    # Discover available KML files
    available_kmls = discover_kml_files(base_dir)

    # Determine map centre and zoom
    has_point_location = any(
        loc.lower() in ["ravenswood"]
        for loc in locations
    )
    has_australia = "Australia" in locations

    if has_point_location and not has_australia:
        center = (-20.5, 147.0)
        zoom = 6
    else:
        center = (-26.5, 134.0)
        zoom = 4

    # Create map
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )

    loaded_locations = {}

    # Process each location
    for location in locations:
        # Find matching KML file
        kml_path = None
        for kml_loc, path in available_kmls.items():
            if kml_loc.lower() == location.lower():
                kml_path = path
                break

        if kml_path is None:
            continue

        # Get colour for this location
        colours = get_location_colour(location)

        # Parse KML (only polygons, no points or boundaries)
        polygons = parse_kml_polygons(kml_path)

        if not polygons:
            continue

        # Create feature group for this location
        fg = folium.FeatureGroup(name=location)

        cell_count = 0
        for poly in polygons:
            coords = poly['coords']
            name = poly['name']
            desc = poly.get('description', '')

            # Truncate long descriptions
            if len(desc) > 100:
                desc = desc[:100] + "..."

            # Use thicker border for single-cell locations (e.g., Ravenswood)
            is_single_cell = len(polygons) <= 3
            weight = 3 if is_single_cell else 1
            fill_opacity = 0.4 if is_single_cell else 0.2

            folium.Polygon(
                locations=coords,
                popup=f"<b>{name}</b><br>{location}<br>{desc}",
                color=colours['color'],
                weight=weight,
                fill=True,
                fillColor=colours['fill'],
                fillOpacity=fill_opacity
            ).add_to(fg)
            cell_count += 1

        fg.add_to(m)
        loaded_locations[location] = {
            'cells': cell_count,
            'colour_name': colours['name'],
            'colour_hex': colours['color']
        }

    # Add layer control if multiple locations
    if len(loaded_locations) > 1:
        folium.LayerControl().add_to(m)

    return m, loaded_locations


def get_map_height() -> int:
    """Return the standard map height in pixels."""
    return MAP_HEIGHT