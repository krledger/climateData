#!/usr/bin/env python3
"""
Generate KML files from downloaded climate data metadata.
Shows EXACTLY what's in the JSON - the actual cells used in analysis.
Pure verification tool - no filtering, no modifications.
"""

import json
from pathlib import Path
import sys
import numpy as np
from typing import List, Tuple


def load_all_metadata(metrics_dir: Path) -> List[Tuple[Path, dict, str]]:
    """Load all metadata files. Returns list of (path, metadata, scenario_name)"""
    json_files = list(metrics_dir.rglob("raw_*.json"))

    if not json_files:
        return []

    metadata_list = []
    for json_file in json_files:
        scenario = json_file.parent.name
        resolution = "monthly" if "monthly" in json_file.name else "daily"
        scenario_id = f"{scenario}_{resolution}"

        with open(json_file, 'r') as f:
            metadata = json.load(f)

        metadata_list.append((json_file, metadata, scenario_id))

    return metadata_list


def validate_grid_consistency(metadata_list: List[Tuple[Path, dict, str]]) -> Tuple[bool, List[str]]:
    """Validate that all scenarios use the same grid"""
    issues = []

    grids = {}
    for path, metadata, scenario_id in metadata_list:
        grid_info = metadata.get('grid_info')

        if not grid_info:
            issues.append(f"❌ {scenario_id}: Missing grid_info")
            continue

        fingerprint = grid_info.get('fingerprint')
        if not fingerprint:
            issues.append(f"⚠️  {scenario_id}: No grid fingerprint")
            continue

        grids[scenario_id] = {
            'fingerprint': fingerprint,
            'model': metadata.get('model'),
            'land_cells': len(grid_info.get('land_sea_mask', {}).get('land_cell_indices', []))
        }

    if len(grids) == 0:
        issues.append("❌ No valid grid information found")
        return False, issues

    fingerprints = set(data['fingerprint'] for data in grids.values())

    if len(fingerprints) == 1:
        issues.append(f"✅ All {len(grids)} scenarios use IDENTICAL grids")
        issues.append(f"   Grid fingerprint: {list(fingerprints)[0]}")

        # Show land cell counts for each scenario
        for scenario_id, data in grids.items():
            issues.append(f"   {scenario_id}: {data['land_cells']} land cells")

        return True, issues

    issues.append(f"❌ GRID MISMATCH: {len(fingerprints)} different grids detected")
    return False, issues


def generate_ravenswood_kml(metadata: dict, output_path: Path, scenario_name: str) -> None:
    """Generate KML for Ravenswood extraction point"""

    grid_info = metadata.get('grid_info')
    if not grid_info:
        print(f"ERROR: No grid_info in metadata")
        return

    ravenswood_data = grid_info.get('ravenswood')
    if not ravenswood_data:
        print(f"ERROR: No Ravenswood data in metadata")
        return

    target = ravenswood_data['target']
    centre = ravenswood_data['cell_centre']
    bounds = ravenswood_data['cell_bounds']
    offset = ravenswood_data['offset_from_target']
    indices = ravenswood_data['grid_indices']

    lat_min, lat_max = bounds['lat']
    lon_min, lon_max = bounds['lon']
    lat_size = lat_max - lat_min
    lon_size = lon_max - lon_min

    avg_lat = (lat_min + lat_max) / 2
    lat_km = lat_size * 111.0
    lon_km = lon_size * 111.0 * abs(np.cos(np.radians(avg_lat)))

    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Ravenswood Grid Cell - {scenario_name}</name>
    <description>
        Actual grid cell used for Ravenswood point extraction
        Model: {metadata.get('model', 'unknown')}
        Scenario: {scenario_name}
    </description>

    <Style id="targetStyle">
        <IconStyle>
            <color>ffff0000</color><scale>1.5</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href></Icon>
        </IconStyle>
        <LabelStyle><color>ffffffff</color><scale>1.2</scale></LabelStyle>
    </Style>
    <Style id="centreStyle">
        <IconStyle>
            <color>ff00ff00</color><scale>1.2</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
        </IconStyle>
        <LabelStyle><color>ffffffff</color><scale>1.0</scale></LabelStyle>
    </Style>
    <Style id="cellStyle">
        <LineStyle><color>ff00ff00</color><width>3</width></LineStyle>
        <PolyStyle><color>4400ff00</color></PolyStyle>
    </Style>

    <Folder>
        <name>Ravenswood Extraction</name>

        <Placemark>
            <name>Target Point</name>
            <description>Requested: {target['lat']:.6f}°, {target['lon']:.6f}°</description>
            <styleUrl>#targetStyle</styleUrl>
            <Point>
                <coordinates>{target['lon']},{target['lat']},0</coordinates>
            </Point>
        </Placemark>

        <Placemark>
            <name>Cell Centre</name>
            <description>
Actual: {centre['lat']:.6f}°, {centre['lon']:.6f}°
Grid indices: [{indices[0]}, {indices[1]}]
Offset: {offset['lat']:+.6f}° lat, {offset['lon']:+.6f}° lon
            </description>
            <styleUrl>#centreStyle</styleUrl>
            <Point>
                <coordinates>{centre['lon']},{centre['lat']},0</coordinates>
            </Point>
        </Placemark>

        <Placemark>
            <name>Actual Grid Cell</name>
            <description>
Model: {metadata.get('model')}
Grid indices: [{indices[0]}, {indices[1]}]
Centre: {centre['lat']:.6f}°, {centre['lon']:.6f}°
Bounds: {lat_min:.6f}° to {lat_max:.6f}° (lat)
        {lon_min:.6f}° to {lon_max:.6f}° (lon)
Size: ~{lat_km:.1f} × ~{lon_km:.1f} km
Area: ~{lat_km * lon_km:.0f} km²

ALL Ravenswood metrics extracted from this cell
            </description>
            <styleUrl>#cellStyle</styleUrl>
            <Polygon>
                <extrude>0</extrude>
                <altitudeMode>clampToGround</altitudeMode>
                <outerBoundaryIs>
                    <LinearRing>
                        <coordinates>
                            {lon_min},{lat_max},0
                            {lon_max},{lat_max},0
                            {lon_max},{lat_min},0
                            {lon_min},{lat_min},0
                            {lon_min},{lat_max},0
                        </coordinates>
                    </LinearRing>
                </outerBoundaryIs>
            </Polygon>
        </Placemark>

    </Folder>
</Document>
</kml>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(kml)


def generate_australia_grid_kml(metadata: dict, output_path: Path, scenario_name: str) -> None:
    """Generate KML showing cells from JSON metadata - pure verification"""

    grid_info = metadata.get('grid_info')
    if not grid_info:
        print(f"ERROR: No grid_info in metadata")
        return

    lat_values = grid_info.get('lat_values', [])
    lon_values = grid_info.get('lon_values', [])

    if not lat_values or not lon_values:
        print(f"ERROR: No lat/lon values in metadata")
        return

    lat_spacing = abs(lat_values[1] - lat_values[0]) if len(lat_values) > 1 else 0
    lon_spacing = abs(lon_values[1] - lon_values[0]) if len(lon_values) > 1 else 0

    # Get cells from JSON metadata
    land_sea_mask = grid_info.get('land_sea_mask', {})
    mask_applied = land_sea_mask.get('applied', False)

    if mask_applied:
        # Land masking was applied - show only the cells listed in JSON
        land_cell_indices = land_sea_mask.get('land_cell_indices', [])
        cells_to_show = land_cell_indices
        description = f"Land cells used in regional statistics: {len(cells_to_show)}"
    else:
        # No masking - show all cells
        n_lat = len(lat_values)
        n_lon = len(lon_values)
        cells_to_show = [[i, j] for i in range(n_lat) for j in range(n_lon)]
        description = f"All grid cells (no land-sea masking): {len(cells_to_show)}"

    ravenswood_data = grid_info.get('ravenswood', {})
    ravenswood_indices = ravenswood_data.get('grid_indices', [None, None])

    area_info = metadata.get('geographical_area', {})
    if not area_info:
        area_info = metadata.get('geographic_area', {})

    # Sample cells if too many
    sample_rate = max(1, len(cells_to_show) // 500)

    kml_parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Australia Grid - {scenario_name}</name>
    <description>
{description}
Model: {metadata.get('model', 'unknown')}
Scenario: {scenario_name}
Coverage: {lat_values[0]:.2f}° to {lat_values[-1]:.2f}° (lat), {lon_values[0]:.2f}° to {lon_values[-1]:.2f}° (lon)
Cell size: ~{lat_spacing:.3f}° × {lon_spacing:.3f}°
    </description>

    <Style id="gridCellStyle">
        <LineStyle><color>ff00ff00</color><width>2</width></LineStyle>
        <PolyStyle><color>4400ff00</color></PolyStyle>
    </Style>
    <Style id="ravenswoodCellStyle">
        <LineStyle><color>ffff00ff</color><width>3</width></LineStyle>
        <PolyStyle><color>66ff00ff</color></PolyStyle>
    </Style>
    <Style id="boundaryStyle">
        <LineStyle><color>ffff0000</color><width>3</width></LineStyle>
        <PolyStyle><color>00000000</color></PolyStyle>
    </Style>

    <Folder>
        <name>Grid Cells from JSON</name>
        <description>{description}</description>
''']

    cell_count = 0
    ravenswood_found = False

    for idx, (i_lat, i_lon) in enumerate(cells_to_show):
        is_ravenswood = (i_lat == ravenswood_indices[0] and i_lon == ravenswood_indices[1])

        # Sample (always include Ravenswood)
        if not is_ravenswood and (idx % sample_rate != 0):
            continue

        lat_centre = lat_values[i_lat]
        lon_centre = lon_values[i_lon]

        lat_half = lat_spacing / 2
        lon_half = lon_spacing / 2

        cell_lat_min = lat_centre - lat_half
        cell_lat_max = lat_centre + lat_half
        cell_lon_min = lon_centre - lon_half
        cell_lon_max = lon_centre + lon_half

        if is_ravenswood:
            style = "ravenswoodCellStyle"
            name = f"RAVENSWOOD [{i_lat}, {i_lon}]"
            ravenswood_found = True
        else:
            style = "gridCellStyle"
            name = f"Cell [{i_lat}, {i_lon}]"

        avg_lat = (cell_lat_min + cell_lat_max) / 2
        lat_km = abs(cell_lat_max - cell_lat_min) * 111.0
        lon_km = abs(cell_lon_max - cell_lon_min) * 111.0 * abs(np.cos(np.radians(avg_lat)))

        kml_parts.append(f'''
        <Placemark>
            <name>{name}</name>
            <description>
Grid indices: [{i_lat}, {i_lon}]
Centre: {lat_centre:.4f}°, {lon_centre:.4f}°
Bounds: {cell_lat_min:.4f}° to {cell_lat_max:.4f}° (lat)
        {cell_lon_min:.4f}° to {cell_lon_max:.4f}° (lon)
Size: ~{lat_km:.1f} × ~{lon_km:.1f} km
Area: ~{lat_km * lon_km:.0f} km²
            </description>
            <styleUrl>#{style}</styleUrl>
            <Polygon>
                <extrude>0</extrude>
                <altitudeMode>clampToGround</altitudeMode>
                <outerBoundaryIs>
                    <LinearRing>
                        <coordinates>
                            {cell_lon_min},{cell_lat_max},0
                            {cell_lon_max},{cell_lat_max},0
                            {cell_lon_max},{cell_lat_min},0
                            {cell_lon_min},{cell_lat_min},0
                            {cell_lon_min},{cell_lat_max},0
                        </coordinates>
                    </LinearRing>
                </outerBoundaryIs>
            </Polygon>
        </Placemark>
''')
        cell_count += 1

    kml_parts.append('    </Folder>\n')

    # Add boundary
    if area_info:
        north = area_info.get('north')
        south = area_info.get('south')
        east = area_info.get('east')
        west = area_info.get('west')

        if None not in [north, south, east, west]:
            kml_parts.append(f'''
    <Folder>
        <name>Download Boundary</name>
        <Placemark>
            <name>Australia Region</name>
            <description>Bounds: {north}°N, {south}°S, {east}°E, {west}°W</description>
            <styleUrl>#boundaryStyle</styleUrl>
            <Polygon>
                <extrude>0</extrude>
                <altitudeMode>clampToGround</altitudeMode>
                <outerBoundaryIs>
                    <LinearRing>
                        <coordinates>
                            {west},{north},0
                            {east},{north},0
                            {east},{south},0
                            {west},{south},0
                            {west},{north},0
                        </coordinates>
                    </LinearRing>
                </outerBoundaryIs>
            </Polygon>
        </Placemark>
    </Folder>
''')

    kml_parts.append('</Document>\n</kml>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(kml_parts))

    print(f"   Generated {cell_count} cells (sampled from {len(cells_to_show)} total)")
    if not ravenswood_found and ravenswood_indices[0] is not None:
        print(f"   ⚠️  Ravenswood not in sample")


def main():
    print("=" * 70)
    print("CLIMATE GRID KML GENERATOR - VERIFICATION TOOL")
    print("=" * 70)
    print("Shows cells exactly as recorded in JSON metadata")
    print()

    metrics_dir = Path(__file__).parent / "metricsDataFiles"

    if not metrics_dir.exists():
        print(f"ERROR: {metrics_dir} not found")
        print("Run climateCMIP6ImporterWeb.py first to generate data")
        sys.exit(1)

    print("Loading metadata...")
    metadata_list = load_all_metadata(metrics_dir)

    if not metadata_list:
        print("ERROR: No metadata JSON files found")
        sys.exit(1)

    print(f"Found {len(metadata_list)} scenario(s)")
    print()

    print("=" * 70)
    print("GRID CONSISTENCY CHECK")
    print("=" * 70)
    is_consistent, issues = validate_grid_consistency(metadata_list)

    for issue in issues:
        print(issue)
    print("=" * 70)
    print()

    if is_consistent:
        print("✓ Generating KML files from first scenario metadata")
        print("  (All scenarios use identical grid)")
        print()

        _, metadata, scenario_id = metadata_list[0]

        ravenswood_path = Path(__file__).parent / "ravenswood_grid_cell.kml"
        generate_ravenswood_kml(metadata, ravenswood_path, "All Scenarios")
        print(f"✓ {ravenswood_path.name}")

        australia_path = Path(__file__).parent / "australia_grid_coverage.kml"
        generate_australia_grid_kml(metadata, australia_path, "All Scenarios")
        print(f"✓ {australia_path.name}")

    else:
        print("⚠️  Generating separate KML files per scenario")
        print()

        for json_file, metadata, scenario_id in metadata_list:
            if 'grid_info' not in metadata:
                continue

            ravenswood_name = f"ravenswood_{scenario_id}.kml"
            ravenswood_path = Path(__file__).parent / ravenswood_name
            generate_ravenswood_kml(metadata, ravenswood_path, scenario_id)
            print(f"   {ravenswood_name}")

            australia_name = f"australia_{scenario_id}.kml"
            australia_path = Path(__file__).parent / australia_name
            generate_australia_grid_kml(metadata, australia_path, scenario_id)
            print(f"   {australia_name}")

    print()
    print("=" * 70)
    print("COMPLETE - KML files show cells from JSON metadata")
    print("=" * 70)
    print()
    print("Open in Google Earth to verify:")
    print("  • Which cells were used in regional statistics")
    print("  • Ravenswood point extraction location")
    print("  • Grid cell boundaries and sizes")
    print("=" * 70)


if __name__ == "__main__":
    main()