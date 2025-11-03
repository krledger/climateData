import os
from climate_viewer_config import FOLIUM_AVAILABLE

if FOLIUM_AVAILABLE:
    import folium

def create_region_map(kml_path: str, ravenswood_kml_path: str = None, center_location: tuple = (-26.5, 133.5),
                      zoom: int = 4):
    if not FOLIUM_AVAILABLE:
        return None, "Folium not available", 0

    if not os.path.exists(kml_path):
        return None, "KML file not found", 0

    m = folium.Map(
        location=center_location,
        zoom_start=zoom,
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a>',
        control_scale=True
    )

    placemark_count = 0
    status_msg = "Success"

    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(kml_path)
        root = tree.getroot()

        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        placemarks = root.findall('.//kml:Placemark', ns)

        placemark_count = len(placemarks)

        for placemark in placemarks:
            name_elem = placemark.find('.//kml:name', ns)
            name = name_elem.text if name_elem is not None else 'Grid Cell'

            is_ravenswood = 'ravenswood' in name.lower()

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
                    if is_ravenswood:
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
                        folium.Polygon(
                            locations=coords_list,
                            popup=name,
                            color='grey',
                            fillColor='grey',
                            fillOpacity=0.2,
                            weight=1
                        ).add_to(m)
                elif len(coords_list) == 1:
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
