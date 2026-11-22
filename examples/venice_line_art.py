"""Draw a compact line-art map of Venice from OpenStreetMap ways."""

import argparse
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from examples._venice_hatching import hatch_segments
from sdf_ui.export import write_svg
from sdf_ui.fonts import default_font_path
from sdf_ui.vector import VectorDrawing, VectorPath

OUTPUT = Path(__file__).resolve().parents[1] / "out" / "examples" / "venice_line_art"
BOUNDS = (45.429, 12.317, 45.446, 12.345)
WIDTH, HEIGHT = 1200, 800
MAP = (234, 120, 966, 744)
PAPER = "#f4f0e7"
INK = "#263d43"
SECONDARY = "#607478"
BUILDING_HATCH = "#9aa5a0"
BUILDING_SPACING = 4.0
BUILDING_HATCH_WIDTH = 0.32
OSM_COPYRIGHT = "https://www.openstreetmap.org/copyright"
WATER_COLOR = "#7199a0"
WATER_LAYER = "water"
BUILDINGS_LAYER = "buildings"
STREETS_LAYER = "streets"
NATURAL_TAG = "natural"
WATER_TAG = "water"
WATERWAY_TAG = "waterway"
CANAL = "canal"
WATERWAY_AREA = {"canal", "river", "riverbank"}
WATER_VALUE = "water"
QUERY = (
    f"[out:json][timeout:45][bbox:{','.join(map(str, BOUNDS))}];"
    "(way[building][building!=no];"
    'way[highway~"^(primary|secondary|tertiary|unclassified|residential|living_street|pedestrian|service|footway|path|cycleway|steps)$"];'
    'way[natural=water][water~"^(canal|river)$"];'
    'way[waterway~"^(canal|river|riverbank)$"];'
    'relation[natural=water][water~"^(canal|river)$"];'
    "relation[type=multipolygon][building][building!=no];);out geom;"
)
ENDPOINT = "https://ovp.dresel.systems/api/interpreter"


def _fetch_osm():
    url = f"{ENDPOINT}?{urlencode({'data': QUERY})}"
    request = Request(
        url,
        headers={
            "User-Agent": "sdf_ui/0.1.0 Venice example (+https://github.com/sebastianjkern/sdf_ui)",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except (OSError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not retrieve OpenStreetMap data: {error}") from error


def _mercator(latitude, longitude):
    latitude = math.radians(latitude)
    return math.radians(longitude), math.log(math.tan(math.pi / 4 + latitude / 2))


def _project(latitude, longitude):
    south, west, north, east = BOUNDS
    left, bottom = _mercator(south, west)
    right, top = _mercator(north, east)
    x, y = _mercator(latitude, longitude)
    map_left, map_top, map_right, map_bottom = MAP
    return (
        map_left + (x - left) / (right - left) * (map_right - map_left),
        map_bottom - (y - bottom) / (top - bottom) * (map_bottom - map_top),
    )


def _clip_segment(first, second):
    x_min, y_min, x_max, y_max = MAP
    dx, dy = second[0] - first[0], second[1] - first[1]
    low, high = 0.0, 1.0
    for p, q in (
        (-dx, first[0] - x_min),
        (dx, x_max - first[0]),
        (-dy, first[1] - y_min),
        (dy, y_max - first[1]),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return None
    start = first if low == 0 else (first[0] + low * dx, first[1] + low * dy)
    end = second if high == 1 else (first[0] + high * dx, first[1] + high * dy)
    return start, end


def _clipped_paths(segments):
    paths = []
    current = []
    for first, second in segments:
        segment = _clip_segment(first, second)
        if segment is None:
            if len(current) > 1:
                paths.append(tuple(current))
            current = []
            continue
        start, end = segment
        if not current or current[-1] != start:
            if len(current) > 1:
                paths.append(tuple(current))
            current = [start]
        current.append(end)
    if len(current) > 1:
        paths.append(tuple(current))
    return paths


def _polygon_hatches(rings):
    return tuple(
        VectorPath(segment, BUILDING_HATCH, BUILDING_HATCH_WIDTH)
        for first, second in hatch_segments(rings, BUILDING_SPACING)
        if (segment := _clip_segment(first, second)) is not None
    )


def _feature_style(tags, widths):
    if "building" in tags:
        return BUILDINGS_LAYER, SECONDARY, 0.62
    if tags.get(WATERWAY_TAG) == CANAL:
        return WATER_LAYER, WATER_COLOR, 0.45
    if tags.get(WATERWAY_TAG) in WATERWAY_AREA or tags.get(WATER_TAG) in WATERWAY_AREA:
        width = 1.15 if tags.get(WATERWAY_TAG) == "river" else 0.85
        return WATER_LAYER, WATER_COLOR, width
    color = (
        INK
        if tags.get("highway") in {"primary", "secondary", "tertiary"}
        else SECONDARY
    )
    return "streets", color, widths.get(tags.get("highway"), 0.45)


def _is_water_area(tags):
    return tags.get(NATURAL_TAG) == WATER_VALUE and tags.get(WATER_TAG) in {
        "canal",
        "river",
    }


def _water_ring(element):
    geometry = element.get("geometry", ())
    if element.get("type") == "way" and len(geometry) > 2:
        return tuple((point["lat"], point["lon"]) for point in geometry)
    return ()


def _water_rings(data):
    rings = []
    for element in data.get("elements", ()):
        tags = element.get("tags", {})
        if _is_water_area(tags) and element.get("type") == "relation":
            rings.extend(
                ring
                for member in element.get("members", ())
                if (ring := _water_ring(member))
            )
            continue
        ring = (
            _water_ring(element)
            if _is_water_area(tags) or tags.get(WATERWAY_TAG) == "riverbank"
            else ()
        )
        if ring:
            rings.append(ring)
    return tuple(dict.fromkeys(rings))


def _water_paths(rings):
    edge_counts = Counter(
        frozenset((first, second)) for ring in rings for first, second in pairwise(ring)
    )
    paths = []
    for ring in rings:
        edges = (
            (_project(*first), _project(*second))
            for first, second in pairwise(ring)
            if edge_counts[frozenset((first, second))] == 1
        )
        paths.extend(
            VectorPath(clipped, WATER_COLOR, 0.85) for clipped in _clipped_paths(edges)
        )
    return paths


def _building_relation_rings(element):
    rings = []
    building_relations = {}
    for member in element.get("members", ()):
        if member.get("role") not in {"outer", "inner"}:
            continue
        geometry = member.get("geometry", ())
        if len(geometry) >= 2:
            rings.append(tuple((point["lat"], point["lon"]) for point in geometry))
            building_relations[member.get("ref")] = True
    return rings, building_relations


def _building_relation_paths(data):
    paths = []
    building_relations = {}
    for element in data.get("elements", ()):
        if element.get("type") != "relation" or "building" not in element.get(
            "tags", {}
        ):
            continue
        rings, members = _building_relation_rings(element)
        building_relations.update(members)
        projected_rings = tuple(
            tuple(_project(*point) for point in ring) for ring in rings
        )
        paths.extend(_polygon_hatches(projected_rings))
        for ring in projected_rings:
            for clipped in _clipped_paths(pairwise(ring)):
                paths.append(VectorPath(clipped, SECONDARY, 0.62))
    return paths, building_relations


def _way_feature_paths(element, widths, building_relations):
    if element.get("type") not in {None, "way"}:
        return None
    tags = element.get("tags", {})
    if element.get("id") in building_relations:
        return None
    if _is_water_area(tags) or tags.get(WATERWAY_TAG) == "riverbank":
        return None
    geometry = element.get("geometry", ())
    if len(geometry) < 2:
        return None
    points = [_project(point["lat"], point["lon"]) for point in geometry]
    layer, color, width = _feature_style(tags, widths)
    paths = []
    if layer == BUILDINGS_LAYER and len(geometry) > 2 and geometry[0] == geometry[-1]:
        paths.extend(_polygon_hatches((tuple(points),)))
    paths.extend(
        VectorPath(clipped, color, width)
        for clipped in _clipped_paths(pairwise(points))
    )
    return layer, paths


def _way_paths(data, widths, building_relations):
    layers = {WATER_LAYER: [], BUILDINGS_LAYER: [], STREETS_LAYER: []}
    for element in data.get("elements", ()):
        feature = _way_feature_paths(element, widths, building_relations)
        if feature:
            layer, paths = feature
            layers[layer].extend(paths)
    return layers


def _feature_paths(data):
    widths = {
        "primary": 1.55,
        "secondary": 1.2,
        "tertiary": 0.95,
        "unclassified": 0.65,
        "residential": 0.62,
        "living_street": 0.6,
        "pedestrian": 0.72,
        "service": 0.48,
        "footway": 0.38,
        "path": 0.38,
        "cycleway": 0.42,
        "steps": 0.38,
    }
    relation_paths, building_relations = _building_relation_paths(data)
    layers = _way_paths(data, widths, building_relations)
    layers[BUILDINGS_LAYER] = relation_paths + layers[BUILDINGS_LAYER]
    layers[WATER_LAYER].extend(_water_paths(_water_rings(data)))
    paths = layers[WATER_LAYER] + layers["buildings"] + layers["streets"]
    return VectorDrawing((WIDTH, HEIGHT), tuple(paths))


def _save_svg(drawing, path):
    write_svg(drawing, path)
    svg = path.read_text(encoding="utf-8")
    header = (
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAPER}"/>\n'
        f'<text x="{MAP[0]}" y="77" font-family="Roboto, sans-serif" font-size="38" '
        f'letter-spacing="3" fill="{INK}">VENICE</text>\n'
        f'<text x="{MAP[0] + 2}" y="111" font-family="Roboto, sans-serif" font-size="13" '
        f'letter-spacing="2" fill="{SECONDARY}">CANAL GRANDE · SAN MARCO</text>\n'
        f'<text x="{MAP[2]}" y="57" text-anchor="end" font-family="Roboto, sans-serif" '
        f'font-size="10" fill="{SECONDARY}"><a href="{OSM_COPYRIGHT}">'
        "Map data from OpenStreetMap</a></text>\n"
    )
    root, separator, body = svg.partition("\n")
    svg = f"{root}{separator}{header}{body}"
    path.write_text(svg, encoding="utf-8")


def _save_png(drawing, path):
    scale = 3
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), PAPER)
    draw = ImageDraw.Draw(image)
    for vector_path in drawing.paths:
        points = [(round(x * scale), round(y * scale)) for x, y in vector_path.points]
        draw.line(
            points,
            fill=vector_path.color,
            width=max(1, round(vector_path.width * scale)),
            joint="curve",
        )
    font_path = default_font_path()
    title = ImageFont.truetype(font_path, 38 * scale)
    label = ImageFont.truetype(font_path, 13 * scale)
    small = ImageFont.truetype(font_path, 10 * scale)
    lettering = ImageDraw.Draw(image)
    lettering.text(
        (MAP[0] * scale, 38 * scale), "VENICE", font=title, fill=INK, spacing=0
    )
    lettering.text(
        (MAP[0] * scale + 2 * scale, 94 * scale),
        "CANAL GRANDE · SAN MARCO",
        font=label,
        fill=SECONDARY,
    )
    lettering.text(
        (MAP[2] * scale, 47 * scale),
        "Map data from OpenStreetMap",
        font=small,
        fill=SECONDARY,
        anchor="ra",
    )
    image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(
        path, pnginfo=_png_metadata()
    )


def _png_metadata():
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Source", "https://www.openstreetmap.org/")
    metadata.add_text("License", OSM_COPYRIGHT)
    return metadata


def venice_line_art(output=OUTPUT, input_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = (
        json.loads(Path(input_path).read_text(encoding="utf-8"))
        if input_path
        else _fetch_osm()
    )
    if not data.get("elements"):
        raise ValueError("OpenStreetMap response contains no elements")
    drawing = _feature_paths(data)
    if not drawing.paths:
        raise ValueError(
            "OpenStreetMap response has no drawable ways in the Venice window"
        )
    _save_svg(drawing, output / "venice_line_art.svg")
    _save_png(drawing, output / "venice_line_art.png")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output directory")
    parser.add_argument("--input", type=Path, help="Use a local Overpass JSON response")
    args = parser.parse_args()
    print(venice_line_art(args.output, args.input))
