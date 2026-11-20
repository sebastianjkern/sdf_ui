"""Compare raw smooth-union values with two distance corrections."""

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from sdf_ui import Canvas, sdf

OUTPUT = Path("out/examples/distance_corrections/distance_corrections.png")
PANEL_SIZE = (400, 460)


def _render_panel(index, label, correct, temporary_directory):
    panel_path = Path(temporary_directory) / f"panel_{index}.png"
    with Canvas(PANEL_SIZE) as canvas:
        field = sdf.circle((140, 255), 98).smooth_union(
            sdf.circle((260, 255), 98), k=70
        )
        diagnostic = correct(field).debug_distance(
            range=52, show_contours=True, spacing=12
        )
        caption = sdf.text(
            label, size=24, box=(20, 396, 360, 44), align="center", valign="middle"
        ).fill((0.94, 0.96, 0.98, 1.0), (0.0, 0.0, 0.0, 0.0))
        diagnostic.alpha_overlay(caption).save(panel_path, canvas=canvas)
    return Image.open(panel_path).convert("RGB")


def _compose_montage(panels):
    gap = 14
    width = PANEL_SIZE[0] * len(panels) + gap * (len(panels) - 1)
    montage = Image.new("RGB", (width, PANEL_SIZE[1]), (8, 12, 21))
    for index, panel in enumerate(panels):
        montage.paste(panel, (index * (PANEL_SIZE[0] + gap), 0))
    draw = ImageDraw.Draw(montage)
    for index in (1, 2):
        x = index * PANEL_SIZE[0] + (2 * index - 1) * gap // 2
        draw.line((x, 0, x, PANEL_SIZE[1]), fill=(35, 48, 65))
    montage.save(OUTPUT)


def distance_corrections_example():
    methods = (
        ("Smooth union", lambda field: field),
        ("Local correction", lambda field: field.normalize_distance(band=40)),
        ("Redistanced", lambda field: field.redistance()),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary_directory:
        panels = [
            _render_panel(index, label, correct, temporary_directory)
            for index, (label, correct) in enumerate(methods)
        ]
        _compose_montage(panels)


if __name__ == "__main__":
    distance_corrections_example()
