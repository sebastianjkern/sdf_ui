"""Compare raw smooth-union values with two distance corrections."""

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from sdf_ui import Canvas, sdf

OUTPUT = Path("out/examples/distance_corrections.png")
PANEL_SIZE = (400, 460)


def distance_corrections_example():
    methods = (
        ("Smooth union", lambda field: field),
        ("Local correction", lambda field: field.normalize_distance(band=40)),
        ("Redistanced", lambda field: field.redistance()),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary_directory:
        panels = []
        for index, (label, correct) in enumerate(methods):
            panel_path = Path(temporary_directory) / f"panel_{index}.png"
            with Canvas(PANEL_SIZE) as canvas:
                field = sdf.circle((140, 255), 98).smooth_union(
                    sdf.circle((260, 255), 98), k=70
                )
                field = correct(field)
                diagnostic = field.debug_distance(
                    range=52, show_contours=True, spacing=12
                )
                caption = sdf.text(
                    label,
                    size=24,
                    box=(20, 396, 360, 44),
                    align="center",
                    valign="middle",
                ).fill((0.94, 0.96, 0.98, 1.0), (0.0, 0.0, 0.0, 0.0))
                diagnostic.alpha_overlay(caption).save(panel_path, canvas=canvas)
            panels.append(Image.open(panel_path).convert("RGB"))

        gap = 14
        width = PANEL_SIZE[0] * len(panels) + gap * (len(panels) - 1)
        montage = Image.new("RGB", (width, PANEL_SIZE[1]), (8, 12, 21))
        for index, panel in enumerate(panels):
            montage.paste(panel, (index * (PANEL_SIZE[0] + gap), 0))
        draw = ImageDraw.Draw(montage)
        draw.line(
            (PANEL_SIZE[0] + gap // 2, 0, PANEL_SIZE[0] + gap // 2, PANEL_SIZE[1]),
            fill=(35, 48, 65),
        )
        draw.line(
            (
                2 * PANEL_SIZE[0] + 3 * gap // 2,
                0,
                2 * PANEL_SIZE[0] + 3 * gap // 2,
                PANEL_SIZE[1],
            ),
            fill=(35, 48, 65),
        )
        montage.save(OUTPUT)


if __name__ == "__main__":
    distance_corrections_example()
