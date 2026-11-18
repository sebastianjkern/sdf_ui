"""Render palette and vector-field presets individually or as a contact sheet."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from sdf_ui import Canvas, color, field, sdf
from sdf_ui.core.palette import Palette

SIZE = (320, 220)
CENTER = (SIZE[0] / 2, SIZE[1] / 2)
OUTPUT_DIR = Path("out/examples/preset_gallery")


@dataclass(frozen=True)
class ExamplePreset:
    """Named example parameters used to produce a comparable render."""

    name: str
    palette: Palette
    vortex_strength: float
    turbulence: float
    ring_spacing: float


PRESETS = {
    "starry": ExamplePreset("starry", color.palette("starry_night"), 2.2, 4.0, 36.0),
    "ember": ExamplePreset("ember", color.palette("ember"), -2.6, 5.5, 32.0),
    "sea_glass": ExamplePreset("sea_glass", color.palette("sea_glass"), 1.7, 3.2, 40.0),
    "moss": ExamplePreset("moss", color.palette("moss"), -1.9, 4.5, 34.0),
}


def build_preset_scene(preset: ExamplePreset):
    """Build the same geometry with the selected palette and field settings."""
    cx, cy = CENTER
    flow = field.vortex(CENTER, radius=130, strength=preset.vortex_strength).add(
        field.turbulence(
            frequency=0.018,
            amplitude=preset.turbulence,
            octaves=3,
            seed=19,
        )
    )
    background_base = color.radial_gradient(
        CENTER,
        preset.palette.sample(0.56),
        preset.palette.sample(0.02),
        inner=12,
        outer=260,
    )
    background = background_base.warp(flow, strength=8.0, steps=4)

    rings = color.clear((0.0, 0.0, 0.0, 0.0))
    for index in range(4):
        radius = 34 + index * preset.ring_spacing
        ring = sdf.ring(CENTER, radius, 2.0 + index * 0.35).warp(
            flow, strength=12.0, steps=4
        )
        rings = rings.alpha_overlay(
            ring.fill(
                preset.palette.sample(0.68 + index * 0.08),
                (0.0, 0.0, 0.0, 0.0),
            ).transparency(0.82)
        )

    rosette = (
        sdf.circle(CENTER, 15)
        .repeat_grid(
            columns=7,
            rows=3,
            spacing=(preset.ring_spacing, preset.ring_spacing),
            center=CENTER,
        )
        .warp(flow, strength=10.0, steps=4)
    )
    marks = rosette.fill(preset.palette.sample(0.9), (0.0, 0.0, 0.0, 0.0)).transparency(
        0.88
    )
    core = sdf.circle(CENTER, 17).fill(preset.palette.sample(1.0), (0.0, 0.0, 0.0, 0.0))
    halo = color.radial_gradient(
        (cx, cy),
        (*preset.palette.sample(0.94)[:3], 0.58),
        (*preset.palette.sample(0.94)[:3], 0.0),
        inner=4,
        outer=55,
    )
    return (
        background_base.alpha_overlay(background)
        .alpha_overlay(rings)
        .alpha_overlay(marks)
        .alpha_overlay(halo)
        .alpha_overlay(core)
    )


def render_preset_gallery(preset_names=None, output_dir=OUTPUT_DIR):
    """Save individual preset renders and a labeled comparison when applicable."""
    names = tuple(PRESETS) if preset_names is None else tuple(preset_names)
    if not names:
        raise ValueError("select at least one preset")
    unknown = set(names) - PRESETS.keys()
    if unknown:
        raise ValueError(f"unknown presets: {', '.join(sorted(unknown))}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    with Canvas(SIZE) as canvas:
        for name in names:
            path = output_dir / f"{name}.png"
            build_preset_scene(PRESETS[name]).save(path, canvas=canvas)
            image_paths.append(path)

    comparison_path = None
    if len(image_paths) > 1:
        columns = 2
        rows = (len(image_paths) + columns - 1) // columns
        panel_height = SIZE[1] + 28
        sheet = Image.new(
            "RGB",
            (columns * SIZE[0], rows * panel_height),
            (9, 16, 34),
        )
        draw = ImageDraw.Draw(sheet)
        for index, (name, path) in enumerate(zip(names, image_paths)):
            x = (index % columns) * SIZE[0]
            y = (index // columns) * panel_height
            draw.text((x + 10, y + 7), name.replace("_", " ").title(), fill="#e7edf8")
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB"), (x, y + 28))
        comparison_path = output_dir / "comparison.png"
        sheet.save(comparison_path)

    return tuple(image_paths), comparison_path


def preset_gallery_example():
    """Render all presets for the examples command registry."""
    return render_preset_gallery()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", action="append", choices=tuple(PRESETS))
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    images, comparison = render_preset_gallery(args.preset, args.output)
    for image in images:
        print(image)
    if comparison is not None:
        print(comparison)


if __name__ == "__main__":
    main()
