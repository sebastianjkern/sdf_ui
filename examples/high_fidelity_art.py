"""Render Starry Current with HDR finishing and export a tiled art print."""

import argparse
from pathlib import Path

from PIL import Image

from examples.starry_current_animation import SIZE as STARRY_SIZE
from examples.starry_current_animation import starry_current_scene
from sdf_ui import Canvas, field, param, sdf

OUTPUT = Path("out/generative_art/high_fidelity")
STILL_SIZE = (960, 640)


def ember_current():
    """A tile-compatible study with procedural light and warped geometry."""
    center = (480, 320)
    scale = param("flow_scale", 1.0)
    current = field.vortex((545, 325), 260, 32 * scale).add(
        field.turbulence(frequency=0.016 / scale, amplitude=5 * scale, seed=27)
    )
    background = sdf.circle((475, 326), 0).fill_linear(
        (0.075, 0.035, 0.12, 1),
        (0.006, 0.014, 0.036, 1),
        inner=24,
        outer=525,
    )
    rings = [
        sdf.circle((center[0] + offset, center[1] - offset * 0.18), radius).subtract(
            sdf.circle((center[0] + offset, center[1] - offset * 0.18), radius - 3)
        )
        for offset, radius in ((-44, 145), (-29, 166), (-9, 189), (18, 213), (34, 234))
    ]
    threads = sdf.union_all(rings).warp(current, strength=1.4, steps=12)
    ink = threads.ink_linear((0.85, 0.19, 0.075, 0.84))
    orbit = sdf.circle((474, 345), 104).subtract(sdf.circle((474, 345), 101))
    rim = orbit.warp(current, strength=1.5, steps=12).ink_linear(
        (1.7, 0.68, 0.22, 0.92)
    )
    sparks = sdf.union_all(
        sdf.circle(point, radius)
        for point, radius in (
            ((620, 437), 9),
            ((343, 464), 5),
            ((293, 222), 4),
            ((564, 164), 6),
        )
    ).ink_linear((3.8, 1.25, 0.27, 1))
    return (
        background.over(ink, rim, sparks).bloom(amount=0.65, passes=3).finish(tone=0.85)
    )


def render(output, scale, tile_size, *, skip_reference=False):
    output.mkdir(parents=True, exist_ok=True)
    if not skip_reference:
        with Canvas(STARRY_SIZE) as canvas:
            scenes = {
                "starry_current_original": starry_current_scene(),
                "starry_current_hdr": (
                    starry_current_scene(linear=True)
                    .bloom(threshold=0.9, amount=0.7, passes=3)
                    .finish(exposure=0.25, tone=0.65)
                ),
            }
            for name, scene in scenes.items():
                # Starry Current was authored in image row order, as in
                # its animated preview. Save the rendered rows in that order.
                with canvas.render(scene.to_rgb(), params={"phase": 0.9}) as frame:
                    with Image.fromarray(frame.tex.read()) as image:
                        image.save(output / f"{name}.png")
    with Canvas(STILL_SIZE, device_scale=scale) as canvas:
        return canvas.save_tiled(
            ember_current(),
            output / f"ember_current_{canvas.size[0]}.png",
            tile_size=(tile_size, tile_size),
            params={"flow_scale": scale},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--scale", type=int, default=1, help="Physical pixels per canvas unit"
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()
    if args.scale < 1 or args.tile_size < 1:
        parser.error("scale and tile-size must be positive")
    print(
        render(
            args.output, args.scale, args.tile_size, skip_reference=args.skip_reference
        )
    )


if __name__ == "__main__":
    main()
