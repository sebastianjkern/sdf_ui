"""Render a seamless, slowly flowing liquid-glass forest animation."""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

from examples.geometric_forest_wallpaper import geometric_forest_scene
from sdf_ui import Canvas
from sdf_ui.learning import Parameter

SIZE = (1280, 720)
DEFAULT_OUTPUT = Path("out/wallpapers/liquid_glass_geometric_forest.gif")


def render_animation(
    output=DEFAULT_OUTPUT, size=SIZE, frames=36, fps=18
):
    """Render a looping animation and a representative still preview."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    phase = Parameter(
        "phase", 0.0, animate=lambda progress: progress * 2.0 * math.pi
    )
    preview = output.with_name(output.stem + "_preview.png")

    with Canvas(size) as canvas:
        scene = geometric_forest_scene(size, animated=True)
        scene.save(preview, canvas=canvas, params={"phase": phase}, progress=0.25)
        canvas.save_animation(
            scene,
            output,
            frames=frames,
            params={"phase": phase},
            fps=fps,
        )

    with Image.open(preview) as image:
        ImageOps.flip(image).save(preview)
    with Image.open(output) as image:
        duration = image.info.get("duration", round(1000 / fps))
        loop = image.info.get("loop", 0)
        frames_out = [
            ImageOps.flip(frame.convert("RGB"))
            for frame in ImageSequence.Iterator(image)
        ]
    frames_out[0].save(
        output,
        save_all=True,
        append_images=frames_out[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    return output, preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=SIZE[0])
    parser.add_argument("--height", type=int, default=SIZE[1])
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=18)
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.frames, args.fps) < 1:
        parser.error("width, height, frames, and fps must be positive")
    animation, preview = render_animation(
        args.output,
        (args.width, args.height),
        frames=args.frames,
        fps=args.fps,
    )
    print(animation)
    print(preview)


if __name__ == "__main__":
    main()
