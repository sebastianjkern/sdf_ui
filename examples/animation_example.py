"""Render a parameter animation as a sequence of frames and an MP4."""

import math
from pathlib import Path

from sdf_ui import Canvas, color, param, sdf
from sdf_ui.learning import Parameter


def animation_example(frames=180, output=Path("out/anim")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Canvas((640, 360)) as canvas:
        background = color.clear((1.0, 1.0, 1.0, 1.0)).cache("background")
        disc1 = sdf.circle(
            (canvas.percent_x(10), canvas.percent_y(50)), canvas.percent_of_min(35)
        ).cache("disc1")
        center_x = Parameter(
            "center_x",
            0.0,
            animate=lambda progress: (
                640 * (7 / 20 + 1 / 4 * math.sin(progress * 2 * math.pi))
            ),
        )
        disc2 = sdf.circle(
            (param("center_x"), canvas.percent_y(50)), canvas.percent_of_min(15)
        )
        image = background.alpha_overlay(
            disc1.smooth_union(disc2).fill("#62bb47", (0.0, 0.0, 0.0, 0.0))
        )

        canvas.save_animation(
            image,
            output / "animation.mp4",
            frames=frames,
            params={"center_x": center_x},
            fps=30,
        )


if __name__ == "__main__":
    animation_example()
