"""Animate vortex and turbulence fields that warp an SDF pattern."""

import math
from pathlib import Path

from sdf_ui import Canvas, color, cos, field, param, sdf, sin
from sdf_ui.learning import Parameter

SIZE = (480, 320)
CENTER = (SIZE[0] / 2, SIZE[1] / 2)
OUTPUT_DIR = Path("out/generative_art")


def field_flow_scene():
    """Build a loop driven by progress expressions on vector-field uniforms."""
    phase = param("phase")
    center_x, center_y = CENTER
    orbit_x = 102 * cos(phase)
    orbit_y = 54 * sin(phase)
    strength = 0.9 + 0.2 * sin(phase * 2)

    left_vortex = field.vortex(
        (center_x - orbit_x, center_y - orbit_y),
        radius=144,
        strength=strength,
    )
    right_vortex = field.vortex(
        (center_x + orbit_x, center_y + orbit_y),
        radius=144,
        strength=-strength * 0.88,
    )
    turbulence = field.turbulence(
        frequency=0.012 + 0.003 * (1 + cos(phase)),
        amplitude=5 + 3 * (1 + sin(phase * 2)),
        octaves=3,
        seed=23,
    )
    flow = left_vortex.add(right_vortex).add(turbulence)

    pattern = sdf.circle(CENTER, 13).repeat_grid(
        columns=9, rows=5, spacing=(46, 46), center=CENTER
    )
    warped_pattern = pattern.warp(flow, strength=1.5, steps=2)
    pattern_fill = warped_pattern.fill("#37bdd5", (0.0, 0.0, 0.0, 0.0))
    pattern_lines = warped_pattern.isolines(
        "#c2f8f3",
        (0.0, 0.0, 0.0, 0.0),
        spacing=8,
        line_width=0.8,
        feather=0.4,
    )

    ring_radius = 112 + 7 * sin(phase)
    orbit_ring = sdf.ring(CENTER, ring_radius, 2.2).warp(
        flow, strength=0.8, steps=2
    )
    ring_layer = orbit_ring.fill("#f4bd72", (0.0, 0.0, 0.0, 0.0))

    markers = color.clear((0.0, 0.0, 0.0, 0.0))
    for x, y, tint in (
        (center_x - orbit_x, center_y - orbit_y, "#ffd18a"),
        (center_x + orbit_x, center_y + orbit_y, "#79f1dd"),
    ):
        marker = sdf.circle((x, y), 5.5).fill(tint, (0.0, 0.0, 0.0, 0.0))
        markers = markers.alpha_overlay(marker)

    backdrop = color.radial_gradient(
        CENTER,
        (0.11, 0.19, 0.31, 1.0),
        (0.012, 0.022, 0.052, 1.0),
        inner=12,
        outer=350,
    ).warp(flow, strength=0.65, steps=2)
    flow_map = flow.contours(spacing=8, line_width=0.9).transparency(0.16)
    return (
        backdrop.alpha_overlay(flow_map)
        .alpha_overlay(pattern_fill)
        .alpha_overlay(pattern_lines)
        .alpha_overlay(ring_layer)
        .alpha_overlay(markers)
    )


def field_flow_animation_example(
    frames=48, output=OUTPUT_DIR / "field_flow_animation"
):
    """Write a looping GIF and a representative PNG preview."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    phase = Parameter(
        "phase",
        0.0,
        animate=lambda progress: progress * 2.0 * math.pi,
    )
    with Canvas(SIZE) as canvas:
        scene = field_flow_scene()
        scene.save(
            output.with_suffix(".png"),
            ctx=canvas,
            params={"phase": phase},
            progress=0.28,
        )
        canvas.save_animation(
            scene,
            output.with_suffix(".gif"),
            frames=frames,
            params={"phase": phase},
            fps=24,
        )


if __name__ == "__main__":
    field_flow_animation_example()
