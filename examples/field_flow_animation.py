"""Animate vortex and turbulence fields that warp an SDF pattern."""

import math
from pathlib import Path

from sdf_ui import Canvas, color, cos, field, param, sdf, sin
from sdf_ui.learning import Parameter

SIZE = (480, 320)
CENTER = (SIZE[0] / 2, SIZE[1] / 2)
OUTPUT_DIR = Path("out/examples/field_flow")


def _flow_for_phase(phase):
    center_x, center_y = CENTER
    orbit_x, orbit_y = 102 * cos(phase), 54 * sin(phase)
    strength = 9 / 10 + 1 / 5 * sin(phase * 2)
    left = field.vortex(
        (center_x - orbit_x, center_y - orbit_y), radius=144, strength=strength
    )
    right = field.vortex(
        (center_x + orbit_x, center_y + orbit_y),
        radius=144, strength=-strength * 22 / 25,
    )
    noise = field.turbulence(
        frequency=3 / 250 + 3 / 1_000 * (1 + cos(phase)),
        amplitude=5 + 3 * (1 + sin(phase * 2)), octaves=3, seed=23,
    )
    return left.add(right).add(noise), orbit_x, orbit_y


def _pattern_layers(flow):
    pattern = sdf.circle(CENTER, 13).repeat_grid(
        columns=9, rows=5, spacing=(46, 46), center=CENTER
    )
    warped = pattern.warp(flow, strength=1.5, steps=2)
    fill = warped.fill("#37bdd5", (0.0, 0.0, 0.0, 0.0))
    lines = warped.isolines(
        "#c2f8f3", (0.0, 0.0, 0.0, 0.0), spacing=8, line_width=0.8, feather=0.4
    )
    return fill, lines


def _orbit_layers(flow, phase, orbit_x, orbit_y):
    ring = sdf.ring(CENTER, 112 + 7 * sin(phase), 2.2).warp(
        flow, strength=0.8, steps=2
    ).fill("#f4bd72", (0.0, 0.0, 0.0, 0.0))
    cx, cy = CENTER
    markers = color.clear((0.0, 0.0, 0.0, 0.0))
    for x, y, tint in (
        (cx - orbit_x, cy - orbit_y, "#ffd18a"),
        (cx + orbit_x, cy + orbit_y, "#79f1dd"),
    ):
        markers = markers.alpha_overlay(
            sdf.circle((x, y), 5.5).fill(tint, (0.0, 0.0, 0.0, 0.0))
        )
    return ring, markers


def _flow_backdrop(flow):
    backdrop = color.radial_gradient(
        CENTER, (0.11, 0.19, 0.31, 1.0), (0.012, 0.022, 0.052, 1.0),
        inner=12, outer=350,
    ).warp(flow, strength=0.65, steps=2)
    return backdrop.alpha_overlay(
        flow.contours(spacing=8, line_width=0.9).transparency(0.16)
    )


def field_flow_scene():
    """Build a loop driven by progress expressions on vector-field uniforms."""
    phase = param("phase")
    flow, orbit_x, orbit_y = _flow_for_phase(phase)
    pattern_fill, pattern_lines = _pattern_layers(flow)
    ring, markers = _orbit_layers(flow, phase, orbit_x, orbit_y)
    return _flow_backdrop(flow).alpha_overlay(pattern_fill).alpha_overlay(
        pattern_lines
    ).alpha_overlay(ring).alpha_overlay(markers)


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
            canvas=canvas,
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
