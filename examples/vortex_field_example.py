"""Distort an SDF with two vortices and visualize their streamlines."""

from pathlib import Path

from sdf_ui import Canvas, color, sdf

OUTPUT_DIR = Path("out/examples")


def _vortex_art(vortices):
    source = (
        sdf.circle((480, 360), 250).subtract(sdf.circle((480, 360), 182))
        .union(sdf.circle((480, 360), 86))
    )
    warped = source.vortex_warp(vortices, radius=175, strength=0.5)
    shape = warped.fill((0.82, 0.86, 0.83, 1.0), (0.0, 0.0, 0.0, 0.0))
    contours = warped.isolines(
        (0.23, 0.36, 0.43, 0.8), (0.0, 0.0, 0.0, 0.0),
        spacing=18, line_width=0.8, feather=0.4,
    )
    streams = color.vortex_streamlines(
        vortices, radius=175, spacing=118, step_size=2.0, steps=170
    )
    return shape, contours, streams


def _vortex_markers(vortices):
    markers = color.clear((0.0, 0.0, 0.0, 0.0))
    for x, y, strength in vortices:
        tint = (0.35, 0.95, 0.87, 1.0) if strength > 0 else (1.0, 0.64, 0.34, 1.0)
        marker = sdf.circle((x, y), 5).fill(tint, (0.0, 0.0, 0.0, 0.0))
        markers = markers.alpha_overlay(marker)
    return markers


def vortex_field_example():
    """Distort the shape and render streamlines around two opposing vortices."""
    vortices = ((300, 360, 1.15), (660, 360, -23 / 20))
    with Canvas((960, 720)) as canvas:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shape, contours, streams = _vortex_art(vortices)
        scene = color.clear((0.025, 0.035, 0.065, 1.0)).alpha_overlay(shape)
        scene = scene.alpha_overlay(contours).alpha_overlay(streams)
        scene.alpha_overlay(_vortex_markers(vortices)).save(
            OUTPUT_DIR / "vortex_distortion.png", canvas=canvas
        )


if __name__ == "__main__":
    vortex_field_example()
