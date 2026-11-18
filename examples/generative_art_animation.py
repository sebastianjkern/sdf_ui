"""Render a looping, SDF-native generative animation: the Tidal Bloom."""

import math
from pathlib import Path

from sdf_ui import Canvas, color, cos, param, sdf, sin
from sdf_ui.learning import Parameter

SIZE = 640
CENTER = (SIZE / 2, SIZE / 2)


def tidal_bloom_scene(canvas):
    """Build a layered orbital flower animated by one normalized phase."""
    phase = param("phase")
    cx, cy = CENTER

    petals = []
    petal_count = 12
    for index in range(petal_count):
        angle = 2.0 * math.pi * index / petal_count
        orbit = 108 + 4 * sin(phase * 2.0 + angle * 3.0)
        center = (
            cx + orbit * math.cos(angle),
            cy + orbit * math.sin(angle),
        )
        length = 66 + 5 * sin(phase * 2.0 + angle * 4.0)
        petal = sdf.ellipse(center, (27, length)).rotate(
            angle - math.pi / 2, center=center
        )
        petals.append(petal)

    bloom = petals[0]
    for petal in petals[1:]:
        bloom = bloom.union(petal)
    bloom = bloom.rotate(phase, center=CENTER)

    background = color.radial_gradient(
        CENTER,
        (0.15, 0.24, 0.40, 1.0),
        (0.012, 0.018, 0.045, 1.0),
        outer=SIZE * 0.76,
    )
    grain = color.film_grain().transparency(0.035)

    # A pair of thin, breathing orbital tracks frames the moving bloom.
    orbital_tracks = color.clear((0.0, 0.0, 0.0, 0.0))
    for radius, width, tint in (
        (220 + 5 * sin(phase * 2.0), 1.6, (0.22, 0.82, 0.86, 0.62)),
        (241 - 4 * sin(phase * 2.0), 1.0, (0.92, 0.63, 0.32, 0.52)),
        (188, 0.8, (0.55, 0.52, 0.94, 0.40)),
    ):
        track = sdf.ring(CENTER, radius, width).fill(
            tint, (0.0, 0.0, 0.0, 0.0), inner=-0.7, outer=0.7
        )
        orbital_tracks = orbital_tracks.alpha_overlay(track)

    body = bloom.fill(
        (0.17, 0.50, 0.69, 1.0),
        (0.0, 0.0, 0.0, 0.0),
        inner=-2.0,
        outer=1.0,
    )
    contour_threads = bloom.isolines(
        (0.56, 0.94, 0.88, 0.82),
        (0.0, 0.0, 0.0, 0.0),
        spacing=17,
        line_width=0.9,
        feather=0.55,
        phase=phase * (17.0 * 4.0 / (2.0 * math.pi)),
    )
    contour_threads = body.alpha_overlay(
        contour_threads.layer_mask(body, bloom.mask())
    )

    # Two constellations orbit in opposite directions, built as single SDF
    # groups so each constellation costs one fill pass rather than one per dot.
    constellations = []
    for count, radius, dot_radius, direction in (
        (9, 219, 3.4, 1.0),
        (7, 164, 2.2, -1.0),
    ):
        dots = []
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            moving_angle = angle + direction * phase
            point = (
                cx + radius * cos(moving_angle),
                cy + radius * sin(moving_angle),
            )
            dots.append(sdf.circle(point, dot_radius))
        constellation = dots[0]
        for dot in dots[1:]:
            constellation = constellation.union(dot)
        constellations.append(constellation)

    gold_stars = constellations[0].fill(
        (1.0, 0.70, 0.37, 1.0), (0.0, 0.0, 0.0, 0.0)
    )
    ice_stars = constellations[1].fill(
        (0.55, 0.88, 1.0, 0.95), (0.0, 0.0, 0.0, 0.0)
    )

    outer_halo = sdf.ring(CENTER, 72, 2.5).fill(
        (0.96, 0.70, 0.39, 0.88), (0.0, 0.0, 0.0, 0.0)
    )
    inner_halo = sdf.ring(CENTER, 52 + 3 * sin(phase * 2.0), 1.2).fill(
        (0.42, 0.91, 0.90, 0.84), (0.0, 0.0, 0.0, 0.0)
    )
    core = color.radial_gradient(
        CENTER,
        (0.95, 0.74, 0.43, 0.98),
        (0.14, 0.42, 0.59, 0.0),
        inner=4,
        outer=66,
    )

    scene = background.alpha_overlay(grain)
    for layer in (
        orbital_tracks,
        body,
        contour_threads,
        gold_stars,
        ice_stars,
        outer_halo,
        inner_halo,
        core,
    ):
        scene = scene.alpha_overlay(layer)
    return scene


def generative_art_animation(
    frames=96, output=Path("out/generative_art/tidal_bloom")
):
    """Render an MP4 loop and a still preview of the Tidal Bloom artwork."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    phase = Parameter(
        "phase",
        0.0,
        animate=lambda progress: progress * 2.0 * math.pi,
    )
    with Canvas((SIZE, SIZE)) as canvas:
        scene = tidal_bloom_scene(canvas)
        scene.save(output.with_suffix(".png"), canvas=canvas, params={"phase": phase})
        canvas.save_animation(
            scene,
            output.with_suffix(".mp4"),
            frames=frames,
            params={"phase": phase},
            fps=30,
        )


if __name__ == "__main__":
    generative_art_animation()
