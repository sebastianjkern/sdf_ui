"""Render a looping, SDF-native generative animation: the Tidal Bloom."""

import math
from pathlib import Path

from sdf_ui import Canvas, color, cos, param, sdf, sin
from sdf_ui.learning import Parameter

SIZE = 640
CENTER = (SIZE / 2, SIZE / 2)


def _bloom_shape(phase):
    cx, cy = CENTER
    petals = []
    for index in range(12):
        angle = 2.0 * math.pi * index / 12
        orbit = 108 + 4 * sin(phase * 2.0 + angle * 3.0)
        center = (cx + orbit * math.cos(angle), cy + orbit * math.sin(angle))
        length = 66 + 5 * sin(phase * 2.0 + angle * 4.0)
        petals.append(sdf.ellipse(center, (27, length)).rotate(angle - math.pi / 2, center=center))
    bloom = petals[0]
    for petal in petals[1:]:
        bloom = bloom.union(petal)
    return bloom.rotate(phase, center=CENTER)


def _orbital_tracks(phase):
    tracks = color.clear((0.0, 0.0, 0.0, 0.0))
    settings = (
        (220 + 5 * sin(phase * 2.0), 1.6, (0.22, 0.82, 0.86, 0.62)),
        (241 - 4 * sin(phase * 2.0), 1.0, (0.92, 0.63, 0.32, 0.52)),
        (188, 0.8, (0.55, 0.52, 0.94, 0.40)),
    )
    for radius, width, tint in settings:
        track = sdf.ring(CENTER, radius, width).fill(
            tint, (0.0, 0.0, 0.0, 0.0), inner=-7 / 10, outer=0.7
        )
        tracks = tracks.alpha_overlay(track)
    return tracks


def _constellation_layers(phase):
    groups = []
    for count, radius, dot_radius, direction in ((9, 219, 3.4, 1.0), (7, 164, 2.2, -1.0)):
        dots = []
        for index in range(count):
            angle = 2.0 * math.pi * index / count + direction * phase
            point = (CENTER[0] + radius * cos(angle), CENTER[1] + radius * sin(angle))
            dots.append(sdf.circle(point, dot_radius))
        group = dots[0]
        for dot in dots[1:]:
            group = group.union(dot)
        groups.append(group)
    return (
        groups[0].fill((1.0, 0.70, 0.37, 1.0), (0.0, 0.0, 0.0, 0.0)),
        groups[1].fill((0.55, 0.88, 1.0, 0.95), (0.0, 0.0, 0.0, 0.0)),
    )


def _bloom_core(phase):
    body = _bloom_shape(phase)
    fill = body.fill((0.17, 0.50, 0.69, 1.0), (0.0, 0.0, 0.0, 0.0), inner=-2.0, outer=1.0)
    threads = body.isolines(
        (0.56, 0.94, 0.88, 0.82), (0.0, 0.0, 0.0, 0.0),
        spacing=17, line_width=0.9, feather=0.55,
        phase=phase * (17.0 * 4.0 / (2.0 * math.pi)),
    )
    return fill, fill.alpha_overlay(threads.layer_mask(fill, body.mask()))


def tidal_bloom_scene(canvas=None):
    """Build a layered orbital flower animated by one normalized phase."""
    phase = param("phase")
    fill, threads = _bloom_core(phase)
    stars = _constellation_layers(phase)
    halos = (
        sdf.ring(CENTER, 72, 2.5).fill((0.96, 0.70, 0.39, 0.88), (0.0, 0.0, 0.0, 0.0)),
        sdf.ring(CENTER, 52 + 3 * sin(phase * 2.0), 1.2).fill((0.42, 0.91, 0.90, 0.84), (0.0, 0.0, 0.0, 0.0)),
        color.radial_gradient(CENTER, (0.95, 0.74, 0.43, 0.98), (0.14, 0.42, 0.59, 0.0), inner=4, outer=66),
    )
    scene = color.radial_gradient(
        CENTER, (0.15, 0.24, 0.40, 1.0), (0.012, 0.018, 0.045, 1.0),
        outer=SIZE * 19 / 25,
    ).film_grain().to_rgb()
    for layer in (_orbital_tracks(phase), fill, threads, *stars, *halos):
        scene = scene.alpha_overlay(layer)
    return scene


def generative_art_animation(
    frames=96, output=Path("out/examples/tidal_bloom/tidal_bloom")
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
