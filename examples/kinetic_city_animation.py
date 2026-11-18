"""Render a looping neon cityscape animation with a gliding night tram."""

import math
from pathlib import Path

from sdf_ui import Canvas, color, param, sdf, sin
from sdf_ui.learning import Parameter

WIDTH = 960
HEIGHT = 540
PHASE = param("phase")


def _union(shapes):
    shapes = iter(shapes)
    result = next(shapes)
    for shape in shapes:
        result = result.union(shape)
    return result


def _filled_union(shapes, tint):
    return _union(shapes).fill(tint, (0.0, 0.0, 0.0, 0.0))


def _rect(center, size, *, corner_radius=0.0, angle=0.0):
    """Create an SDF box from the intuitive full width and height."""
    return sdf.rect(
        center,
        (size[0] / 2, size[1] / 2),
        corner_radius=corner_radius,
        angle=angle,
    )


def _rounded_rect(center, size, corner_radius):
    return sdf.rounded_rect(
        center, (size[0] / 2, size[1] / 2), corner_radius
    )


def kinetic_city_scene():
    """Build a layered, animated skyline with a moving train and reflections."""
    train_x = -120 + (WIDTH + 240) * PHASE / (2.0 * math.pi)

    sky = color.clear((0.012, 0.02, 0.045, 1.0))
    atmospheric_glow = color.radial_gradient(
        (720, HEIGHT - 156),
        (0.80, 0.19, 0.12, 0.30),
        (0.02, 0.04, 0.10, 0.0),
        inner=12,
        outer=480,
    )

    sun = sdf.circle((720, HEIGHT - 164), 86).fill(
        (1.0, 0.34, 0.19, 1.0), (0.0, 0.0, 0.0, 0.0)
    )
    sun_halo = sdf.ring((720, HEIGHT - 164), 97, 1.2).fill(
        (1.0, 0.63, 0.34, 0.64), (0.0, 0.0, 0.0, 0.0)
    )

    distant_buildings = []
    for index in range(15):
        x = -20 + index * 70
        width = 52 + (index * 17 % 22)
        height = 100 + (index * 53 % 118)
        distant_buildings.append(
            _rect((x, 110 + height / 2), (width, height), corner_radius=2)
        )
        if index % 3 == 1:
            distant_buildings.append(
                _rect((x, 110 + height + 9), (5, 18), corner_radius=1)
            )
    distant_skyline = _filled_union(
        distant_buildings, (0.055, 0.105, 0.19, 1.0)
    )

    middle_buildings = []
    for index in range(11):
        x = 16 + index * 92
        width = 70 + (index * 13 % 20)
        height = 135 + (index * 47 % 150)
        middle_buildings.append(
            _rect((x, 110 + height / 2), (width, height), corner_radius=2)
        )
        if index % 2 == 0:
            middle_buildings.append(
                _rect((x + 16, 110 + height + 12), (3, 24), corner_radius=1)
            )
    middle_skyline = _filled_union(
        middle_buildings, (0.045, 0.14, 0.22, 1.0)
    )

    foreground_buildings = []
    window_shapes = []
    cool_window_shapes = []
    building_specs = (
        (54, 72, 188),
        (158, 78, 272),
        (268, 82, 210),
        (370, 72, 318),
        (468, 84, 238),
        (578, 78, 290),
        (680, 88, 202),
        (792, 80, 330),
        (900, 76, 250),
    )
    for building_index, (x, width, height) in enumerate(building_specs):
        top = 110 + height
        foreground_buildings.append(
            _rect((x, 110 + height / 2), (width, height), corner_radius=1.5)
        )
        if building_index in (1, 3, 5, 7):
            crown_width = width * 0.55
            foreground_buildings.append(
                _rect(
                    (x, top + 9),
                    (crown_width, 18),
                    corner_radius=1,
                )
            )
        for row, y in enumerate(range(int(top - 28), 120, -31)):
            for col, dx in enumerate((-width * 0.25, 0, width * 0.25)):
                if (building_index * 7 + row * 3 + col * 5) % 5 == 0:
                    continue
                window = _rect(
                    (x + dx, y),
                    (7, 12 if row % 3 else 18),
                    corner_radius=1,
                )
                if (building_index + row + col) % 4 == 0:
                    cool_window_shapes.append(window)
                else:
                    window_shapes.append(window)

    foreground = _filled_union(
        foreground_buildings, (0.018, 0.034, 0.075, 1.0)
    )
    building_edge = _union(foreground_buildings).outline(
        (0.16, 0.57, 0.67, 0.48), (0.0, 0.0, 0.0, 0.0), inflate=-0.7
    )
    warm_windows = _filled_union(
        window_shapes,
        (1.0, 0.49 + 0.12 * sin(PHASE * 2.0), 0.22, 0.9),
    )
    cool_windows = _filled_union(
        cool_window_shapes, (0.23, 0.78, 0.94, 0.78)
    )

    rain = []
    for index in range(19):
        x = 28 + (index * 137 % 900)
        y = HEIGHT - (42 + (index * 71 % 180)) - 7 * sin(PHASE + index * 1.8)
        rain.append(
            _rect((x, y), (1.4, 18 + index % 4 * 5), angle=-0.12)
        )
    rain_layer = _filled_union(rain, (0.23, 0.43, 0.58, 0.38))

    track = _rect((WIDTH / 2, HEIGHT - 326), (WIDTH, 3)).fill(
        (0.15, 0.74, 0.82, 0.68), (0.0, 0.0, 0.0, 0.0)
    )
    track_shadow = _rect((WIDTH / 2, HEIGHT - 334), (WIDTH, 2)).fill(
        (0.08, 0.22, 0.33, 0.9), (0.0, 0.0, 0.0, 0.0)
    )
    train_trails = _filled_union(
        [
            _rect((train_x - 117, HEIGHT - 300), (86, 2), corner_radius=1),
            _rect((train_x - 105, HEIGHT - 306), (64, 1.2), corner_radius=0.5),
        ],
        (0.13, 0.82, 0.90, 0.62),
    )
    train = _rounded_rect((train_x, HEIGHT - 303), (158, 38), 8).fill(
        (0.92, 0.30, 0.20, 1.0), (0.0, 0.0, 0.0, 0.0)
    )
    train_glass = _filled_union(
        [
            _rounded_rect((train_x - 43, HEIGHT - 306), (25, 15), 2),
            _rounded_rect((train_x - 8, HEIGHT - 306), (25, 15), 2),
            _rounded_rect((train_x + 27, HEIGHT - 306), (25, 15), 2),
        ],
        (0.045, 0.16, 0.25, 1.0),
    )
    train_highlight = _rect((train_x + 9, HEIGHT - 285), (86, 2)).fill(
        (1.0, 0.66, 0.42, 0.9), (0.0, 0.0, 0.0, 0.0)
    )

    water = _rect((WIDTH / 2, HEIGHT - 486), (WIDTH, 108)).fill(
        (0.018, 0.055, 0.095, 0.96), (0.0, 0.0, 0.0, 0.0)
    )
    warm_reflections = []
    cool_reflections = []
    for index in range(23):
        x = 22 + (index * 173 % 930)
        y = HEIGHT - (450 + (index * 29 % 82)) - 3 * sin(PHASE * 2.0 + index)
        length = 18 + (index * 19 % 68)
        streak = _rect((x, y), (length, 1.2 + index % 3 * 0.55), corner_radius=0.5)
        (warm_reflections if index % 4 == 0 else cool_reflections).append(streak)
    warm_water = _filled_union(
        warm_reflections, (0.98, 0.39, 0.19, 0.68)
    )
    cool_water = _filled_union(
        cool_reflections, (0.14, 0.66, 0.77, 0.53)
    )
    train_reflection = _rect((train_x - 22, HEIGHT - 463), (138, 5)).fill(
        (0.95, 0.27, 0.14, 0.36), (0.0, 0.0, 0.0, 0.0)
    )

    layers = (
        atmospheric_glow,
        sun,
        sun_halo,
        rain_layer,
        distant_skyline,
        middle_skyline,
        foreground,
        building_edge,
        warm_windows,
        cool_windows,
        track_shadow,
        track,
        train_trails,
        train,
        train_glass,
        train_highlight,
        water,
        warm_water,
        cool_water,
        train_reflection,
    )
    scene = sky
    for layer in layers:
        scene = scene.alpha_overlay(layer)
    return scene


def kinetic_city_animation(
    frames=96, output=Path("out/generative_art/blue_hour_relay")
):
    """Render the looping city animation and its PNG preview."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    phase = Parameter(
        "phase",
        0.0,
        animate=lambda progress: progress * 2.0 * math.pi,
    )
    with Canvas((WIDTH, HEIGHT)) as canvas:
        scene = kinetic_city_scene()
        scene.save(
            output.with_suffix(".png"),
            canvas=canvas,
            params={"phase": phase},
            progress=0.5,
        )
        canvas.save_animation(
            scene,
            output.with_suffix(".mp4"),
            frames=frames,
            params={"phase": phase},
            fps=30,
        )


if __name__ == "__main__":
    kinetic_city_animation()
