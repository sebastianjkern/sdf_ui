"""Render an SDF landscape with layered aurora, hills, moon, and stars."""

from pathlib import Path

from sdf_ui import Canvas, color, field, sdf


OUTPUT = Path("out/examples/aurora_landscape.png")
SIZE = (960, 640)


def aurora_landscape(output=OUTPUT):
    width, height = SIZE
    sky = color.clear((0.012, 0.025, 0.09, 1.0))
    aurora = color.radial_gradient(
        (width * 0.48, height * 0.78),
        (0.13, 0.88, 0.68, 0.68),
        (0.0, 0.0, 0.0, 0.0),
        inner=20,
        outer=390,
    )
    ribbon = sdf.polyline(
        [
            (20, 360),
            (130, 440),
            (260, 405),
            (390, 500),
            (520, 450),
            (650, 530),
            (800, 465),
            (950, 520),
        ],
        radius=38,
    ).warp(
        field.turbulence(frequency=0.006, amplitude=24, octaves=3, seed=9),
        strength=0.35,
    )
    ribbon_glow = ribbon.fill((0.08, 0.72, 0.54, 0.35), (0, 0, 0, 0))
    scene = sky.alpha_overlay(aurora).alpha_overlay(ribbon_glow)

    moon = sdf.circle((760, 470), 48).fill((0.98, 0.88, 0.62, 1.0))
    moon_cut = sdf.circle((780, 490), 44).fill((0.0, 0.0, 0.0, 0.0))
    scene = scene.alpha_overlay(moon).alpha_overlay(moon_cut)

    far_hill = sdf.polyline(
        [(0, 185), (160, 245), (330, 195), (490, 250), (690, 180), (960, 240)],
        radius=230,
    ).fill((0.055, 0.19, 0.23, 1.0))
    near_hill = sdf.polyline(
        [(0, 115), (190, 170), (390, 95), (610, 155), (810, 105), (960, 155)],
        radius=200,
    ).fill((0.018, 0.075, 0.12, 1.0))
    scene = scene.alpha_overlay(far_hill).alpha_overlay(near_hill)

    stars = [
        sdf.circle((x, y), radius).fill((0.88, 0.94, 1.0, 0.9))
        for x, y, radius in (
            (82, 520, 2), (174, 570, 1.5), (280, 535, 2), (370, 590, 1.5),
            (485, 555, 2), (585, 600, 1.5), (670, 555, 2), (870, 570, 1.5),
        )
    ]
    for star in stars:
        scene = scene.alpha_overlay(star)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas(SIZE) as canvas:
        scene.save(output, canvas=canvas)
    return output


if __name__ == "__main__":
    print(aurora_landscape())
