from pathlib import Path
from random import choice, random

from sdf_ui import Canvas, color, sdf

WIDTH = 1000
HEIGHT = 1250


def _over(base, *layers):
    for layer in layers:
        base = base.alpha_overlay(layer)
    return base


def dawn_example(output="dawn.png"):
    output = Path(output)
    cache = {}

    palette = [
        "#f1b82b",
        "#268ac2",
        "#499fae",
        "#c32c1b",
        "#ee8a2d",
        "#f57b78",
        "#2c8956",
        "#eac1b0",
        "#cc411e",
        "#f4b82f",
    ]

    red = "#cc411e"
    yellow = "#f4b82f"
    blue = "#1c328b"

    with Canvas((WIDTH, HEIGHT)) as ctx:
        background = color.clear(red).cache("background")

        sea_sdf = sdf.rect(
            (WIDTH * 0.50, HEIGHT * 0.249),
            (WIDTH * 0.55, HEIGHT * 0.25),
            (0, 0, 0, 0),
        ).cache("sea_sdf")

        sea = sea_sdf.fill(blue, "#00000000").cache("sea")

        sun_sdf = sdf.circle(
            (WIDTH * 0.50, HEIGHT * 0.50),
            WIDTH * 0.25,
        ).cache("sun_sdf")

        # Equivalent to the old:
        # sea_mask = sea_sdf.generate_mask()
        # sun = sun.mask(transparent, sea_mask.invert())
        #
        # This keeps only the part of the sun outside the sea shape.

        

        sun = (
            (sun_sdf - sea_sdf)
            .fill(yellow, "#00000000")
            .cache("sun")
        )

        stripe_size = (WIDTH * 0.25, 5)
        distance = (HEIGHT * 0.50 - 40) / 20

        stripe_layers = []
        for i, y in enumerate(range(20, int(HEIGHT * 0.50) - 20, int(distance))):
            x = WIDTH * 0.50 + (random() - 0.5) * WIDTH * 0.25
            stripe_color = choice(palette)

            stripe = (
                sdf.rect(
                    (x, y),
                    stripe_size,
                    (0, 0, 0, 0),
                )
                .fill(stripe_color, "#00000000")
                .cache(f"stripe_{i}")
            )

            stripe_layers.append(stripe)

        # Film grain does not appear in the first script's new API.
        # If your installed sdf_ui version exposes it under the new API,
        # add it here as another overlay layer.

        scene = _over(
            background,
            sea,
            sun,
            *stripe_layers,
        ).to_rgb()

        with ctx.session(cache=cache) as renderer:
            rendered = renderer.render(scene)
            rendered.save(str(output))

            stats = renderer.stats
            cache_info = renderer.cache_info()

            print(
                f"Saved {output} "
                f"({stats.shader_dispatches} dispatches, "
                f"{stats.texture_allocations} textures, "
                f"{cache_info.hits} cache hits, "
                f"{cache_info.misses} cache misses, "
                f"{stats.elapsed_seconds:.3f}s)"
            )


if __name__ == "__main__":
    dawn_example()