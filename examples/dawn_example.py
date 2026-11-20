from pathlib import Path
from random import choice, random

from sdf_ui import Canvas, color, sdf

WIDTH = 1000
HEIGHT = 1250


def dawn_example(output=Path("out/examples/dawn/dawn.png")):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
            (WIDTH * 0.50, HEIGHT * 249 / 1_000),
            (WIDTH * 11 / 20, HEIGHT * 1 / 4),
            (0, 0, 0, 0),
        ).cache("sea_sdf")

        sea = sea_sdf.fill(blue, "#00000000").cache("sea")

        sun_sdf = sdf.circle(
            (WIDTH * 0.50, HEIGHT * 0.50),
            WIDTH * 1 / 4,
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

        stripe_layers = _dawn_stripes(palette)

        # Film grain does not appear in the first script's new API.
        # If your installed sdf_ui version exposes it under the new API,
        # add it here as another overlay layer.

        scene = background.over(
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


def _dawn_stripes(palette):
    stripe_size = (WIDTH * 1 / 4, 5)
    distance = (HEIGHT * 0.50 - 40) / 20
    stripe_layers = []
    for index, y in enumerate(range(20, int(HEIGHT * 0.50) - 20, int(distance))):
        x = WIDTH * 0.50 + (random() - 0.5) * WIDTH * 1 / 4
        stripe = sdf.rect((x, y), stripe_size, (0, 0, 0, 0)).fill(
            choice(palette), "#00000000"
        ).cache(f"stripe_{index}")
        stripe_layers.append(stripe)
    return stripe_layers


if __name__ == "__main__":
    dawn_example()
