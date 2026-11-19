from pathlib import Path

from sdf_ui import Canvas, color, sdf

TILING = 10
OUTPUT_DIR = Path("out/examples")

FG_COLOR = (0, 0, 0, 1)
BG_COLOR = (1, 1, 1, 1)
TRANSPARENT = (0, 0, 0, 0)


def impossible_city_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        grid_height = ctx.percent(100 / TILING)
        grid_width = ctx.percent(100 / (TILING + 2))

        bg = color.clear(BG_COLOR).cache("city_background")

        cubes = []

        for w in range(int((TILING + 2) / 2 + 0.5)):
            for h in range(int(TILING / 2 + 0.5)):
                cube = sdf.isometric_cube(
                    (grid_width + w * 2 * grid_width, h * 2 * grid_height),
                    grid_width,
                    grid_height,
                    stroke_width=4,
                ).fill(FG_COLOR, TRANSPARENT)
                cubes.append(cube)

        scene = bg.over(*cubes, cache=False)
        scene.save(OUTPUT_DIR / "impossible_city_square.png", canvas=ctx)
