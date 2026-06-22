from functools import reduce
from pathlib import Path

import random

from sdf_ui import Canvas, color, sdf

OUTPUT_DIR = Path("out/examples")

def voronoi_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        points = [(random.randint(0, 1080), random.randint(0, 1080)) for _ in range(100)]
        discs = [sdf.circle(point, 20).cache(f"disc_{index}") for index, point in enumerate(points)]

        combined_discs = reduce(lambda x, y: x | y, discs).cache("combined_discs")
        filled = combined_discs.fill((1, 1, 1, 1), (0.1, 0.1, 0.1, 1.0), inflate=25, inner=0, outer=50)
        backdrop = color.radial_gradient((540, 540), (0.16, 0.18, 0.22, 1.0), (0.03, 0.04, 0.06, 1.0), outer=640)
        cache = {}
        scene = backdrop.alpha_overlay(filled)
        scene.save(str(OUTPUT_DIR / "voronoi_square.png"), ctx, cache=cache)
