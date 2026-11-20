from pathlib import Path

from sdf_ui import Canvas, sdf

OUTPUT_DIR = Path("out/examples")

def masked_union_example():
    SIZE = (1080, 1080)

    with Canvas(SIZE) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        points = [(245, 301),
            (362, 485),
            (306, 292),
            (66, 342),
            (213, 198),
            (303, 18),
            (344, 174),
            (443, 482),
            (163, 294),
            (161, 453)]
        discs = [sdf.circle(point, 50).cache(f"disc_{index}") for index, point in enumerate(points)]

        combined_discs = sdf.union_all(discs).cache("combined_discs")
        cache = {}

        combined_discs.save(OUTPUT_DIR / "masked_union_square.png", canvas=ctx, cache=cache)
