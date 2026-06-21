from functools import reduce

from sdf_ui import Canvas, sdf

def masked_union_example():
    SIZE = (512, 512)

    with Canvas(SIZE) as ctx:
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

        combined_discs = reduce(lambda x, y: x.union(y), discs).cache("combined_discs")
        masks = [mask for _, mask in (discs[0].masked_union(disc) for disc in discs[1:])]

        mask = reduce(lambda m1, m2: m1.multiply(m2), masks).uncached()
        cache = {}

        combined_discs.save("voronoi.png", ctx, cache=cache)
        combined_discs.show(ctx, cache=cache)
        mask.show(ctx, cache=cache)
