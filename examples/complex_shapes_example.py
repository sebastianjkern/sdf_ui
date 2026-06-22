from pathlib import Path

from sdf_ui import Canvas, sdf
from sdf_ui.util import hex_col

COLORS = [
    "#62bb47",
    "#fcb827",
    "#f6821f",
    "#e03a3c",
    "#963d97",
    "#009ddc"
]

OUTPUT_DIR = Path("out/examples")

def complex_shapes_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        d1 = sdf.circle((380, 540), 180)
        d2 = sdf.circle((700, 540), 180)
        union = d1.smooth_union(d2).cache("union")
        c0 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), -50)
        c1 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), 0)
        c2 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), 50)
        c = c0.alpha_overlay(c1).alpha_overlay(c2)
        c.save(str(OUTPUT_DIR / "complex_square.png"), ctx)
