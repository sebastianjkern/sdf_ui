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

def complex_shapes_example():
    with Canvas((500, 500)) as ctx:
        d1 = sdf.circle((175, 250), 80)
        d2 = sdf.circle((325, 250), 80)
        union = d1.smooth_union(d2).cache("union")
        c0 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), -50)
        c1 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), 0)
        c2 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), 50)
        c = c0.alpha_overlay(c1).alpha_overlay(c2)
        c.show(ctx)
