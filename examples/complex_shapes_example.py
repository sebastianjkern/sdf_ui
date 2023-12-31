from src.sdf_ui import *

COLORS = [
    "#62bb47",
    "#fcb827",
    "#f6821f",
    "#e03a3c",
    "#963d97",
    "#009ddc"
]

def complex_shapes_example():
    with Context((500, 500)):
        d1 = disc((175, 250), 80)
        d2 = disc((325, 250), 80)
        union = d1.smooth_union(d2)
        c0 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), -50)
        c1 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), 0)
        c2 = union.outline(hex_col(COLORS[0]), (0.0, 0.0, 0.0, 0.0), 50)
        c = c0.alpha_overlay(c1).alpha_overlay(c2)
        c.show()
