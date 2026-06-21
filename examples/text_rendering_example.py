from sdf_ui import Canvas, sdf

def text_rendering_example():
    with Canvas((500, 500)) as ctx:
        l1 = sdf.line((250, 100), (50, 450))
        l2 = sdf.line((250, 100), (450, 450))
        l3 = sdf.line((100, 250), (400, 250))

        text = (
            l1.union(l2)
            .union(l3)
            .fill(inflate=5, fg_color=(1, 1, 1, 1), bg_color=(0, 0, 0, 1))
        )
        text.show(ctx)
