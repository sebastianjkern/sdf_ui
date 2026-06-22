from sdf_ui import Canvas, sdf


def isolines_example():
    with Canvas((1920, 1080)) as ctx:
        sdf.disc(("50%x", "50%y"), 0).isolines(
            fg_color="#0f172a",
            bg_color="#f8fafc",
            spacing=100,
            line_width=4,
            feather=0,
            phase=0.0,
        ).show(ctx)
