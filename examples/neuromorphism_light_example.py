from pathlib import Path

from sdf_ui import Canvas, color, sdf

WIDTH = 1600
HEIGHT = 1050
TRANSPARENT = "#00000000"


def _over(base, *layers):
    for layer in layers:
        base = base.alpha_overlay(layer)
    return base


def _rounded_rect(center, half_size, radius):
    return sdf.rounded_rect(
        center,
        half_size,
        (radius, radius, radius, radius),
    )


def _lit(shape, surface="#f2f5f8", background=TRANSPARENT, bevel=78):
    return shape.light(
        fg_color=surface,
        bg_color=background,
        light_dir=(-0.55, -0.72, 0.42),
        ambient=0.62,
        diffuse=0.62,
        specular=0.05,
        shininess=40.0,
        normal_strength=5.2,
        bevel=bevel,
        shade_background=True,
        background_bevel=bevel,
        height_profile="circle_in",
        height_gamma=0.72,
        height=0.5,
    )


def neuromorphism_light_example(output=None):
    output = Path(output) if output else None
    cache = {}

    with Canvas((WIDTH, HEIGHT)) as ctx:
        background = color.clear("#edf1f5").cache("background")

        slab = _rounded_rect(
            ("50%x", "50%y"),
            ("30.5%x", "28%y"),
            "8.75%min",
        )
        circle_cutout = sdf.circle(("34%x", "57%y"), "8%min")
        small_cutout = sdf.circle(("69.5%x", "50%y"), "6.1%min")
        lower_cutout = _rounded_rect(
            ("50%x", "37%y"),
            ("11.25%x", "3%y"),
            "3%y",
        )

        cut_plane = (
            slab.subtract(circle_cutout)
            .subtract(small_cutout)
            .subtract(lower_cutout)
        )

        plane = _lit(cut_plane, background="#edf1f5", bevel="8%min")

        scene = _over(
            background,
            plane,
        )

        with ctx.session(cache=cache) as renderer:
            rendered = renderer.render(scene.to_rgb())
            if output is not None:
                rendered.save(str(output), conversion=False)
            else:
                rendered.show(conversion=False)
