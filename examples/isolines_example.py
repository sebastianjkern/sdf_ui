from sdf_ui import Canvas, color, sdf


def isolines_example():
    with Canvas((960, 540)) as ctx:
        background = color.clear("#111827").cache("background")
        shape = (
            sdf.circle(("42%x", "48%y"), "18%min")
            .smooth_union(
                sdf.circle(("58%x", "48%y"), "18%min"),
                k="3%min",
            )
            .smooth_union(
                sdf.rect(
                    ("50%x", "62%y"),
                    ("22%min", "10%min"),
                    ("6%min", "6%min", "6%min", "6%min"),
                ),
                k="4%min",
            )
            .cache("shape")
        )

        contour_field = shape.isolines(
            fg_color="#f8fafc",
            bg_color="#111827",
            spacing=12.0,
            line_width=2.0,
            feather=0.75,
            phase=2.0,
        ).cache("contours")

        image = background.alpha_overlay(contour_field)
        image.show(ctx)

