from sdf_ui import Canvas, sdf


def light_example():
    with Canvas((1920, 1080)) as ctx:
        background = "#071117"
        light_dir = (-0.55, -0.75, 0.8)

        large_disc = sdf.circle(("42%x", "50%y"), "23%min").light(
            fg_color="#47d5ff",
            bg_color=background,
            light_dir=light_dir,
            ambient=0.16,
            diffuse=1.15,
            specular=0.45,
            shininess=28.0,
            normal_strength=8.0,
            bevel="8%min",
        )

        small_disc = sdf.circle(("57%x", "53%y"), "16%min").light(
            fg_color="#ffcf5a",
            bg_color=(0, 0, 0, 0),
            light_dir=light_dir,
            ambient=0.16,
            diffuse=1.15,
            specular=0.45,
            shininess=28.0,
            normal_strength=8.0,
            bevel="6%min",
        )

        ring = sdf.circle(("66%x", "43%y"), "10%min").subtract(
            sdf.circle(("66%x", "43%y"), "6%min")
        ).light(
            fg_color="#f8fafc",
            bg_color=(0, 0, 0, 0),
            light_dir=light_dir,
            ambient=0.12,
            diffuse=1.2,
            specular=0.7,
            shininess=20.0,
            normal_strength=9.0,
            bevel="3.5%min",
        )

        large_disc.alpha_overlay(small_disc).alpha_overlay(ring).show(ctx)
