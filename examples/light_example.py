from pathlib import Path

from sdf_ui import Canvas, sdf

OUTPUT_DIR = Path("out/examples")


def light_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        background = "#071117"
        light_dir = (-0.55, -0.75, 0.8)

        large_disc = sdf.circle(("38%x", "50%y"), "22%min").light(
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

        small_disc = sdf.circle(("60%x", "55%y"), "15%min").light(
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

        ring = sdf.circle(("70%x", "40%y"), "10%min").subtract(
            sdf.circle(("70%x", "40%y"), "6%min")
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

        large_disc.alpha_overlay(small_disc).alpha_overlay(ring).save(str(OUTPUT_DIR / "light_square.png"), ctx)
