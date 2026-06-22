from pathlib import Path

from sdf_ui import Canvas, sdf

OUTPUT_DIR = Path("out/examples")


def isolines_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sdf.disc(("50%x", "50%y"), 0).isolines(
            fg_color="#0f172a",
            bg_color="#f8fafc",
            spacing=100,
            line_width=4,
            feather=0,
            phase=0.0,
        ).save(str(OUTPUT_DIR / "isolines_square.png"), ctx)
