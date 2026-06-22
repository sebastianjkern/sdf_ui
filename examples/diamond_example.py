from pathlib import Path

from sdf_ui import Canvas, sdf

OUTPUT_DIR = Path("out/examples")


def diamond_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sdf.diamond((540, 540), (260, 200)).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).save(str(OUTPUT_DIR / "diamond_square.png"), ctx)
