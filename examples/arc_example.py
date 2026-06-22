from math import pi
from pathlib import Path

from sdf_ui import Canvas, sdf

OUTPUT_DIR = Path("out/examples")


def arc_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sdf.arc((540, 540), 320, -pi * 0.15, pi * 1.15).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
            2,
        ).save(str(OUTPUT_DIR / "arc_square.png"), ctx)
