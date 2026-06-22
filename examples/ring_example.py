from pathlib import Path

from sdf_ui import Canvas, sdf

OUTPUT_DIR = Path("out/examples")


def ring_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sdf.ring((540, 540), 280, 84).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).save(str(OUTPUT_DIR / "ring_square.png"), ctx)
