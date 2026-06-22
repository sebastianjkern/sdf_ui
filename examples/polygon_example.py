from pathlib import Path

from sdf_ui import Canvas, sdf

OUTPUT_DIR = Path("out/examples")


def polygon_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sdf.polygon(
            (
                (220, 160),
                (860, 160),
                (860, 300),
                (520, 300),
                (520, 780),
                (860, 780),
                (860, 920),
                (220, 920),
            )
        ).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).save(str(OUTPUT_DIR / "polygon_square.png"), ctx)
