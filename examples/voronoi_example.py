from pathlib import Path

from sdf_ui import Canvas, color, sdf

OUTPUT_DIR = Path("out/examples")

def voronoi_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        points = [
            (150, 170), (390, 130), (690, 165), (940, 210),
            (235, 465), (515, 390), (820, 500),
            (115, 790), (410, 730), (700, 850), (980, 760),
        ]
        palette = [
            (0.95, 0.39, 0.30, 1), (0.95, 0.68, 0.31, 1),
            (0.82, 0.82, 0.38, 1), (0.37, 0.72, 0.52, 1),
            (0.31, 0.68, 0.76, 1), (0.34, 0.48, 0.80, 1),
        ]
        cells = [
            sdf.voronoi(points, cell=index).fill(
                palette[index % len(palette)], (0.0, 0.0, 0.0, 0.0)
            )
            for index in range(len(points))
        ]
        filled = cells[0]
        for cell in cells[1:]:
            filled = filled.alpha_overlay(cell)
        filled = filled.to_rgb()
        backdrop = color.radial_gradient((540, 540), (0.16, 0.18, 0.22, 1.0), (0.03, 0.04, 0.06, 1.0), outer=640)
        cache = {}
        scene = backdrop.alpha_overlay(filled)
        scene.save(OUTPUT_DIR / "voronoi_square.png", canvas=ctx, cache=cache)
