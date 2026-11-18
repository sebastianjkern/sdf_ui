"""Distort an SDF with two vortices and visualize their streamlines."""

from pathlib import Path

from sdf_ui import Canvas, color, sdf


OUTPUT_DIR = Path("out/examples")


def vortex_field_example():
    size = (960, 720)
    vortices = ((300, 360, 1.15), (660, 360, -1.15))
    with Canvas(size) as canvas:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        source = (
            sdf.circle((480, 360), 250)
            .subtract(sdf.circle((480, 360), 182))
            .union(sdf.circle((480, 360), 86))
        )
        warped = source.vortex_warp(vortices, radius=175, strength=0.5)
        backdrop = color.clear((0.025, 0.035, 0.065, 1.0))
        shape = warped.fill((0.82, 0.86, 0.83, 1.0), (0.0, 0.0, 0.0, 0.0))
        contours = warped.isolines(
            (0.23, 0.36, 0.43, 0.8),
            (0.0, 0.0, 0.0, 0.0),
            spacing=18,
            line_width=0.8,
            feather=0.4,
        )
        flow = color.vortex_streamlines(
            vortices,
            radius=175,
            spacing=118,
            step_size=2.0,
            steps=170,
        )
        markers = color.clear((0.0, 0.0, 0.0, 0.0))
        for center_x, center_y, strength in vortices:
            marker_color = (0.35, 0.95, 0.87, 1.0) if strength > 0 else (1.0, 0.64, 0.34, 1.0)
            marker = sdf.circle((center_x, center_y), 5).fill(
                marker_color, (0.0, 0.0, 0.0, 0.0)
            )
            markers = markers.alpha_overlay(marker)

        scene = backdrop.alpha_overlay(shape)
        scene = scene.alpha_overlay(contours).alpha_overlay(flow).alpha_overlay(markers)
        scene.save(OUTPUT_DIR / "vortex_distortion.png", canvas=canvas)


if __name__ == "__main__":
    vortex_field_example()
