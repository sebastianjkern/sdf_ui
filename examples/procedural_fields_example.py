"""Show finite SDF repetition, composable vector fields, and field inspection."""

from pathlib import Path

from sdf_ui import Canvas, color, field, sdf

OUTPUT_DIR = Path("out/examples")


def procedural_fields_example():
    size = (900, 600)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with Canvas(size) as canvas:
        backdrop = color.clear((0.018, 0.026, 0.055, 1.0))
        # A finite grid keeps repeated source geometry clipped to its own tiles.
        tile = sdf.circle((450, 300), 24).union(sdf.circle((450, 300), 10))
        grid = tile.repeat_grid(7, 5, (72, 72), center=(450, 300))
        grid_color = grid.fill((0.24, 0.76, 0.92, 1.0), (0.0, 0.0, 0.0, 0.0))
        grid_lines = grid.isolines(
            (0.68, 0.93, 1.0, 0.85), (0.0, 0.0, 0.0, 0.0),
            spacing=8, line_width=0.8, feather=0.35,
        )
        grid_scene = backdrop.alpha_overlay(grid_color).alpha_overlay(grid_lines)
        grid_scene.save(OUTPUT_DIR / "finite_field_grid.png", ctx=canvas)

        flow = field.vortex((450, 300), 260, 0.75).add(
            field.radial((300, 300), 220, -0.28)
        ).add(field.turbulence(frequency=0.018, amplitude=10, octaves=3)).add(
            field.constant((0.8, -0.25)).scale(0.5)
        )
        source = (
            sdf.circle((450, 300), 178)
            .subtract(sdf.circle((450, 300), 126))
            .union(sdf.circle((450, 300), 68))
        )
        warped = source.warp(flow, strength=1.0, steps=2)
        shape = warped.fill((1.0, 0.48, 0.28, 1.0), (0.0, 0.0, 0.0, 0.0))
        contours = warped.isolines(
            (0.30, 0.83, 0.95, 0.88), (0.0, 0.0, 0.0, 0.0),
            spacing=12, line_width=0.8, feather=0.4,
        )
        visual = flow.visualize(magnitude=0.075).transparency(0.62)
        inspected = flow.contours(spacing=10, line_width=1.0).transparency(0.48)
        field_backdrop = color.radial_gradient(
            (450, 300), (0.06, 0.12, 0.24, 1.0), (0.025, 0.035, 0.08, 1.0),
            inner=20, outer=520,
        ).warp(flow, strength=1.5, steps=2)
        scene = field_backdrop.alpha_overlay(shape).alpha_overlay(contours)
        scene = scene.alpha_overlay(visual).alpha_overlay(inspected)
        scene.save(OUTPUT_DIR / "composed_vector_fields.png", ctx=canvas)


if __name__ == "__main__":
    procedural_fields_example()
