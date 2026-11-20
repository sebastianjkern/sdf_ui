"""Show finite SDF repetition, composable vector fields, and field inspection."""

from pathlib import Path

from sdf_ui import Canvas, color, field, sdf

OUTPUT_DIR = Path("out/examples")


def _render_grid(canvas):
    backdrop = color.clear((0.018, 0.026, 0.055, 1.0))
    tile = sdf.circle((450, 300), 24).union(sdf.circle((450, 300), 10))
    grid = tile.repeat_grid(7, 5, (72, 72), center=(450, 300))
    grid_color = grid.fill((0.24, 0.76, 0.92, 1.0), (0.0, 0.0, 0.0, 0.0))
    grid_lines = grid.isolines(
        (0.68, 0.93, 1.0, 0.85), (0.0, 0.0, 0.0, 0.0),
        spacing=8, line_width=0.8, feather=0.35,
    )
    backdrop.alpha_overlay(grid_color).alpha_overlay(grid_lines).save(
        OUTPUT_DIR / "finite_field_grid.png", canvas=canvas
    )


def _make_flow():
    return field.vortex((450, 300), 260, 0.75).add(
        field.radial((300, 300), 220, -7 / 25)
    ).add(field.turbulence(frequency=0.018, amplitude=10, octaves=3)).add(
        field.constant((0.8, -1 / 4)).scale(0.5)
    )


def _make_warped_shape(flow):
    source = (
        sdf.circle((450, 300), 178).subtract(sdf.circle((450, 300), 126))
        .union(sdf.circle((450, 300), 68))
    )
    warped = source.warp(flow, strength=1.0, steps=2)
    shape = warped.fill((1.0, 0.48, 0.28, 1.0), (0.0, 0.0, 0.0, 0.0))
    contours = warped.isolines(
        (0.30, 0.83, 0.95, 0.88), (0.0, 0.0, 0.0, 0.0),
        spacing=12, line_width=0.8, feather=0.4,
    )
    return shape, contours


def _make_field_overlays(flow):
    visualization = flow.visualize(magnitude=0.075).transparency(0.62)
    inspection = flow.contours(spacing=10, line_width=1.0).transparency(0.48)
    backdrop = color.radial_gradient(
        (450, 300), (0.06, 0.12, 0.24, 1.0), (0.025, 0.035, 0.08, 1.0),
        inner=20, outer=520,
    ).warp(flow, strength=1.5, steps=2)
    return backdrop, visualization, inspection


def _render_vector_field(canvas):
    flow = _make_flow()
    shape, contours = _make_warped_shape(flow)
    backdrop, visualization, inspection = _make_field_overlays(flow)
    scene = backdrop.alpha_overlay(shape).alpha_overlay(contours)
    scene.alpha_overlay(visualization).alpha_overlay(inspection).save(
        OUTPUT_DIR / "composed_vector_fields.png", canvas=canvas
    )


def procedural_fields_example():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with Canvas((900, 600)) as canvas:
        _render_grid(canvas)
        _render_vector_field(canvas)


if __name__ == "__main__":
    procedural_fields_example()
