import numpy as np

from sdf_ui import Canvas, color, param, sdf


def _pixels(image):
    return np.asarray(image.read())


def test_automatic_dirty_render_matches_full_render_when_circle_moves():
    shape = sdf.circle((param("x"), 24), 7).fill(
        (1.0, 0.2, 0.1, 1.0), (0.0, 0.0, 0.0, 0.0)
    )
    scene = color.clear((0.0, 0.0, 0.0, 0.0)).alpha_overlay(shape).to_rgb()

    with Canvas((64, 48)) as canvas:
        with canvas.session(params={"x": 12}, dirty_regions=True) as renderer:
            renderer.render(scene)
            renderer.params = {"x": 48}
            dirty = renderer.render(scene)
            dirty_pixels = _pixels(dirty.tex)
            assert renderer.stats.dirty_region_pixels > 0

        with canvas.session(params={"x": 48}) as renderer:
            full = renderer.render(scene)
            full_pixels = _pixels(full.tex)

    assert np.array_equal(dirty_pixels, full_pixels)


def test_automatic_dirty_render_reuses_static_scene_after_seed_frame():
    scene = color.clear("#203040").to_rgb()
    with Canvas((32, 24)) as canvas:
        with canvas.session(dirty_regions=True) as renderer:
            renderer.render(scene)
            previous = renderer.stats.dirty_layer_hits
            renderer.render(scene)
            assert renderer.stats.dirty_layer_hits > previous
