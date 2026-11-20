import numpy as np
import pytest
from PIL import Image

from sdf_ui import Canvas, color, field, sdf
from sdf_ui.learning import Objective, ParameterStore


def test_per_cell_and_per_pixel_sample_distinct_coordinates():
    # The X component is positive at (19, 10), but smaller at its cell center.
    map_texture = field.radial((10, 10), 12, 5)
    per_cell = sdf.circle((0, 0), 3 + map_texture.channel("x").at_cell())
    per_pixel = sdf.circle((0, 0), 3 + map_texture.channel("x").at_pixel())
    with Canvas((40, 20)) as canvas:
        for shape, expected in ((per_cell, 1.0 - 3.0), (per_pixel, None)):
            scene = shape.repeat_grid(2, 1, (20, 20), (20, 10))
            # Use the first cell center to check the reference sample, then
            # compare a point inside that cell to catch per-pixel modulation.
            with canvas.render(scene) as result:
                distances = result.read()
                assert distances[10, 10] == pytest.approx(-3)
                distance = distances[10, 11]
                if expected is None:
                    assert distance < -2.3
                else:
                    assert distance == pytest.approx(expected)


def test_signed_distances_can_drive_a_repeated_vector_field():
    distance = sdf.circle((20, 10), 4).channel().at_cell()
    amplitude = distance.remap((-4, 6), (2, 8))
    template = field.radial((0, 0), radius=9, strength=amplitude)
    scene = template.repeat_grid(3, 1, (10, 20), (20, 10))
    with Canvas((40, 20)) as canvas, canvas.render(scene) as output:
        vectors = output.read()
        assert vectors[10, 11, 0] > vectors[10, 21, 0] > 0
        assert vectors[10, 31, 0] == pytest.approx(vectors[10, 11, 0])


def test_texture_channels_from_uploaded_image_drive_baked_rotation(tmp_path):
    pixels = np.zeros((36, 36, 4), dtype=np.uint8)
    pixels[:, 18:, 0] = 255
    pixels[..., 3] = 255
    map_path = tmp_path / "map.png"
    Image.fromarray(pixels).save(map_path)
    with Canvas((36, 36)) as canvas, canvas.load_image(map_path) as image:
        angle = image.channel("red").at_cell() * (np.pi / 2)
        stamp = sdf.rect((18, 18), (7, 2)).ink_linear((0.8, 0.3, 0.1, 1))
        scene = stamp.repeat_grid(2, 1, (18, 18), (18, 18), rotation=angle, baked=True)
        with canvas.render(scene) as result:
            assert result.read()[18, 14, 3] > 0.9
            assert result.read()[18, 32, 3] < 0.05
            assert result.read()[23, 27, 3] > 0.9
        whole, tiles = tmp_path / "whole.png", tmp_path / "tiles.png"
        with canvas.render(stamp) as rendered_stamp:
            saved_scene = rendered_stamp.repeat_grid(
                2, 1, (18, 18), (18, 18), rotation=angle
            ).finish()
            canvas.save(saved_scene, whole)
            canvas.save_tiled(saved_scene, tiles, tile_size=(13, 11))
        with Image.open(whole) as full, Image.open(tiles) as tiled:
            np.testing.assert_array_equal(np.asarray(full), np.asarray(tiled))


def test_compiled_graph_reuses_plan_and_propagates_gradients_to_the_map():
    weights = ParameterStore()
    width = weights.bind("width", 2)
    chroma = weights.bind("chroma", 0.2)

    def scene(width, chroma):
        guide = color.linear_rgba((chroma, 0, 0, 1))
        radius = width + guide.channel().at_cell()
        dots = sdf.circle((0, 0), radius).repeat_grid(4, 4, (7, 7), (14, 14))
        return dots.ink_linear((0.2, 0.3, 0.8, 1)).finish(tone=0, dither=0)

    with (
        Canvas((28, 28)) as canvas,
        canvas.capture(scene(2.4, 0.5)) as target,
        canvas.compile({"image": scene(width, chroma)}, weights) as graph,
    ):
        objective = Objective().mse("image", target)
        with graph.run(objective) as execution:
            before = execution.loss
            execution.backward()
        direct = float(weights["width"].grad)
        mapped = float(weights["chroma"].grad)
        assert direct < 0
        assert mapped == pytest.approx(direct, abs=3e-5)
        weights["chroma"].assign(0.5)
        with graph.run(objective) as execution:
            assert execution.loss < before


def test_procedural_dependencies_tile_exactly_across_odd_tile_sizes(tmp_path):
    flow = field.turbulence(frequency=0.012, amplitude=1, seed=4)
    intensity = color.linear_rgba((0.5 + 0.5 * flow.channel("x").at_pixel(), 0, 0, 1))
    radius = intensity.channel().at_cell().remap((0, 1), (1, 7))
    turn = flow.channel("y").at_cell() * 2
    stamp = sdf.ngon((0, 0), radius, 5, rotation=turn)
    height = flow.channel("y").at_pixel().remap((-1, 1), (0.2, 1.5))
    painted = stamp.repeat_grid(8, 8, (18, 18), (72, 72)).ink_linear(
        (height, 0.2, 0.1, 1)
    )
    scene = color.linear_rgba((0.04, 0.09, 0.1, 1)).over(painted).finish()
    whole, tiles = tmp_path / "whole.png", tmp_path / "tiles.png"
    with Canvas((144, 144)) as canvas:
        canvas.save(scene, whole)
        canvas.save_tiled(scene, tiles, tile_size=(37, 41))
    with Image.open(whole) as full, Image.open(tiles) as tiled:
        np.testing.assert_array_equal(np.asarray(full), np.asarray(tiled))


def test_spatial_animation_does_not_retain_per_frame_cache_entries():
    from sdf_ui import param

    driver = color.linear_rgba((param("change"), 0, 0, 1)).channel().at_cell()
    scene = sdf.rect((0, 0), (1 + driver, 2)).repeat(6)
    params = {"change": 0.0}
    with (
        Canvas((24, 24)) as canvas,
        canvas.session(params=params, dirty_regions=True) as renderer,
    ):
        first = None
        for i in range(8):
            params["change"] = i / 8
            with renderer.render(scene, dirty_rect="auto") as result:
                current = result.read().copy()
            if first is None:
                first = current
            elif i == 7:
                assert not np.array_equal(current, first)
            assert renderer.cache_info().entries == 0
