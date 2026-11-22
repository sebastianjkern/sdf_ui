import math

import numpy as np
import pytest
from PIL import Image

from sdf_ui import Canvas, HeightTexture, paint, param, sdf, surface
from sdf_ui.learning import Objective, ParameterStore

_IMAGE = "image"
_RIDGE_X = "ridge_x"
_STEP = 0.002
_MIN_GRADIENT = 1e-5


def _panel():
    return sdf.rounded_rect((25, 18), (22, 14), 4).subtract(sdf.circle((31, 16), 4))


def test_surface_shading_clips_only_alpha_and_matches_tiled(tmp_path):
    shape = _panel()
    ramp = surface.ramp((0, 0), (0.3, -3 / 20))
    height = ramp.add(surface.gaussian((13, 23), (4.0, 8.0), 5.0))
    scene = shape.light_surface(height, "#7b9eae")
    with Canvas((50, 36)) as canvas:
        with canvas.render(shape.light_surface(ramp, "#7b9eae")) as flat:
            constant = flat.read()
        with canvas.render(scene) as rendered:
            full = rendered.read()
        path = canvas.save_tiled(scene, tmp_path / "tiled.png", tile_size=(11, 9))

    np.testing.assert_array_equal(constant[..., 3], full[..., 3])
    np.testing.assert_array_equal(constant[20, 8], constant[20, 15])
    assert not np.array_equal(constant[23, 10], full[23, 10])
    assert full[16, 31, 3] == 0
    with Image.open(path) as tiled:
        np.testing.assert_array_equal(np.asarray(tiled), full[::-1])


def _learning_scene(params):
    center = params.bind(_RIDGE_X, 10.4)
    amplitude = params.bind("amplitude", 3.4)
    light_x = params.bind("light_x", 0.39)
    height = surface.ramp((0, 0), (0.12, 0.08)).add(
        surface.gaussian((center, 12.2), (3.3, 6.6), amplitude)
    )
    image = _panel().light_surface(height, "#678ca2", light_dir=(light_x, 0.3, 0.94))
    return height, image


def _finite_difference(graph, target, parameter):
    value = parameter.item()
    losses = []
    for delta in (_STEP, -_STEP):
        parameter.assign(value + delta)
        with graph.run(target) as run:
            losses.append(float(run.loss))
    parameter.assign(value)
    return (losses[0] - losses[1]) / (2 * _STEP)


def test_learning_surface_matches_native_finite_differences():
    params = ParameterStore()
    height, scene = _learning_scene(params)
    target = Objective().mse(_IMAGE, 0.31).mse("height", 0.8, weight=0.025)
    with (
        Canvas((30, 24)) as canvas,
        canvas.compile({_IMAGE: scene, "height": height}, params) as graph,
    ):
        with graph.run(target) as execution:
            execution.backward()
            native = execution.read(_IMAGE)
        reference = graph.render(backend="reference")[_IMAGE].numpy()
        np.testing.assert_allclose(native, reference, atol=2e-5)

        with graph.run() as cached:
            assert cached.cache_hits == len(graph.plan.passes)
            np.testing.assert_array_equal(cached.read(_IMAGE), native)

        for name in (_RIDGE_X, "amplitude", "light_x"):
            parameter = params[name]
            gradient = float(parameter.grad)
            estimate = _finite_difference(graph, target, parameter)
            assert abs(gradient) > _MIN_GRADIENT
            assert gradient == pytest.approx(estimate, rel=0.02, abs=2e-5)


def test_hatch_paths_use_height_field_and_follow_light():
    shape = _panel()
    height = surface.gaussian((param(_RIDGE_X), 18), (5.0, 13.0), 9.0)
    passes = (
        paint.HatchPass(math.pi / 4, 3, min_shadow=0.22),
        paint.HatchPass(-math.pi / 4, 3, min_shadow=0.57),
    )

    def drawing(canvas, direction, ridge_x):
        style = paint.HatchStyle(
            passes, light=paint.HatchLight(direction=direction, surface=height)
        )
        scene = paint.scene(paint.fill(shape, "#294657", hatch=style))
        return scene.to_vector(canvas=canvas, params={_RIDGE_X: ridge_x})

    with Canvas((50, 36)) as canvas:
        left = drawing(canvas, (-1, 0, 1), 15)
        right = drawing(canvas, (1, 0, 1), 15)
        moved = drawing(canvas, (-1, 0, 1), 19)
    assert left.paths
    assert left != right
    assert left != moved


def test_materialized_height_retains_graph_type_and_value():
    field = surface.ramp((0, 0), (0.2, -3 / 10))
    with Canvas((9, 7)) as canvas:
        with canvas.render(field) as source:
            assert isinstance(source, HeightTexture)
            with canvas.compile({"height": source}) as graph, graph.run() as execution:
                np.testing.assert_allclose(execution.read("height"), source.read())
        with pytest.raises(TypeError, match="should be height"):
            sdf.circle((4, 3), 2).light_surface(sdf.circle((4, 3), 2), "#fff")


def test_surface_uses_ink_alpha_and_validates_lazy_gaussian_spread():
    height = surface.gaussian((4, 3), (param("spread"), 2), 2)
    image = sdf.circle((4, 3), 2).light_surface(height, "#6699cc80")
    with Canvas((9, 7)) as canvas:
        with canvas.render(image, {"spread": 2}) as rendered:
            assert rendered.read()[3, 4, 3] == 128
        with pytest.raises(ValueError, match="two positive values"):
            canvas.render(image, {"spread": 0})
