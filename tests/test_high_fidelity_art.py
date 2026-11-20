
import numpy as np
import pytest
from PIL import Image

from sdf_ui import Canvas, color, field, sdf
from sdf_ui.learning import DifferentiableRenderer, Objective, ParameterStore

_AMOUNT_TEXT = 'amount'

_FINITE_DIFFERENCE_STEP = 1e-5

def _hdr_scene(stops, level, threshold, amount, tone, dither):
    return (
        color.linear_rgba((0.03, 0.05, 0.07, 1))
        .over(sdf.circle((12.2, 11.1), 7).ink_linear((level, 1.2, 0.6, 1)))
        .bloom(threshold=threshold, amount=amount)
        .finish(exposure=stops, tone=tone, dither=dither)
    )


def _hdr_reference(output, target, weights, names):
    renderer = DifferentiableRenderer((24, 22), params=weights)
    difference = renderer.render(output) - renderer.render(target)
    loss = (difference * difference).mean()
    loss.backward()
    gradients = tuple(float(weights[name].grad) for name in names)
    for name in names:
        weights[name].zero_grad()
    return loss, gradients


def test_linear_compositing_keeps_highlight_values_and_encodes_srgb():
    light = color.linear_rgba((2.5, 0.4, 0.02, 0.5))
    scene = color.linear_rgba((0.1, 0.2, 0.02, 1)).over(light)
    with Canvas((7, 5)) as canvas:
        with canvas.render(scene) as raw:
            assert raw.tex.format == "rgba32float"
            np.testing.assert_allclose(
                raw.tex.read()[0, 0], (1.3, 0.3, 0.02, 1), atol=1e-6
            )
        with canvas.render(scene.finish(tone=0, dither=0)) as result:
            assert result.tex.format == "rgba8unorm"
            assert tuple(result.tex.read()[0, 0]) == (255, 149, 39, 255)


def test_hdr_learning_matches_cpu_with_live_finishing_controls():
    initial = (0.45, 2.4, 0.95, 0.2, 0.8, 0.6)
    weights = ParameterStore()
    names = ("exposure", "radiance", "threshold", _AMOUNT_TEXT, "tone", "dither")
    knobs = [weights.bind(name, value) for name, value in zip(names, initial)]

    output = _hdr_scene(*knobs)
    target = _hdr_scene(0.2, 2, 0.8, 0.5, 0.6, 0.4)
    loss, reference_gradients = _hdr_reference(output, target, weights, names)

    with (
        Canvas((24, 22)) as canvas,
        canvas.capture(target) as image,
        canvas.compile({"image": output}, weights) as graph,
    ):
        objective = Objective().mse("image", image)
        with graph.run(objective) as execution:
            assert execution.loss == pytest.approx(loss.item(), abs=3e-7)
            execution.backward()
        gradients = tuple(float(weights[name].grad) for name in names)
        np.testing.assert_allclose(gradients, reference_gradients, atol=1e-7, rtol=5e-4)
        assert all(value != 0 for value in gradients)
        weights["exposure"].assign(0.2)
        with graph.run(objective) as execution:
            assert execution.loss != pytest.approx(loss.item(), abs=3e-7)


def test_film_grain_is_subtle_and_keeps_black_hue_and_alpha():
    rgb = np.array([0.32, 0.14, 0.04], dtype=np.float32)
    source = color.linear_rgba((*rgb, 0.6))
    with Canvas((53, 47)) as canvas:
        with canvas.render(source.film_grain(0)) as plain:
            np.testing.assert_array_equal(
                plain.read(), np.broadcast_to([*rgb, np.float32(0.6)], plain.tex.shape)
            )
        with canvas.render(source.film_grain()) as grain:
            values = grain.read()
        with canvas.render(color.linear_rgba((0, 0, 0, 0.6)).film_grain()) as black:
            assert np.all(black.read()[..., :3] == 0)
    multipliers = values[..., :3] / rgb
    assert np.array_equal(values[..., 3], np.full(values.shape[:2], 0.6, np.float32))
    assert np.allclose(multipliers, multipliers[..., :1])
    assert abs(multipliers.mean() - 1) < 3 / 1_000
    assert 1 / 200 < multipliers.std() < 1 / 25


def test_film_grain_controls_train_on_gpu_and_match_cpu():
    size = (23, 19)
    weights = ParameterStore()
    amount = weights.bind(_AMOUNT_TEXT, 0.15)
    grain_size = weights.bind("size", 1.47)
    source = color.linear_rgba((0.45, 0.22, 0.08, 1))
    output = source.film_grain(amount, size=grain_size)
    target = source.film_grain(0.25, size=1.52)
    loss, expected = _film_grain_reference(size, weights, output, target)

    with (
        Canvas(size) as canvas,
        canvas.capture(target) as image,
        canvas.compile({"image": output}, weights) as graph,
    ):
        objective = Objective().mse("image", image)
        with graph.run(objective) as execution:
            assert execution.loss == pytest.approx(loss.item(), abs=1e-7)
            execution.backward()
        assert all(value != 0 for value in expected)
        np.testing.assert_allclose(
            [weights[name].grad for name in (_AMOUNT_TEXT, "size")],
            expected,
            atol=1e-7,
            rtol=1e-4,
        )
        weights[_AMOUNT_TEXT].assign(0.25)
        weights["size"].assign(1.52)
        with graph.run(objective) as execution:
            assert execution.loss == pytest.approx(0, abs=1e-10)


def _film_grain_reference(size, weights, output, target):
    names = (_AMOUNT_TEXT, "size")
    reference = DifferentiableRenderer(size, params=weights)
    error = reference.render(output) - reference.render(target)
    loss = (error * error).mean()
    loss.backward()
    expected = [float(weights[name].grad) for name in names]
    target_pixels = reference.render(target)
    _check_film_grain_finite_differences(reference, output, target_pixels, weights, names, expected)
    return loss, expected


def _check_film_grain_finite_differences(reference, output, target_pixels, weights, names, expected):
    for name, derivative in zip(names, expected):
        knob = weights[name]
        start = knob.item()
        estimates = _film_grain_estimates(reference, output, target_pixels, knob, start)
        assert (estimates[1] - estimates[0]) / (2 * _FINITE_DIFFERENCE_STEP) == pytest.approx(
            derivative, abs=1e-8
        )
        knob.assign(start)
        knob.zero_grad()


def _film_grain_estimates(reference, output, target_pixels, knob, start):
    estimates = []
    for delta in (-_FINITE_DIFFERENCE_STEP, _FINITE_DIFFERENCE_STEP):
        knob.assign(start + delta)
        difference = reference.render(output) - target_pixels
        estimates.append((difference * difference).mean().item())
    return estimates


def _tiled_scene(flow, linear):
    shape = sdf.circle((45, 38), 18).warp(flow, strength=0.75, steps=4)
    if linear:
        return (
            sdf.circle((30, 20), 0)
            .fill_linear((0.05, 0.07, 0.10, 1), (0.01, 0.014, 0.027, 1), outer=60)
            .over(shape.ink_linear((3.4, 1.5, 0.5, 0.9)))
            .warp(flow, strength=1.2, steps=3)
            .bloom(amount=0.6, passes=2)
            .film_grain(0.12, size=1.35)
            .finish(tone=0.6)
        )
    return (
        color.clear("#101425")
        .over(shape.ink("#ffc87a"), sdf.circle((60, 20), 8).ink("#94abff"))
        .warp(flow, strength=1.2, steps=3)
        .post.blur(n=1, base=13)
    )


@pytest.mark.parametrize("linear", [False, True])
def test_tiled_still_matches_whole_frame_with_nested_warps_and_blur(tmp_path, linear):
    flow = field.vortex((45, 36), 24, 9).add(field.constant((2.1, -17 / 10))).scale(0.8)
    scene = _tiled_scene(flow, linear)
    full = tmp_path / "whole.png"
    tiled = tmp_path / "tiled.png"
    with Canvas((83, 67), device_scale=2) as canvas:
        canvas.save(scene, full)
        canvas.save_tiled(scene, tiled, tile_size=(57, 46))
    with Image.open(full) as reference, Image.open(tiled) as result:
        np.testing.assert_array_equal(np.asarray(result), np.asarray(reference))


def test_tiled_export_rejects_unbounded_sampling_before_creating_a_file(tmp_path):
    path = tmp_path / "unsupported.png"
    scene = sdf.circle((16, 16), 4).repeat_grid(2, 2, (8, 8), (16, 16)).ink("#ffffff")
    with Canvas((32, 32)) as canvas:
        with pytest.raises(NotImplementedError, match="repeat_grid"):
            canvas.save_tiled(scene, path, tile_size=(16, 16))
    assert not path.exists()
