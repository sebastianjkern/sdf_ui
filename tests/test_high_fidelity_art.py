import numpy as np
import pytest
from PIL import Image

from sdf_ui import Canvas, color, field, sdf
from sdf_ui.learning import DifferentiableRenderer, Objective, ParameterStore


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
    names = ("exposure", "radiance", "threshold", "amount", "tone", "dither")
    knobs = [weights.bind(name, value) for name, value in zip(names, initial)]

    def scene(stops, level, threshold, amount, tone, dither):
        return (
            color.linear_rgba((0.03, 0.05, 0.07, 1))
            .over(sdf.circle((12.2, 11.1), 7).ink_linear((level, 1.2, 0.6, 1)))
            .bloom(threshold=threshold, amount=amount)
            .finish(exposure=stops, tone=tone, dither=dither)
        )

    output = scene(*knobs)
    target = scene(0.2, 2, 0.8, 0.5, 0.6, 0.4)
    reference = DifferentiableRenderer((24, 22), params=weights)
    difference = reference.render(output) - reference.render(target)
    loss = (difference * difference).mean()
    loss.backward()
    reference_gradients = tuple(float(weights[name].grad) for name in names)
    for name in names:
        weights[name].zero_grad()

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


@pytest.mark.parametrize("linear", [False, True])
def test_tiled_still_matches_whole_frame_with_nested_warps_and_blur(tmp_path, linear):
    flow = field.vortex((45, 36), 24, 9).add(field.constant((2.1, -1.7))).scale(0.8)
    shape = sdf.circle((45, 38), 18).warp(flow, strength=0.75, steps=4)
    if linear:
        scene = (
            sdf.circle((30, 20), 0)
            .fill_linear((0.05, 0.07, 0.10, 1), (0.01, 0.014, 0.027, 1), outer=60)
            .over(shape.ink_linear((3.4, 1.5, 0.5, 0.9)))
            .warp(flow, strength=1.2, steps=3)
            .bloom(amount=0.6, passes=2)
            .finish(tone=0.6)
        )
    else:
        scene = (
            color.clear("#101425")
            .over(shape.ink("#ffc87a"), sdf.circle((60, 20), 8).ink("#94abff"))
            .warp(flow, strength=1.2, steps=3)
            .post.blur(n=1, base=13)
        )
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
