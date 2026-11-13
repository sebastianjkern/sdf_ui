"""Float render values, reusable tapes, numerical gradients, and GL parity."""

import gc
import weakref

import numpy as np
import pytest
from PIL import Image

from sdf_ui import Canvas, color, param, percent_of_min, sdf, sin
from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.registry import registry
from sdf_ui.core.color import ColorTexture
from sdf_ui.core.sdf import SDFTexture
from sdf_ui.learning.autodiff import Parameter, Tensor
from sdf_ui.learning.optimizers import Adam
from sdf_ui.learning.render import DifferentiableRenderer, UnsupportedGradientError


def _check_gradients(scene, parameters, size=(15, 13), *, tolerance=2e-5):
    renderer = DifferentiableRenderer(size, parameters)
    shape = renderer.render(scene).shape
    weights = np.random.default_rng(73).uniform(0.2, 1.0, shape)

    def objective():
        value = renderer.render(scene)
        return (value * weights + value * value * 0.03).mean()

    for parameter in parameters.values():
        parameter.zero_grad()
    objective().backward()
    analytic = {name: parameter.grad.copy() for name, parameter in parameters.items()}
    epsilon = 1e-5
    for name, parameter in parameters.items():
        initial = parameter.numpy()
        numeric = np.zeros_like(initial)
        for index in np.ndindex(initial.shape):
            plus, minus = initial.copy(), initial.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            parameter.assign(plus)
            high = objective().item()
            parameter.assign(minus)
            low = objective().item()
            parameter.assign(initial)
            numeric[index] = (high - low) / (2.0 * epsilon)
        assert np.all(np.isfinite(analytic[name]))
        np.testing.assert_allclose(
            analytic[name], numeric, atol=tolerance, rtol=tolerance, err_msg=name
        )


def test_circle_pixel_coordinates_orientation_and_direct_parameter():
    radius = Parameter("radius", 2.3)
    renderer = DifferentiableRenderer((7, 5))
    scene = sdf.disc((2, 1), radius)
    result = renderer.render(scene)
    y, x = np.indices((5, 7))
    np.testing.assert_allclose(result.numpy(), np.hypot(x - 2, y - 1) - 2.3)
    assert result.shape == (5, 7)
    result.mean().backward()
    assert radius.grad == pytest.approx(-1.0)
    radius.assign(3.3)
    np.testing.assert_allclose(renderer.render(scene).numpy(), result.numpy() - 1.0)


def test_reusable_expressions_percentages_and_tapes():
    radius = Parameter("radius", 0.4)
    renderer = DifferentiableRenderer((20, 10), {"radius": radius})
    scene = sdf.circle(("50%x", "50%y"), percent_of_min(10) + sin(param("radius")) + 1)
    first = renderer.render(scene)
    first.mean().backward()
    assert radius.grad == pytest.approx(-np.cos(0.4))
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None  # the renderer keeps no output/tape cache
    radius.assign(0.8)
    radius.zero_grad()
    second = renderer.render(scene)
    assert second.numpy()[5, 10] == pytest.approx(-2.0 - np.sin(0.8))
    second.mean().backward()
    assert radius.grad == pytest.approx(-np.cos(0.8))


@pytest.mark.parametrize(
    "operation",
    ["union", "intersection", "subtract", "smooth_union", "interpolate", "abs"],
)
def test_sdf_combinations_match_shader_formulas(operation):
    renderer = DifferentiableRenderer((11, 9))
    first, second = sdf.circle((3.2, 4.1), 2.4), sdf.disc((7, 3), 3.2)
    a, b = renderer.render(first).numpy(), renderer.render(second).numpy()
    if operation == "abs":
        scene, expected = first.abs(), np.abs(a)
    elif operation == "smooth_union":
        scene = first.smooth_union(second, k=2.5)
        weight = np.clip(0.5 + 0.5 * (b - a) / 2.5, 0, 1)
        expected = b * (1 - weight) + a * weight - 2.5 * weight * (1 - weight)
    elif operation == "interpolate":
        scene, expected = first.interpolate(second, t=0.31), a * 0.69 + b * 0.31
    else:
        scene = getattr(first, operation)(second)
        expected = {
            "union": np.minimum(a, b),
            "intersection": np.maximum(a, b),
            "subtract": np.maximum(a, -b),
        }[operation]
    np.testing.assert_allclose(renderer.render(scene).numpy(), expected)


def test_circle_fill_radius_center_color_finite_differences():
    parameters = {
        "radius": Parameter("radius", 3.17),
        "center": Parameter("center", [7.23, 5.61]),
        "foreground": Parameter("foreground", [0.67, 0.31, 0.48, 0.83]),
    }
    scene = (
        sdf.circle(param("center"), param("radius"))
        .fill(param("foreground"), (0.13, 0.17, 0.21, 0.4))
        .to_rgb()
    )
    _check_gradients(scene, parameters)


def test_rotated_rounded_rectangle_finite_differences():
    parameters = {
        "center": Parameter("center", [7.23, 5.61]),
        "size": Parameter("size", [3.13, 2.27]),
        "corner": Parameter("corner", [0.7, 0.9, 0.5, 1.2]),
        "angle": Parameter("angle", 0.21),
    }
    scene = (
        sdf.rounded_rect(
            param("center"), param("size"), param("corner"), param("angle")
        )
        .fill((0.6, 0.4, 0.3, 1))
        .to_rgb()
    )
    _check_gradients(scene, parameters)


@pytest.mark.parametrize("operation", ["translate", "scale", "rotate"])
def test_bilinear_transform_source_and_coordinate_gradients(operation):
    parameters = {"radius": Parameter("radius", 3.17)}
    source = sdf.circle((7.23, 5.61), param("radius"))
    if operation == "translate":
        parameters["offset"] = Parameter("offset", [0.37, -0.21])
        scene = source.translate(param("offset"))
    elif operation == "scale":
        parameters["factor"] = Parameter("factor", 1.17)
        parameters["center"] = Parameter("center", [6.13, 5.27])
        scene = source.scale(param("factor"), center=param("center"))
    else:
        parameters["angle"] = Parameter("angle", 0.17)
        parameters["center"] = Parameter("center", [6.13, 5.27])
        scene = source.rotate(param("angle"), center=param("center"))
    _check_gradients(scene, parameters)


def test_translate_clamps_border_without_scaling_distances():
    renderer = DifferentiableRenderer((5, 4))
    source = sdf.circle((2, 2), 1)
    original = renderer.render(source).numpy()
    shifted = renderer.render(source.translate((1, 0))).numpy()
    np.testing.assert_allclose(shifted[:, 1:], original[:, :-1])
    np.testing.assert_allclose(shifted[:, 0], original[:, 0])
    assert np.max(np.abs(shifted)) > 1.0


def test_lab_colors_conversion_alpha_and_byte_color_shorthand():
    renderer = DifferentiableRenderer((5, 3))
    lab = renderer.render(color.clear("#000000")).numpy()
    np.testing.assert_allclose(lab[..., :3], np.broadcast_to([0, 0.5, 0.5], (3, 5, 3)))
    expected = np.array((0.2, 0.4, 0.6, 0.8))
    for initial in [(51, 102, 153, 204), "#369c", expected]:
        actual = renderer.render(color.clear_color(initial).to_rgb()).numpy()
        np.testing.assert_allclose(
            actual, np.broadcast_to(expected, actual.shape), atol=4e-5
        )
    # Conversion preserves alpha and has a usable gradient through both spaces.
    rgba = Parameter("rgba", [0.24, 0.37, 0.62, 0.73])
    scene = color.clear(param("rgba")).to_rgb().to_lab().to_rgb()
    _check_gradients(scene, {"rgba": rgba})


def test_color_layers_use_rgb_and_straight_alpha():
    renderer = DifferentiableRenderer((4, 3))
    base = color.clear((0.2, 0.4, 0.7, 0.5))
    top = color.clear((0.8, 0.6, 0.3, 0.25)).transparency(0.5)
    b = renderer.render(base.to_rgb()).numpy()
    t = renderer.render(top.to_rgb()).numpy()
    overlay = renderer.render(base.alpha_overlay(top)).numpy()
    alpha = t[..., 3:4] + b[..., 3:4] * (1 - t[..., 3:4])
    expected = (
        t[..., :3] * t[..., 3:4] + b[..., :3] * b[..., 3:4] * (1 - t[..., 3:4])
    ) / alpha
    np.testing.assert_allclose(overlay[..., :3], expected)
    np.testing.assert_allclose(overlay[..., 3:4], alpha)
    np.testing.assert_allclose(renderer.render(base.multiply(top)).numpy(), b * t)
    transparent = color.clear((0.9, 0.8, 0.2, 0.0))
    empty = renderer.render(transparent.alpha_overlay(transparent)).numpy()
    np.testing.assert_array_equal(empty, 0.0)


def test_color_layers_and_fill_controls_finite_differences():
    parameters = {
        "foreground": Parameter("foreground", [0.6, 0.4, 0.3, 0.7]),
        "alpha": Parameter("alpha", 0.63),
        "inflate": Parameter("inflate", 0.12),
        "inner": Parameter("inner", -1.73),
        "outer": Parameter("outer", 0.17),
        "mix": Parameter("mix", 0.23),
        "smoothness": Parameter("smoothness", 1.9),
    }
    a, b = sdf.circle((7.23, 5.61), 3.17), sdf.rect((4.13, 6.27), (2.3, 1.7))
    shape = a.smooth_union(b, param("smoothness")).interpolate(b, param("mix"))
    image = shape.fill(
        param("foreground"),
        (0.2, 0.3, 0.4, 0.3),
        param("inflate"),
        param("inner"),
        param("outer"),
    )
    scene = (
        color.clear((0.4, 0.3, 0.6, 0.7))
        .alpha_overlay(image.transparency(param("alpha")))
        .multiply(color.clear((0.6, 0.8, 0.5, 1)))
    )
    _check_gradients(scene, parameters)


def test_generate_mask_uses_existing_fill_semantics():
    radius = Parameter("radius", 3.17)
    renderer = DifferentiableRenderer((15, 13))
    shape = sdf.circle((7.23, 5.61), radius)
    mask = renderer.render(shape.generate_mask())
    explicit = renderer.render(shape.fill((1, 1, 1, 1), (0, 0, 0, 1)).to_rgb())
    np.testing.assert_allclose(mask.numpy(), explicit.numpy())
    mask.mean().backward()
    assert radius.grad > 0


def test_float_image_target_optimization_decreases_loss():
    parameters = {
        "radius": Parameter("radius", 3.1),
        "center": Parameter("center", [7.1, 9.2]),
        "foreground": Parameter("foreground", [0.5, 0.6, 0.3, 0.9]),
    }
    renderer = DifferentiableRenderer((18, 16), parameters)
    background = (0.1, 0.12, 0.15, 1)
    target = renderer.render(
        sdf.circle((9.1, 7.2), 5.1).fill((0.8, 0.3, 0.6, 0.9), background).to_rgb()
    ).numpy()
    scene = (
        sdf.circle(param("center"), param("radius"))
        .fill(param("foreground"), background)
        .to_rgb()
    )
    optimizer = Adam(parameters.values(), lr=0.08)
    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        difference = renderer.render(scene) - target
        loss = (difference * difference).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.02
    assert parameters["radius"].item() == pytest.approx(5.1, abs=0.15)
    np.testing.assert_allclose(parameters["center"].numpy(), [9.1, 7.2], atol=0.15)


def test_plugin_hook_defaults_modes_memo_and_fresh_render(monkeypatch):
    calls = []

    def hook(renderer, inputs, params):
        calls.append(params["gain"])
        return inputs[0] * params["gain"]

    plugin = Plugin(
        "test_learning_gain",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        (TextureKind.SDF,),
        params=("gain",),
        defaults={"gain": 2.0},
        differentiable_func=hook,
    )
    registry.ensure_loaded()
    monkeypatch.setitem(registry._plugins, plugin.name, plugin)
    radius = Parameter("radius", 2.3)
    child = registry.build(plugin.name, sdf.circle((3.2, 2.4), radius))
    child.should_cache = False  # DAG sharing also applies to uncached GPU nodes
    scene = child.interpolate(child, 0.5)
    renderer = DifferentiableRenderer((6, 5))
    first = renderer.render(scene)
    assert calls == [2.0]
    first.mean().backward()
    assert radius.grad == pytest.approx(-2.0)
    radius.assign(3.3)
    second = renderer.render(scene)
    assert calls == [2.0, 2.0]
    np.testing.assert_allclose(second.numpy(), first.numpy() - 2.0)

    rgb = Plugin(
        "test_learning_rgb",
        PluginFamily.SHADING,
        TextureKind.COLOR,
        mode="RGB",
        differentiable_func=lambda renderer, inputs, params: Tensor(
            np.broadcast_to([0.3, 0.5, 0.7, 1.0], (5, 6, 4))
        ),
    )
    monkeypatch.setitem(registry._plugins, rgb.name, rgb)
    rgb_node = registry.build(rgb.name)
    inherited = ColorTexture(
        op="transparency", inputs=(rgb_node,), params={"alpha": 0.5}
    )
    multiplied = renderer.render(inherited.multiply(rgb_node)).numpy()
    np.testing.assert_allclose(multiplied[0, 0], [0.09, 0.25, 0.49, 0.5])


def test_supported_operations_are_sorted_and_discoverable_without_a_renderer():
    operations = DifferentiableRenderer.supported_operations()
    assert isinstance(operations, tuple)
    assert operations == tuple(sorted(set(operations)))
    assert {
        "circle",
        "disc",
        "rect",
        "rounded_rect",
        "fill",
        "to_rgb",
        "to_lab",
        "translate",
        "scale",
        "rotate",
        "generate_mask",
    } <= set(operations)
    assert {"blur", "ring", "bezier", "light", "masked_union"} <= set(operations)
    assert "skew" in operations


def test_supported_operations_reflect_current_custom_hooks_without_running_them(
    monkeypatch,
):
    calls = []

    def hook(*args):
        calls.append(True)
        raise RuntimeError("Discovery must not execute this hook")

    renderer = DifferentiableRenderer((4, 4))
    before = renderer.supported_operations()
    custom = Plugin(
        "test_discover_learning_hook",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        differentiable_func=hook,
    )
    gpu_only = Plugin(
        "test_discover_gpu_only",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        render_func=hook,
    )
    monkeypatch.setitem(registry._plugins, custom.name, custom)
    monkeypatch.setitem(registry._plugins, gpu_only.name, gpu_only)
    # Non-public hooks must be visible, and GPU render functions are insufficient.
    assert not custom.public
    assert renderer.supported_operations() == tuple(sorted((*before, custom.name)))
    monkeypatch.delitem(registry._plugins, custom.name)
    assert renderer.supported_operations() == before
    assert calls == []


def test_unsupported_graph_is_rejected_before_any_hook_or_gpu_call(monkeypatch):
    calls = []
    plugin = Plugin(
        "test_learning_side_effect",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        differentiable_func=lambda *args: calls.append(True),
    )
    registry.ensure_loaded()
    monkeypatch.setitem(registry._plugins, plugin.name, plugin)
    supported = registry.build(plugin.name)
    unsupported = SDFTexture(op="gpu_only_without_derivative")
    scene = supported.union(unsupported)
    with pytest.raises(UnsupportedGradientError, match="gpu_only_without_derivative"):
        DifferentiableRenderer((5, 5)).render(scene)
    assert calls == []
    with pytest.raises(UnsupportedGradientError, match="made_up"):
        DifferentiableRenderer((5, 5)).render(SDFTexture(op="made_up"))


def test_unsupported_parameter_graph_preflight(monkeypatch):
    calls = []
    plugin = Plugin(
        "test_learning_nested",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        params=("extra",),
        differentiable_func=lambda *args: calls.append(1),
    )
    registry.ensure_loaded()
    monkeypatch.setitem(registry._plugins, plugin.name, plugin)
    scene = registry.build(plugin.name, param("nested"))
    renderer = DifferentiableRenderer(
        (5, 5), {"nested": SDFTexture(op="gpu_only_without_derivative")}
    )
    with pytest.raises(UnsupportedGradientError, match="gpu_only_without_derivative"):
        renderer.render(scene)
    assert calls == []


def test_bad_hook_fails_explicitly(monkeypatch):
    registry.ensure_loaded()
    plugin = Plugin(
        "test_bad_learning_hook",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        differentiable_func=lambda *args: np.zeros((4, 4)),
    )
    monkeypatch.setitem(registry._plugins, plugin.name, plugin)
    with pytest.raises(TypeError, match="must return a Tensor"):
        DifferentiableRenderer((4, 4)).render(registry.build(plugin.name))


def test_save_converts_lab_to_rgb_flips_and_preserves_gradients(tmp_path):
    rgba = Parameter("rgba", [0.2, 0.4, 0.7, 0.8])
    renderer = DifferentiableRenderer((9, 7))
    scene = sdf.circle((4, 1), 2.4).fill(rgba, (0, 0, 0, 0))
    result = renderer.render(scene)
    result.mean().backward()
    previous = rgba.grad.copy()
    path = tmp_path / "scene.png"
    assert renderer.save(scene, path) == path
    pixels = np.asarray(Image.open(path))
    rgb = renderer.render(scene.to_rgb()).numpy()
    expected = np.flipud(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8))
    np.testing.assert_array_equal(pixels, expected)
    np.testing.assert_array_equal(rgba.grad, previous)
    assert pixels[5, 4, 3] > pixels[1, 4, 3]
    with pytest.raises(ValueError, match="color scene"):
        renderer.save(sdf.circle((1, 1), 1), path)


@pytest.fixture
def gl_canvas():
    try:
        canvas = Canvas((21, 17))
    except Exception as exc:
        pytest.skip(f"OpenGL context unavailable: {exc}")
    with canvas:
        yield canvas


@pytest.mark.parametrize(
    "scene",
    [
        sdf.circle((8.2, 7.3), 4.1),
        sdf.rounded_rect((10.2, 8.3), (4.1, 3.2), (0.8, 1.2, 0.5, 1.5), 0.31),
        sdf.circle((8.2, 7.3), 4.1).translate((1.3, -0.7)),
        sdf.circle((8.2, 7.3), 4.1).scale(1.17, (9.1, 7.2)),
        sdf.rect((8.2, 7.3), (4.1, 3.2), 0.5).rotate(0.23, (9.1, 7.2)),
    ],
)
def test_sdf_and_transform_gpu_parity(gl_canvas, scene):
    floating = DifferentiableRenderer(gl_canvas.size).render(scene).numpy()
    gpu = gl_canvas.render(scene)
    values = np.frombuffer(gpu.tex.read(), dtype=np.float32).reshape(floating.shape)
    np.testing.assert_allclose(floating, values, atol=0.04, rtol=0.002)


@pytest.mark.parametrize(
    "scene",
    [
        color.clear((0.2, 0.4, 0.6, 0.8)).to_rgb(),
        sdf.circle((8.2, 7.3), 4.1)
        .fill((0.7, 0.3, 0.5, 0.8), (0.2, 0.3, 0.4, 0.6))
        .to_rgb(),
        color.clear((0.2, 0.4, 0.6, 0.8)).alpha_overlay(
            color.clear((0.6, 0.3, 0.4, 0.3))
        ),
        color.clear((0.2, 0.4, 0.6, 0.8)).multiply(color.clear((0.6, 0.3, 0.4, 0.3))),
    ],
)
def test_color_gpu_parity_with_rgba8_quantization_tolerance(gl_canvas, scene):
    floating = DifferentiableRenderer(gl_canvas.size).render(scene).numpy()
    gpu = gl_canvas.render(scene)
    values = (
        np.frombuffer(gpu.tex.read(), dtype=np.uint8).reshape(floating.shape) / 255.0
    )
    np.testing.assert_allclose(floating, values, atol=0.035)


@pytest.mark.parametrize("size", [(0, 3), (3, -1), (2.5, 3), (True, 3)])
def test_invalid_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        DifferentiableRenderer(size)


@pytest.mark.parametrize("factor", [0.0, 1e-14])
def test_scale_rejects_singular_factor(factor):
    with pytest.raises(ValueError, match="finite and non-zero"):
        DifferentiableRenderer((5, 5)).render(sdf.circle((2, 2), 1).scale(factor))
