"""Raw shader values and piecewise input/uniform gradients."""

import numpy as np
import pytest

from sdf_ui.gpu.nodes.base import PluginFamily
from sdf_ui.gpu.nodes.registry import registry
from sdf_ui.learning.autodiff import Parameter, Tensor
from sdf_ui.learning.shaders.effects import DEFAULTS, RULES
from tests.helpers import shader_context

_LAYER_MASK_TEXT = "layer_mask"
_FINITE_DIFFERENCE_STEP = 2e-6

_OFFSET_TEXT = "offset"

_ORIGTEX_TEXT = "origTex"

_WEIGHT_TEXT = "weight"

NAMES = (
    "overlay",
    _LAYER_MASK_TEXT,
    "multiply",
    "transparency",
    "invert",
    "to_rgb",
    "to_lab",
    "dithering",
    "dither_1bit",
    "blur_hor_9",
    "blur_ver_9",
    "blur_hor_13",
    "blur_ver_13",
)
BLURS = NAMES[-4:]


def _case(name, size=(3, 2)):
    rng = np.random.default_rng(826)
    uniforms = {
        "overlay": ("tex0", "tex1"),
        _LAYER_MASK_TEXT: ("tex0", "tex1", "mask"),
        "multiply": ("mask1", "mask2"),
        "transparency": ("tex0",),
    }.get(name, (_ORIGTEX_TEXT,))
    inputs = {key: rng.uniform(0.17, 0.83, (size[1], size[0], 4)) for key in uniforms}
    if name == "to_rgb":
        inputs[_ORIGTEX_TEXT][..., 1:3] = rng.uniform(0.45, 0.55, (size[1], size[0], 2))
    params = {"alpha": 0.37} if name == "transparency" else {}
    if name in BLURS:
        count = len(DEFAULTS[name][_WEIGHT_TEXT])
        params = {
            _OFFSET_TEXT: np.array(
                [0.37, 1.37, -237 / 100, 5.37, -737 / 100, 2.37, 0.37, 4.37]
            )[:count],
            _WEIGHT_TEXT: np.array(DEFAULTS[name][_WEIGHT_TEXT]),
        }
    return shader_context(size), inputs, params


def _run(name, context, inputs, params):
    return RULES[name](
        context,
        {key: Tensor(value) for key, value in inputs.items()},
        params,
    )


def _blur_reference(image, offset, weight, axis, output_size=None):
    """Scalar GLSL transcription, independent of vectorized gather code."""
    height, width = image.shape[:2]
    output_size = (width, height) if output_size is None else output_size
    result = np.zeros((output_size[1], output_size[0], 4))

    def load(x, y):
        if 0 <= x < width and 0 <= y < height:
            return image[y, x]
        return np.zeros(4)

    for y in range(output_size[1]):
        for x in range(output_size[0]):
            value = load(x, y) * weight[0]
            factor = 1
            for i in range(1, len(weight)):
                dx, dy = (int(offset[i]), 0) if axis == 1 else (0, int(offset[i]))
                value, factor = _blur_pair(
                    image,
                    value,
                    x,
                    y,
                    i,
                    dx,
                    dy,
                    weight[i],
                    axis,
                    width,
                    height,
                    factor,
                )
            result[y, x] = value
    return result


def _blur_pair(image, value, x, y, index, dx, dy, weight, axis, width, height, factor):
    left_x, left_y = abs(x - dx), abs(y - dy)
    if 0 <= left_x < width and 0 <= left_y < height:
        sample = image[left_y, left_x]
    else:
        sample = np.zeros(4)
    value = value + sample * weight
    coordinate = x if axis == 1 else y
    extent = width if axis == 1 else height
    if coordinate + index >= extent - 1:
        factor = -1
    right_x, right_y = x + factor * dx, y + factor * dy
    if 0 <= right_x < width and 0 <= right_y < height:
        sample = image[right_y, right_x]
    else:
        sample = np.zeros(4)
    value = value + sample * weight
    return value, factor


def test_rules_cover_actual_standalone_descriptors():
    registry.ensure_loaded()
    descriptors = {
        shader.name: shader
        for plugin in registry._plugins.values()
        if plugin.family in (PluginFamily.LAYER, PluginFamily.POSTPROCESSING)
        for shader in (plugin.shader, *plugin.extra_shaders)
        if shader is not None
    }
    assert (
        {name for name, shader in descriptors.items() if shader.wgsl is None}
        == set(NAMES)
        == set(RULES)
    )
    assert all(
        callable(shader.differentiation.reference) for shader in descriptors.values()
    )
    assert set(DEFAULTS) == set(BLURS)
    assert all(callable(rule) for rule in RULES.values())


@pytest.mark.parametrize("name", NAMES)
def test_every_input_and_uniform_gradient_matches_finite_differences(name):
    context, inputs, params = _case(name)
    leaves = {key: Tensor(value, requires_grad=True) for key, value in inputs.items()}
    uniforms = {key: Tensor(value, requires_grad=True) for key, value in params.items()}
    result = RULES[name](context, leaves, uniforms)
    assert isinstance(result, Tensor)
    assert result.shape == (2, 3, 4)
    seed = np.random.default_rng(94).uniform(-9 / 10, 1.3, result.shape)
    ((result * seed) + result * result * 31 / 1_000).sum().backward()

    def objective():
        value = _run(name, context, inputs, params).numpy()
        return np.sum(value * seed + value * value * 31 / 1_000)

    _assert_effect_gradients(name, inputs, params, leaves, uniforms, objective)


def _assert_effect_gradients(name, inputs, params, leaves, uniforms, objective):
    for source, tensors in ((inputs, leaves), (params, uniforms)):
        for key, tensor in tensors.items():
            assert tensor.grad is not None, (name, key)
            original = np.array(source[key], dtype=float)
            numerical = np.zeros_like(original)
            for index in np.ndindex(original.shape):
                plus, minus = original.copy(), original.copy()
                plus[index] += 1 / 1_000_000
                minus[index] -= 1 / 1_000_000
                source[key] = plus
                high = objective()
                source[key] = minus
                low = objective()
                source[key] = original
                numerical[index] = (high - low) / _FINITE_DIFFERENCE_STEP
            np.testing.assert_allclose(
                tensor.grad, numerical, atol=3e-7, rtol=3e-6, err_msg=name + ":" + key
            )


def test_layer_values_use_raw_channels_and_glsl_image_uniform_order():
    context = shader_context((2, 1))
    first = np.array([[[0.2, 0.8, 1.3, 0.25], [-1 / 5, 0.6, 0.1, 0.6]]])
    second = np.array([[[0.9, 0.3, 2 / 5, 0.7], [0.5, 0.2, 0.8, 0.8]]])
    _assert_overlay_blend(context, first, second)
    mask = np.array([[[0.3, 0.99, 0.8, 0.7], [1.3, 0.1, 0.2, 0.3]]])
    _assert_mask_blends(context, first, second, mask)
    _assert_mask_modes(context, first, second, mask)
    _assert_simple_layer_modes(context, first, second)


def _assert_overlay_blend(context, first, second):
    alpha = first[..., 3:4] + (1 - first[..., 3:4]) * second[..., 3:4]
    rgb = (
        first[..., :3] * first[..., 3:4]
        + second[..., :3] * (1 - first[..., 3:4]) * second[..., 3:4]
    ) / alpha
    np.testing.assert_allclose(
        _run("overlay", context, {"tex0": first, "tex1": second}, {}).numpy(),
        np.concatenate((rgb, alpha), axis=-1),
    )


def _assert_mask_blends(context, first, second, mask):
    np.testing.assert_allclose(
        _run(
            _LAYER_MASK_TEXT, context, {"tex0": first, "tex1": second, "mask": mask}, {}
        ).numpy(),
        second * (1 - mask[..., :1]) + first * mask[..., :1],
    )


def _assert_mask_modes(context, first, second, mask):
    multiply_expected = second * (1 - mask[..., :1]) + first * second * mask[..., :1]
    np.testing.assert_allclose(
        _run(
            _LAYER_MASK_TEXT,
            context,
            {"tex0": first, "tex1": second, "mask": mask},
            {"mode": 1.0},
        ).numpy(),
        multiply_expected,
    )
    screen = 1 - (1 - first) * (1 - second)
    np.testing.assert_allclose(
        _run(
            _LAYER_MASK_TEXT,
            context,
            {"tex0": first, "tex1": second, "mask": mask},
            {"mode": 2.0},
        ).numpy(),
        second * (1 - mask[..., :1]) + screen * mask[..., :1],
    )


def _assert_simple_layer_modes(context, first, second):
    np.testing.assert_allclose(
        _run("multiply", context, {"mask1": first, "mask2": second}, {}).numpy(),
        first * second,
    )
    expected = first.copy()
    expected[..., 3] *= -2 / 5
    np.testing.assert_allclose(
        _run("transparency", context, {"tex0": first}, {"alpha": -2 / 5}).numpy(),
        expected,
    )
    expected[..., :3] = 1 - first[..., :3]
    expected[..., 3] = first[..., 3]
    np.testing.assert_allclose(
        _run("invert", context, {_ORIGTEX_TEXT: first}, {}).numpy(), expected
    )


def test_overlay_empty_branch_has_finite_gradients():
    context = shader_context((2, 1))
    top = Tensor([[[0.7, 0.3, 0.2, 0.0], [0.4, 0.7, 0.3, 0.0]]], requires_grad=True)
    base = Tensor([[[0.2, 0.1, 0.5, 0.0], [0.1, 0.5, 0.3, 0.0]]], requires_grad=True)
    value = RULES["overlay"](context, {"tex0": top, "tex1": base}, {})
    np.testing.assert_array_equal(value.numpy(), 0.0)
    value.sum().backward()
    for tensor in (top, base):
        np.testing.assert_array_equal(tensor.grad[..., :3], 0.0)
        np.testing.assert_array_equal(tensor.grad[..., 3], 1.0)


def test_dithering_threshold_orientation_equality_period_and_identity_gradient():
    context = shader_context((10, 9))
    # [x=1][y=0] is 48, [x=0][y=1] is 32 in GLSL's array.
    image = np.full((9, 10, 4), 3 / 5)
    image[0, 0, :3] = 1 / 64  # equality takes the >= branch
    image[..., 3] = 0.27
    color = Tensor(image, requires_grad=True)
    value = RULES["dithering"](context, {_ORIGTEX_TEXT: color}, {})
    np.testing.assert_allclose(value.numpy()[0, 0, :3], 1 / 64 + 1 / 255)
    np.testing.assert_allclose(value.numpy()[0, 1, :3], 3 / 5)
    np.testing.assert_allclose(value.numpy()[1, 0, :3], 3 / 5 + 1 / 255)
    np.testing.assert_allclose(value.numpy()[1, 1, :3], 3 / 5 + 1 / 255)
    np.testing.assert_array_equal(value.numpy()[0, 1], value.numpy()[8, 9])
    np.testing.assert_array_equal(value.numpy()[..., 3], image[..., 3])
    value.sum().backward()
    np.testing.assert_array_equal(color.grad, 1.0)


def test_1bit_uses_luminance_constant_alpha_and_connected_true_zero_gradient():
    context = shader_context((3, 1))
    image = Tensor(
        [[[0.0, 0.0, 1.0, 0.2], [1.0, 0.0, 0.0, 0.7], [0.0, 1.0, 0.0, 0.1]]],
        requires_grad=True,
    )
    value = RULES["dither_1bit"](context, {_ORIGTEX_TEXT: image}, {})
    np.testing.assert_array_equal(
        value.numpy(), [[[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1]]]
    )
    value.sum().backward()
    np.testing.assert_array_equal(image.grad, 0.0)


@pytest.mark.parametrize("name", BLURS)
@pytest.mark.parametrize("size", [(1, 1), (3, 2), (11, 10)])
def test_blur_exact_defaults_and_fractional_negative_oversized_offsets(name, size):
    context, inputs, params = _case(name, size)
    axis = 1 if "hor" in name else 0
    for uniforms in ({}, params):
        effective = {**DEFAULTS[name], **uniforms}
        expected = _blur_reference(
            inputs[_ORIGTEX_TEXT],
            effective[_OFFSET_TEXT],
            effective[_WEIGHT_TEXT],
            axis,
        )
        np.testing.assert_allclose(
            _run(name, context, inputs, uniforms).numpy(), expected, atol=1e-15
        )


def test_blur_asymmetry_is_not_symmetric_reflect_padding_and_counts_repeated_gathers():
    context = shader_context((6, 1))
    image = Tensor(np.broadcast_to(np.arange(6)[None, :, None], (1, 6, 4)), True)
    offset = Tensor([9.3, 1.7, 0.3, 0.3, 0.3], True)
    weight = Tensor([0, 1, 0, 0, 0], True)
    value = RULES["blur_hor_9"](
        context, {_ORIGTEX_TEXT: image}, {_OFFSET_TEXT: offset, _WEIGHT_TEXT: weight}
    )
    np.testing.assert_array_equal(value.numpy()[0, :, 0], [2, 2, 4, 6, 6, 8])
    value[0, 0, 0].backward()
    np.testing.assert_array_equal(image.grad[0, :, 0], [0, 2, 0, 0, 0, 0])
    np.testing.assert_array_equal(image.grad[..., 1:], 0.0)
    np.testing.assert_array_equal(offset.grad, 0.0)
    np.testing.assert_array_equal(weight.grad, [0, 2, 0, 0, 0])


@pytest.mark.parametrize("name", BLURS)
def test_blur_uses_input_extent_for_borders_and_zero_for_oob_center(name):
    context = shader_context((7, 6))
    source = np.arange(24, dtype=float).reshape(2, 3, 4)
    defaults = DEFAULTS[name]
    expected = _blur_reference(
        source,
        defaults[_OFFSET_TEXT],
        defaults[_WEIGHT_TEXT],
        1 if "hor" in name else 0,
        output_size=context.size,
    )
    np.testing.assert_allclose(
        _run(name, context, {_ORIGTEX_TEXT: source}, {}).numpy(), expected
    )


def test_uniform_tensor_containers_preserve_weight_gradients_and_versions():
    context, inputs, _ = _case("blur_hor_9")
    gain = Parameter("gain", 0.37)
    value = _run("blur_hor_9", context, inputs, {_WEIGHT_TEXT: [gain, 0, 0, 0, 0]})
    value.sum().backward()
    assert gain.grad == pytest.approx(inputs[_ORIGTEX_TEXT].sum())
    gain.assign(0.4)
    with pytest.raises(RuntimeError, match="stale"):
        value.sum().backward()


@pytest.mark.parametrize("name", BLURS)
def test_blur_rejects_wrong_uniform_array_shapes(name):
    context, inputs, _ = _case(name)
    for key in (_OFFSET_TEXT, _WEIGHT_TEXT):
        with pytest.raises(ValueError, match="offsets and weights"):
            _run(name, context, inputs, {key: [0.2]})
