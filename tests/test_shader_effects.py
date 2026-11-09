"""Raw shader values, piecewise input/uniform gradients, and direct GL parity."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui.core.plugins.base import PluginFamily
from sdf_ui.core.plugins.registry import registry
from sdf_ui.learning.autodiff import Parameter, Tensor
from sdf_ui.learning.shader_effects import DEFAULTS, RULES

NAMES = (
    "overlay",
    "layer_mask",
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


def _context(size):
    y, x = np.indices((size[1], size[0]), dtype=float)
    return SimpleNamespace(size=size, x=Tensor(x), y=Tensor(y))


def _case(name, size=(3, 2)):
    rng = np.random.default_rng(826)
    uniforms = {
        "overlay": ("tex0", "tex1"),
        "layer_mask": ("tex0", "tex1", "mask"),
        "multiply": ("mask1", "mask2"),
        "transparency": ("tex0",),
    }.get(name, ("origTex",))
    inputs = {key: rng.uniform(0.17, 0.83, (size[1], size[0], 4)) for key in uniforms}
    if name == "to_rgb":
        inputs["origTex"][..., 1:3] = rng.uniform(0.45, 0.55, (size[1], size[0], 2))
    params = {"alpha": 0.37} if name == "transparency" else {}
    if name in BLURS:
        count = len(DEFAULTS[name]["weight"])
        params = {
            "offset": np.array([0.37, 1.37, -2.37, 5.37, -7.37, 2.37, 0.37, 4.37])[
                :count
            ],
            "weight": np.array(DEFAULTS[name]["weight"]),
        }
    return _context(size), inputs, params


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
                value = value + load(abs(x - dx), abs(y - dy)) * weight[i]
                if (x if axis == 1 else y) + i >= (width if axis == 1 else height) - 1:
                    factor = -1
                value = value + load(x + factor * dx, y + factor * dy) * weight[i]
            result[y, x] = value
    return result


def test_rules_cover_actual_standalone_descriptors():
    registry.ensure_loaded()
    descriptors = set()
    for plugin in registry._plugins.values():
        if plugin.family in (PluginFamily.LAYER, PluginFamily.POSTPROCESSING):
            if plugin.shader is not None:
                descriptors.add(plugin.shader.name)
            descriptors.update(shader.name for shader in plugin.extra_shaders)
    assert descriptors == set(NAMES) == set(RULES)
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
    seed = np.random.default_rng(94).uniform(-0.9, 1.3, result.shape)
    ((result * seed) + result * result * 0.031).sum().backward()

    def objective():
        value = _run(name, context, inputs, params).numpy()
        return np.sum(value * seed + value * value * 0.031)

    for source, tensors in ((inputs, leaves), (params, uniforms)):
        for key, tensor in tensors.items():
            assert tensor.grad is not None, (name, key)
            original = np.array(source[key], dtype=float)
            numerical = np.zeros_like(original)
            for index in np.ndindex(original.shape):
                plus, minus = original.copy(), original.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                source[key] = plus
                high = objective()
                source[key] = minus
                low = objective()
                source[key] = original
                numerical[index] = (high - low) / 2e-6
            np.testing.assert_allclose(
                tensor.grad, numerical, atol=3e-7, rtol=3e-6, err_msg=name + ":" + key
            )


def test_layer_values_use_raw_channels_and_glsl_image_uniform_order():
    context = _context((2, 1))
    first = np.array([[[0.2, 0.8, 1.3, 0.25], [-0.2, 0.6, 0.1, 0.6]]])
    second = np.array([[[0.9, 0.3, 0.4, 0.7], [0.5, 0.2, 0.8, 0.8]]])
    alpha = first[..., 3:4] + (1 - first[..., 3:4]) * second[..., 3:4]
    rgb = (
        first[..., :3] * first[..., 3:4]
        + second[..., :3] * (1 - first[..., 3:4]) * second[..., 3:4]
    ) / alpha
    np.testing.assert_allclose(
        _run("overlay", context, {"tex0": first, "tex1": second}, {}).numpy(),
        np.concatenate((rgb, alpha), axis=-1),
    )
    mask = np.array([[[0.3, 0.99, 0.8, 0.7], [1.3, 0.1, 0.2, 0.3]]])
    np.testing.assert_allclose(
        _run(
            "layer_mask", context, {"tex0": first, "tex1": second, "mask": mask}, {}
        ).numpy(),
        second * (1 - mask[..., :1]) + first * mask[..., :1],
    )
    np.testing.assert_allclose(
        _run("multiply", context, {"mask1": first, "mask2": second}, {}).numpy(),
        first * second,
    )
    expected = first.copy()
    expected[..., 3] *= -0.4
    np.testing.assert_allclose(
        _run("transparency", context, {"tex0": first}, {"alpha": -0.4}).numpy(),
        expected,
    )
    expected[..., :3] = 1 - first[..., :3]
    expected[..., 3] = first[..., 3]
    np.testing.assert_allclose(
        _run("invert", context, {"origTex": first}, {}).numpy(), expected
    )


def test_overlay_empty_branch_has_finite_gradients():
    context = _context((2, 1))
    top = Tensor([[[0.7, 0.3, 0.2, 0.0], [0.4, 0.7, 0.3, 0.0]]], requires_grad=True)
    base = Tensor([[[0.2, 0.1, 0.5, 0.0], [0.1, 0.5, 0.3, 0.0]]], requires_grad=True)
    value = RULES["overlay"](context, {"tex0": top, "tex1": base}, {})
    np.testing.assert_array_equal(value.numpy(), 0.0)
    value.sum().backward()
    for tensor in (top, base):
        np.testing.assert_array_equal(tensor.grad[..., :3], 0.0)
        np.testing.assert_array_equal(tensor.grad[..., 3], 1.0)


def test_dithering_threshold_orientation_equality_period_and_identity_gradient():
    context = _context((10, 9))
    # [x=1][y=0] is 48, [x=0][y=1] is 32 in GLSL's array.
    image = np.full((9, 10, 4), 0.6)
    image[0, 0, :3] = 1 / 64  # equality takes the >= branch
    image[..., 3] = 0.27
    color = Tensor(image, requires_grad=True)
    value = RULES["dithering"](context, {"origTex": color}, {})
    np.testing.assert_allclose(value.numpy()[0, 0, :3], 1 / 64 + 1 / 255)
    np.testing.assert_allclose(value.numpy()[0, 1, :3], 0.6)
    np.testing.assert_allclose(value.numpy()[1, 0, :3], 0.6 + 1 / 255)
    np.testing.assert_allclose(value.numpy()[1, 1, :3], 0.6 + 1 / 255)
    np.testing.assert_array_equal(value.numpy()[0, 1], value.numpy()[8, 9])
    np.testing.assert_array_equal(value.numpy()[..., 3], image[..., 3])
    value.sum().backward()
    np.testing.assert_array_equal(color.grad, 1.0)


def test_1bit_uses_luminance_constant_alpha_and_connected_true_zero_gradient():
    context = _context((3, 1))
    image = Tensor(
        [[[0.0, 0.0, 1.0, 0.2], [1.0, 0.0, 0.0, 0.7], [0.0, 1.0, 0.0, 0.1]]],
        requires_grad=True,
    )
    value = RULES["dither_1bit"](context, {"origTex": image}, {})
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
            inputs["origTex"], effective["offset"], effective["weight"], axis
        )
        np.testing.assert_allclose(
            _run(name, context, inputs, uniforms).numpy(), expected, atol=1e-15
        )


def test_blur_asymmetry_is_not_symmetric_reflect_padding_and_counts_repeated_gathers():
    context = _context((6, 1))
    image = Tensor(np.broadcast_to(np.arange(6)[None, :, None], (1, 6, 4)), True)
    offset = Tensor([9.3, 1.7, 0.3, 0.3, 0.3], True)
    weight = Tensor([0, 1, 0, 0, 0], True)
    value = RULES["blur_hor_9"](
        context, {"origTex": image}, {"offset": offset, "weight": weight}
    )
    np.testing.assert_array_equal(value.numpy()[0, :, 0], [2, 2, 4, 6, 6, 8])
    value[0, 0, 0].backward()
    np.testing.assert_array_equal(image.grad[0, :, 0], [0, 2, 0, 0, 0, 0])
    np.testing.assert_array_equal(image.grad[..., 1:], 0.0)
    np.testing.assert_array_equal(offset.grad, 0.0)
    np.testing.assert_array_equal(weight.grad, [0, 2, 0, 0, 0])


@pytest.mark.parametrize("name", BLURS)
def test_blur_uses_input_extent_for_borders_and_zero_for_oob_center(name):
    context = _context((7, 6))
    source = np.arange(24, dtype=float).reshape(2, 3, 4)
    defaults = DEFAULTS[name]
    expected = _blur_reference(
        source,
        defaults["offset"],
        defaults["weight"],
        1 if "hor" in name else 0,
        output_size=context.size,
    )
    np.testing.assert_allclose(
        _run(name, context, {"origTex": source}, {}).numpy(), expected
    )


def test_uniform_tensor_containers_preserve_weight_gradients_and_versions():
    context, inputs, _ = _case("blur_hor_9")
    gain = Parameter("gain", 0.37)
    value = _run("blur_hor_9", context, inputs, {"weight": [gain, 0, 0, 0, 0]})
    value.sum().backward()
    assert gain.grad == pytest.approx(inputs["origTex"].sum())
    gain.assign(0.4)
    with pytest.raises(RuntimeError, match="stale"):
        value.sum().backward()


@pytest.mark.parametrize("name", BLURS)
def test_blur_rejects_wrong_uniform_array_shapes(name):
    context, inputs, _ = _case(name)
    for key in ("offset", "weight"):
        with pytest.raises(ValueError, match="offsets and weights"):
            _run(name, context, inputs, {key: [0.2]})


@pytest.fixture(scope="module")
def gl_context():
    import moderngl

    try:
        context = moderngl.create_standalone_context(require=430, backend="egl")
    except Exception as exc:
        pytest.skip("OpenGL 4.3 EGL context unavailable: {}".format(exc))
    try:
        yield context
    finally:
        context.release()


def _gl_run(context, name, size, inputs, params, *, floating):
    descriptor = next(item for item in registry.shader_files() if item.name == name)
    source = (Path(__file__).parents[1] / "src/sdf_ui" / descriptor.path).read_text()
    # Float image formats isolate arithmetic parity from rgba8 storage effects.
    if floating:
        source = source.replace("rgba8", "rgba32f")
    shader = context.compute_shader(source)
    dtype = "f4" if floating else "f1"
    textures = []
    try:
        dest = context.texture(size, 4, dtype=dtype)
        textures.append(dest)
        dest.bind_to_image(0, read=False, write=True)
        shader["destTex"].value = 0
        for binding, (key, value) in enumerate(inputs.items(), start=1):
            payload = (
                value.astype(np.float32)
                if floating
                else np.rint(value * 255).astype(np.uint8)
            )
            tex = context.texture(size, 4, data=payload.tobytes(), dtype=dtype)
            textures.append(tex)
            tex.bind_to_image(binding, read=True, write=False)
            shader[key].value = binding
        for key, value in params.items():
            shader[key].value = tuple(value) if np.ndim(value) else float(value)
        shader.run((size[0] + 15) // 16, (size[1] + 15) // 16, 1)
        context.memory_barrier()
        array = np.frombuffer(dest.read(), dtype=np.float32 if floating else np.uint8)
        return array.reshape(size[1], size[0], 4).astype(float) / (
            1 if floating else 255
        )
    finally:
        for texture in reversed(textures):
            texture.release()
        shader.release()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "floating", [False, True], ids=["raw-rgba8", "float-arithmetic"]
)
def test_direct_glsl_parity_without_plugin_host_paths(gl_context, name, floating):
    size = (19, 17)  # partial compute workgroups exercise shader bounds guards
    context, inputs, _ = _case(name, size)
    if not floating:
        inputs = {key: np.rint(value * 255) / 255 for key, value in inputs.items()}
    params = {"alpha": 0.37} if name == "transparency" else {}
    expected = _run(name, context, inputs, params).numpy()
    actual = _gl_run(gl_context, name, size, inputs, params, floating=floating)
    if not floating:
        expected = np.clip(expected, 0, 1)
    np.testing.assert_allclose(actual, expected, atol=3e-5 if floating else 1.01 / 255)


@pytest.mark.parametrize("name", BLURS)
def test_direct_glsl_fractional_negative_blur_offsets_at_valid_loads(gl_context, name):
    size = (23, 21)
    context, inputs, params = _case(name, size)
    # Custom offsets exercise integer casts and an i-based sign flip with all
    # sampled coordinates valid in this crop. CPU-only tests cover zero OOB.
    expected = _run(name, context, inputs, params).numpy()
    actual = _gl_run(gl_context, name, size, inputs, params, floating=True)
    np.testing.assert_allclose(actual[8:-8, 8:-8], expected[8:-8, 8:-8], atol=3e-6)
