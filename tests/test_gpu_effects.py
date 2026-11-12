"""Native layer/postprocess GPU VJPs and the exact reflected blur transpose."""

from contextlib import ExitStack

import numpy as np
import pytest

from sdf_ui import Canvas
from sdf_ui.backends.opengl.backend import GPUShaderBackend
from sdf_ui.backends.opengl.glsl import PRELUDE
from sdf_ui.backends.opengl.nodes.base import PluginFamily
from sdf_ui.backends.opengl.nodes.registry import registry
from sdf_ui.learning import shader_derivatives, shader_effects
from sdf_ui.learning.shader_effects import DEFAULTS
from sdf_ui.backends.opengl.shader_sources.effects import SOURCES

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
SIZE = (5, 3)


def _contract(name):
    return next(s for s in registry.shader_files() if s.name == name).differentiation


@pytest.fixture(scope="module")
def backend():
    try:
        canvas = Canvas(SIZE)
    except Exception as error:
        pytest.skip("OpenGL context unavailable: {}".format(error))
    with canvas:
        if canvas._mgl_ctx.version_code < 430:
            pytest.skip("Native GPU derivatives require OpenGL 4.3")
        with GPUShaderBackend(canvas) as runtime:
            yield runtime


def _case(name, size=SIZE):
    rng = np.random.default_rng(826)
    shape = (size[1], size[0], 4)
    inputs = {
        argument.name: rng.uniform(0.17, 0.83, shape).astype(np.float32)
        for argument in _contract(name).inputs
    }
    if name == "to_rgb":
        inputs["origTex"][..., 1:3] = rng.uniform(0.45, 0.55, shape[:2] + (2,))
    params = {"alpha": np.float32(-0.37)} if name == "transparency" else {}
    if name in BLURS:
        count = len(DEFAULTS[name]["weight"])
        params = {
            "offset": np.array(
                [0.37, 1.37, -2.37, 5.37, -7.37, 2.37, 0.37, 4.37],
                dtype=np.float32,
            )[:count],
            "weight": np.array(DEFAULTS[name]["weight"], dtype=np.float32),
        }
    seed = rng.normal(size=shape).astype(np.float32)
    return inputs, params, seed


def _forward(backend, name, inputs, params):
    with ExitStack() as stack:
        images = {
            key: stack.enter_context(backend.upload(value))
            for key, value in inputs.items()
        }
        outputs = backend.forward(name, images, params)
        for output in outputs:
            stack.enter_context(output)
        assert len(outputs) == 1
        return outputs[0].read()


def _no_cpu_reference(*args, **kwargs):
    pytest.fail("Native effects execution called a CPU derivative rule")


def test_sources_cover_actual_layer_and_postprocessing_descriptors():
    registry.ensure_loaded()
    names = set()
    for plugin in registry._plugins.values():
        if plugin.family in (PluginFamily.LAYER, PluginFamily.POSTPROCESSING):
            if plugin.shader is not None:
                names.add(plugin.shader.name)
            names.update(shader.name for shader in plugin.extra_shaders)
    assert names == set(NAMES) == set(SOURCES)
    assert all(isinstance(source, str) for source in SOURCES.values())


def _dual_values(backend, name, inputs, params):
    """Execute the body's primal values, which backward normally discards."""
    contract = _contract(name)
    size = (next(iter(inputs.values())).shape[1], next(iter(inputs.values())).shape[0])
    declarations, loads, macros = [], [], []
    for index, _ in enumerate(contract.inputs):
        declarations.append(
            "layout(rgba32f,binding=%d) readonly uniform image2D source%d;"
            % (index, index)
        )
        loads.append(
            "if(which==%d) return c(imageLoad(source%d,p)[channel]);" % (index, index)
        )
    for argument in contract.parameters:
        uniform = "param_" + argument.name
        if argument.shape:
            declarations.append("uniform float %s[%d];" % (uniform, argument.shape[0]))
            macros.append("#define U_%s(i) c(%s[i])" % (argument.name, uniform))
        else:
            declarations.append("uniform float %s;" % uniform)
            macros.append("#define U_%s c(%s)" % (argument.name, uniform))
    source = (
        "#version 430\n"
        + PRELUDE
        + "\n"
        + "\n".join(declarations)
        + "\nuniform ivec2 size;\n"
        + "D ld(int which,ivec2 p,int channel) {"
        + "if(any(lessThan(p,ivec2(0)))||any(greaterThanEqual(p,size))) return c(0.0);"
        + "".join(loads)
        + "return c(0.0);}\n"
        + "\n".join(macros)
        + "\n"
        + SOURCES[name]
        + """
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) writeonly buffer Values { float values[]; };
void main() {
    ivec2 pos=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(pos,size))) return;
    D o[8]; evaluate(pos,o);
    int start=(pos.y*size.x+pos.x)*4;
    for(int channel=0;channel<4;channel++) values[start+channel]=o[channel].v;
}
"""
    )
    with ExitStack() as stack:
        program = backend.ctx.compute_shader(source)
        stack.callback(program.release)
        result = backend.ctx.buffer(reserve=size[0] * size[1] * 16)
        stack.callback(result.release)
        result.bind_to_storage_buffer(0)
        for index, argument in enumerate(contract.inputs):
            image = stack.enter_context(backend.upload(inputs[argument.name]))
            image.texture.bind_to_image(index, read=True, write=False)
        program["size"].value = size
        for key, value in backend.uniforms(name, params).items():
            uniform = "param_" + key
            if uniform in program:
                program[uniform].write(value.tobytes())
        program.run((size[0] + 15) // 16, (size[1] + 15) // 16, 1)
        backend.ctx.memory_barrier()
        return np.frombuffer(result.read(), dtype=np.float32).reshape(
            size[1], size[0], 4
        )


@pytest.mark.parametrize("name", NAMES)
def test_dual_body_values_match_original_fp32_glsl(backend, name):
    # This also validates the dual body's fixed Bayer table and threshold
    # orientation: their values do not affect either dither shader's VJP.
    inputs, params, _ = _case(name, (10, 9))
    actual = _dual_values(backend, name, inputs, params)
    original = _forward(backend, name, inputs, params)
    np.testing.assert_allclose(actual, original, atol=4e-6, rtol=2e-5)


@pytest.mark.parametrize("name", NAMES)
def test_native_forward_and_every_vjp_against_reference(backend, monkeypatch, name):
    inputs, params, seed = _case(name)
    contract = _contract(name)
    expected_value = contract.evaluate(SIZE, inputs, params).numpy()
    expected = contract.vjp(SIZE, inputs, params, seed)
    snapshots = {key: value.copy() for key, value in inputs.items()}

    # CPU rules serve only as an independent oracle before native execution.
    monkeypatch.setattr(shader_derivatives, "evaluate_shader", _no_cpu_reference)
    monkeypatch.setattr(shader_derivatives, "shader_vjp", _no_cpu_reference)
    monkeypatch.setitem(shader_effects.RULES, name, _no_cpu_reference)
    before = backend.stats.copy()
    actual_value = _forward(backend, name, inputs, params)
    actual = backend.vjp(name, inputs, params, seed)
    assert backend.stats["forward_dispatches"] == before["forward_dispatches"] + 2
    assert backend.stats["backward_dispatches"] == before["backward_dispatches"] + 1
    np.testing.assert_allclose(actual_value, expected_value, atol=4e-6, rtol=2e-5)
    for field in ("inputs", "parameters"):
        result, reference = getattr(actual, field), getattr(expected, field)
        assert result.keys() == reference.keys()
        for key in reference:
            assert np.all(np.isfinite(result[key])), (name, key)
            np.testing.assert_allclose(
                result[key],
                reference[key],
                atol=2e-5,
                rtol=6e-5,
                err_msg=name + ":" + key,
            )
    for key, snapshot in snapshots.items():
        np.testing.assert_array_equal(inputs[key], snapshot)


def _blur_transpose(image, offsets, weights, seed, axis):
    """Scalar stencil oracle, including repeated/OOB load adjoints."""
    height, width = image.shape[:2]
    value = np.zeros_like(image, dtype=float)
    image_gradient = np.zeros_like(image, dtype=float)
    weight_gradient = np.zeros_like(weights, dtype=float)
    for y in range(height):
        for x in range(width):
            taps = [(x, y, 0)]
            factor = 1
            for i in range(1, len(weights)):
                dx, dy = (int(offsets[i]), 0) if axis == 1 else (0, int(offsets[i]))
                taps.append((abs(x - dx), abs(y - dy), i))
                if (x if axis == 1 else y) + i >= (width if axis == 1 else height) - 1:
                    factor = -1
                taps.append((x + factor * dx, y + factor * dy, i))
            for sx, sy, index in taps:
                if 0 <= sx < width and 0 <= sy < height:
                    value[y, x] += image[sy, sx] * weights[index]
                    image_gradient[sy, sx] += seed[y, x] * weights[index]
                    weight_gradient[index] += np.dot(image[sy, sx], seed[y, x])
    return value, image_gradient, weight_gradient


@pytest.mark.parametrize("name", BLURS)
@pytest.mark.parametrize("size", [(1, 1), (4, 3), (10, 9)])
@pytest.mark.parametrize("defaults", [True, False], ids=["defaults", "custom-offsets"])
def test_blur_scatter_exact_asymmetric_reflection_and_truncation(
    backend,
    name,
    size,
    defaults,
):
    inputs, custom, seed = _case(name, size)
    params = {} if defaults else custom
    effective = {**DEFAULTS[name], **params}
    image = inputs["origTex"]
    expected, image_gradient, weight_gradient = _blur_transpose(
        image,
        effective["offset"],
        effective["weight"],
        seed,
        axis=1 if "hor" in name else 0,
    )
    actual = backend.vjp(name, inputs, params, seed)
    np.testing.assert_allclose(
        _forward(backend, name, inputs, params), expected, atol=2e-6
    )
    np.testing.assert_allclose(actual.inputs["origTex"], image_gradient, atol=3e-6)
    np.testing.assert_allclose(actual.parameters["weight"], weight_gradient, atol=2e-5)
    np.testing.assert_array_equal(actual.parameters["offset"], 0)
    # This inner-product identity checks that reflected duplicates scatter-add.
    np.testing.assert_allclose(
        np.sum(expected * seed),
        np.sum(image * actual.inputs["origTex"]),
        atol=2e-5,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "name", ["overlay", "layer_mask", "multiply", "transparency", "invert"]
)
def test_raw_channels_and_uniform_gradients_against_native_forward(backend, name):
    inputs, params, seed = _case(name)
    rng = np.random.default_rng(1107)
    for image in inputs.values():
        image[..., :3] = rng.uniform(-0.3, 1.4, image[..., :3].shape)
    directions = {key: rng.normal(size=value.shape) for key, value in inputs.items()}
    gradients = backend.vjp(name, inputs, params, seed)
    epsilon = 1e-3
    high = _forward(
        backend,
        name,
        {key: value + epsilon * directions[key] for key, value in inputs.items()},
        params,
    )
    low = _forward(
        backend,
        name,
        {key: value - epsilon * directions[key] for key, value in inputs.items()},
        params,
    )
    numerical = np.sum((high.astype(float) - low) * seed) / (2 * epsilon)
    analytic = sum(np.sum(gradients.inputs[key] * d) for key, d in directions.items())
    np.testing.assert_allclose(analytic, numerical, atol=7e-4, rtol=4e-4)
    if name == "transparency":
        high = _forward(backend, name, inputs, {"alpha": params["alpha"] + epsilon})
        low = _forward(backend, name, inputs, {"alpha": params["alpha"] - epsilon})
        numerical = np.sum((high.astype(float) - low) * seed) / (2 * epsilon)
        np.testing.assert_allclose(gradients.parameters["alpha"], numerical, atol=3e-4)


@pytest.mark.parametrize("alpha", [0.0, -0.4])
def test_overlay_nonpositive_alpha_branch_is_finite_and_retains_alpha_vjp(
    backend, alpha
):
    inputs, _, seed = _case("overlay")
    inputs["tex0"][..., 3] = alpha
    inputs["tex1"][..., 3] = 0
    value = _forward(backend, "overlay", inputs, {})
    actual = backend.vjp("overlay", inputs, {}, seed)
    np.testing.assert_array_equal(value[..., :3], 0)
    np.testing.assert_array_equal(value[..., 3], np.float32(alpha))
    for key in inputs:
        np.testing.assert_array_equal(actual.inputs[key][..., :3], 0)
    np.testing.assert_allclose(actual.inputs["tex0"][..., 3], seed[..., 3])
    np.testing.assert_allclose(
        actual.inputs["tex1"][..., 3], (1 - alpha) * seed[..., 3]
    )


@pytest.mark.parametrize("name", ["dithering", "dither_1bit"])
def test_dither_fixed_bayer_orientation_equality_and_true_derivatives(backend, name):
    size = (10, 9)
    image = np.full((size[1], size[0], 4), 0.6, dtype=np.float32)
    image[..., 3] = 0.27
    image[0, 0, :3] = 1 / 64  # equality selects the upper branch
    inputs = {"origTex": image}
    seed = np.random.default_rng(9).normal(size=image.shape).astype(np.float32)
    value = _forward(backend, name, inputs, {})
    actual = backend.vjp(name, inputs, {}, seed)
    if name == "dithering":
        np.testing.assert_allclose(value[0, 0, :3], 1 / 64 + 1 / 255)
        np.testing.assert_allclose(value[0, 1, :3], 0.6)
        np.testing.assert_allclose(value[1, 0, :3], 0.6 + 1 / 255)
        np.testing.assert_array_equal(value[..., 3], image[..., 3])
        np.testing.assert_array_equal(actual.inputs["origTex"], seed)
    else:
        np.testing.assert_array_equal(value[0, 0], 1)
        np.testing.assert_array_equal(value[0, 1, :3], 0)
        np.testing.assert_array_equal(value[1, 0, :3], 1)
        np.testing.assert_array_equal(value[..., 3], 1)
        np.testing.assert_array_equal(actual.inputs["origTex"], 0)
    np.testing.assert_array_equal(value[0, 1], value[8, 9])


@pytest.mark.parametrize("name", ["to_lab", "to_rgb"])
def test_color_space_piecewise_branches_and_alpha_derivative(backend, name):
    image = np.array(
        [
            [[0.0001, 0.02, 0.03, -0.2], [0.1, 0.4, 0.8, 1.3]],
            [[0.7, 0.2, 0.5, 0.37], [1.2, -0.1, 0.03, 0.6]],
        ],
        dtype=np.float32,
    )
    if name == "to_rgb":
        # Both inverse Lab branches, and negative linear RGB components.
        image[..., :3] = [
            [[0.02, 0.5, 0.5], [0.5, 0.49, 0.51]],
            [[0.8, 0.7, 0.2], [0.03, 0.52, 0.47]],
        ]
    inputs = {"origTex": image}
    seed = np.random.default_rng(18).normal(size=image.shape).astype(np.float32)
    expected = _contract(name).vjp((2, 2), inputs, {}, seed)
    actual = backend.vjp(name, inputs, {}, seed)
    np.testing.assert_allclose(
        actual.inputs["origTex"],
        expected.inputs["origTex"],
        atol=2e-5,
        rtol=6e-5,
    )
    np.testing.assert_array_equal(actual.inputs["origTex"][..., 3], seed[..., 3])
    np.testing.assert_array_equal(
        _forward(backend, name, inputs, {})[..., 3], image[..., 3]
    )
