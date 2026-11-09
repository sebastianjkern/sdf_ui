"""GPU shading duals against independent analytic Tensor references."""

import re
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui.core.plugins.registry import registry
from sdf_ui.learning.autodiff import Tensor
from sdf_ui.learning.shader_derivatives import shader_vjp
from sdf_ui.learning.shader_shading import RULES
from sdf_ui.shaders.shading import SOURCES


def _contract(name):
    return next(s for s in registry.shader_files() if s.name == name).differentiation


def _case(name, size=(4, 3), **overrides):
    width, height = size
    rng = np.random.default_rng(512)
    inputs = {"sdf": rng.uniform(-1.3, 2.8, (height, width))}
    fg, bg = [0.43, 0.28, 0.57, 0.71], [0.13, 0.21, 0.32, 0.27]
    if name == "clear_color":
        inputs, params = {}, {"color": fg}
    elif name == "fill":
        params = dict(color=fg, background=bg, inflate=0.13, first=-1.57, second=1.81)
    elif name == "outline":
        params = dict(outline=fg, background=bg, inflate=0.13)
    elif name == "fill_from_texture":
        inputs["origTex"] = rng.uniform(0.1, 0.9, (height, width, 4))
        params = dict(background=bg, inflate=0.13)
    elif name == "isolines":
        params = dict(
            line_color=fg,
            background=bg,
            inflate=0.13,
            spacing=2.17,
            line_width=0.67,
            feather=0.31,
            phase=-3.29,
        )
    elif name == "light":
        params = dict(
            fgColor=fg,
            bgColor=bg,
            lightDir=[0.23, -0.37, 0.89],
            ambient=0.23,
            diffuse=0.57,
            specular=0.13,
            shininess=3.7,
            normalStrength=1.37,
            bevel=3.71,
            shadeBackground=0.73,
            backgroundBevel=2.93,
            heightProfile=0.0,
            heightGamma=1.23,
            height=1.17,
            inflate=0.13,
            inner=-1.57,
            outer=1.81,
        )
    elif name == "film_grain":
        inputs, params = {}, {"time": 9.3}
    elif name == "perlin_noise":
        inputs, params = {}, {}
    else:
        assert name == "partial_derivative"
        params = {}
    params.update(overrides)
    # Use identical float32 data for CPU reference and GPU upload.
    return (
        {key: value.astype(np.float32) for key, value in inputs.items()},
        {key: np.asarray(value, dtype=np.float32) for key, value in params.items()},
    )


def _reference(name, size, inputs, params):
    y, x = np.indices((size[1], size[0]), dtype=float)
    return RULES[name](
        SimpleNamespace(size=size, x=Tensor(x), y=Tensor(y)),
        {key: Tensor(value) for key, value in inputs.items()},
        {key: Tensor(value) for key, value in params.items()},
    ).numpy()


def _assert_gradients(actual, expected, name):
    for field in ("inputs", "parameters"):
        result, reference = getattr(actual, field), getattr(expected, field)
        assert result.keys() == reference.keys()
        for key, value in reference.items():
            assert np.all(np.isfinite(result[key])), f"{name}: {key}"
            np.testing.assert_allclose(
                result[key], value, atol=8e-5, rtol=3e-4, err_msg=f"{name}: {key}"
            )


def test_sources_cover_exact_shader_descriptors_and_rgba_api():
    descriptors = {
        s.name
        for s in registry.shader_files()
        if s.path.startswith("core/plugins/shading/")
    }
    assert set(SOURCES) == descriptors == set(RULES)
    for source in SOURCES.values():
        assert re.search(r"void evaluate\(ivec2 pos, out D o\[8\]\)", source)
        assert "#version" not in source
        for index, channel in enumerate("xyzw"):
            assert f"o[{index}] = color.{channel}" in source


@pytest.fixture(scope="module")
def gl_context():
    import moderngl

    try:
        context = moderngl.create_standalone_context(require=430, backend="egl")
    except Exception as exc:
        pytest.skip(f"OpenGL 4.3 EGL context unavailable: {exc}")
    yield context
    context.release()


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_dual_bodies_compile_with_shared_emitter(gl_context, name):
    from sdf_ui.backends.opengl.glsl import derivative_source

    shader = gl_context.compute_shader(
        derivative_source(_contract(name), SOURCES[name])
    )
    shader.release()


def _audit_body(context, name, size, inputs, params):
    """Evaluate primal/JVP with the real prelude, independent of VJP scattering."""
    from sdf_ui.backends.opengl.glsl import PRELUDE

    contract = _contract(name)
    values, spans, macros, offsets, channels = [], {}, [], [], []
    for category, arguments in (
        ("inputs", contract.inputs),
        ("parameters", contract.parameters),
    ):
        for argument in arguments:
            data = np.asarray(
                (inputs if category == "inputs" else params)[argument.name]
            )
            start = len(values)
            values.extend(data.reshape(-1))
            spans[category, argument.name] = slice(start, len(values))
            if category == "inputs":
                offsets.append(start)
                channels.append(4 if argument.shape else 1)
            else:
                terms = ",".join(f"dataD({start + i})" for i in range(data.size))
                expression = terms if data.size == 1 else f"D{data.size}({terms})"
                macros.append(f"#define U_{argument.name} {expression}")
    direction = np.random.default_rng(983).normal(size=len(values)).astype("f4")
    for key in contract.zero_gradient_parameters:
        direction[spans["parameters", key]] = 0.0
    source = (
        """#version 430
layout(local_size_x=1, local_size_y=1) in;
uniform ivec2 size;
layout(std430,binding=0) readonly buffer Values { float values[]; };
layout(std430,binding=1) readonly buffer Tangents { float tangents[]; };
layout(std430,binding=2) writeonly buffer Result { vec2 result[]; };
"""
        + PRELUDE
        + """
D dataD(int i) { return D(values[i], tangents[i]); }
D ld(int which, ivec2 p, int channel) {
    if (any(lessThan(p,ivec2(0))) || any(greaterThanEqual(p,size))) return c(0.0);
"""
    )
    for index, (offset, count) in enumerate(zip(offsets, channels)):
        source += f"if(which=={index}) return dataD({offset}+(p.y*size.x+p.x)*{count}+channel);\n"
    source += "return c(0.0);}\n" + "\n".join(macros) + "\n" + SOURCES[name]
    source += """
void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    D o[8]; evaluate(pos, o);
    int start = (pos.y*size.x+pos.x)*4;
    for(int i=0;i<4;i++) result[start+i] = vec2(o[i].v, o[i].d);
}
"""
    with ExitStack() as stack:
        shader = context.compute_shader(source)
        stack.callback(shader.release)
        payloads = (
            np.asarray(values or [0.0], dtype="f4").tobytes(),
            (direction if len(direction) else np.zeros(1, "f4")).tobytes(),
            bytes(size[0] * size[1] * 4 * 8),
        )
        for index, data in enumerate(payloads):
            buffer = context.buffer(data)
            stack.callback(buffer.release)
            buffer.bind_to_storage_buffer(index)
        shader["size"].value = size
        shader.run(group_x=size[0], group_y=size[1])
        context.memory_barrier()
        result = (
            np.frombuffer(buffer.read(), dtype="f4")
            .reshape(size[1], size[0], 4, 2)
            .copy()
        )
    directions = {category: {} for category in ("inputs", "parameters")}
    for (category, key), span in spans.items():
        shape = (inputs if category == "inputs" else params)[key].shape
        directions[category][key] = direction[span].reshape(shape)
    return result[..., 0], result[..., 1], directions


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_body_primal_and_dual_use_real_prelude(gl_context, name):
    size = (4, 3)
    inputs, params = _case(name, size)
    primal, tangent, directions = _audit_body(gl_context, name, size, inputs, params)
    tolerance = 2.0 / 1024 + 3e-5 if name == "film_grain" else 3e-5
    np.testing.assert_allclose(
        primal, _reference(name, size, inputs, params), atol=tolerance, rtol=0
    )
    upstream = np.random.default_rng(37).normal(size=primal.shape).astype("f4")
    expected = shader_vjp(_contract(name), size, inputs, params, upstream)
    projection = sum(
        np.sum(getattr(expected, category)[key] * direction)
        for category, values in directions.items()
        for key, direction in values.items()
    )
    np.testing.assert_allclose(
        np.sum(tangent * upstream), projection, atol=8e-5, rtol=3e-4
    )


def test_perlin_body_values_across_lattice_cells(gl_context):
    size = (401, 233)
    primal, tangent, _ = _audit_body(gl_context, "perlin_noise", size, {}, {})
    np.testing.assert_allclose(
        primal, _reference("perlin_noise", size, {}, {}), atol=3e-5, rtol=0
    )
    np.testing.assert_array_equal(tangent, 0.0)
    assert primal[..., 0].min() < 0.0


@pytest.fixture
def backend(gl_context):
    from sdf_ui import Canvas
    from sdf_ui.backends.opengl.backend import GPUShaderBackend

    @contextmanager
    def open_backend(size=(4, 3)):
        with Canvas(size) as canvas, GPUShaderBackend(canvas) as value:
            yield value

    return open_backend


def _check_vjp(backend, name, *, size=(4, 3), inputs=None, **overrides):
    images, params = _case(name, size, **overrides)
    if inputs is not None:
        images = inputs
    upstream = np.random.default_rng(73).uniform(-0.7, 1.0, (size[1], size[0], 4))
    upstream = upstream.astype(np.float32)
    expected = shader_vjp(_contract(name), size, images, params, upstream)
    with backend(size) as runtime:
        actual = runtime.vjp(name, images, params, upstream)
    _assert_gradients(actual, expected, name)
    return actual


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_gpu_vjp_matches_analytic_reference(backend, name):
    _check_vjp(backend, name)


@pytest.mark.parametrize("profile", range(6))
def test_light_height_profiles_gpu_vjp(backend, profile):
    actual = _check_vjp(backend, "light", heightProfile=float(profile))
    np.testing.assert_array_equal(actual.parameters["heightProfile"], 0.0)
    for key in ("lightDir", "normalStrength", "heightGamma", "shininess", "specular"):
        assert np.any(actual.parameters[key] != 0), key
    assert np.any(actual.inputs["sdf"] != 0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"shadeBackground": 0.0},
        {"backgroundBevel": -0.3},
        {"specular": -0.1, "ambient": -0.2, "diffuse": -0.3},
        {"heightGamma": 0.3, "shininess": 0.3},
        {"height": -0.2, "normalStrength": 80.0},
    ],
)
def test_light_branches_gpu_vjp(backend, overrides):
    _check_vjp(backend, "light", **overrides)


@pytest.mark.parametrize("profile", range(6))
@pytest.mark.parametrize("distance", [-10.0, 10.0])
def test_light_flat_endpoints_gpu_vjp(backend, profile, distance):
    actual = _check_vjp(
        backend,
        "light",
        size=(1, 1),
        inputs={"sdf": np.full((1, 1), distance, dtype=np.float32)},
        heightProfile=float(profile),
        heightGamma=0.3,
    )
    np.testing.assert_array_equal(actual.inputs["sdf"], 0.0)


@pytest.mark.parametrize("size", [(1, 1), (1, 3), (4, 1), (4, 3)])
def test_light_clamped_sobel_gpu_scatter(backend, size):
    _check_vjp(backend, "light", size=size)


@pytest.mark.parametrize("size", [(1, 1), (1, 3), (4, 1), (4, 3)])
def test_partial_derivative_gpu_oob_and_transpose(backend, size):
    width, height = size
    source = np.arange(width * height, dtype=np.float32).reshape(height, width) + 0.3
    with backend(size) as runtime:
        actual = runtime.vjp(
            "partial_derivative", {"sdf": source}, {}, np.ones((height, width, 4), "f4")
        )
    expected = np.zeros_like(source)
    expected[:, 0] += 1
    expected[0, :] += 1
    np.testing.assert_array_equal(actual.inputs["sdf"], expected)


def test_fill_from_texture_all_rgba_gpu_gradients(backend):
    actual = _check_vjp(backend, "fill_from_texture")
    assert np.all(np.any(actual.inputs["origTex"] != 0, axis=(0, 1)))
    assert np.any(actual.inputs["sdf"] != 0)
    assert np.any(actual.parameters["background"] != 0)
    assert actual.parameters["inflate"] != 0


@pytest.mark.parametrize("invalid", [False, True])
def test_isolines_gpu_invalid_distance_and_clamped_uniforms(backend, invalid):
    inputs, _ = _case("isolines")
    if invalid:
        inputs["sdf"][:] = 1e21
    _check_vjp(
        backend, "isolines", inputs=inputs, spacing=-0.2, line_width=-0.3, feather=-0.7
    )


@pytest.mark.parametrize("name", ["film_grain", "perlin_noise"])
def test_noise_gpu_vjp_is_zero(backend, name):
    actual = _check_vjp(backend, name, size=(19, 13))
    assert actual.inputs == {}
    for value in actual.parameters.values():
        np.testing.assert_array_equal(value, 0.0)
