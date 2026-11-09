"""Shader-name contracts, analytic gradients, and real GLSL compute parity."""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui.core.plugins.registry import registry
from sdf_ui.learning.autodiff import Parameter, Tensor
from sdf_ui.learning.shader_shading import RULES


def _context(size=(4, 3)):
    y, x = np.indices((size[1], size[0]), dtype=float)
    return SimpleNamespace(size=size, x=Tensor(x), y=Tensor(y))


def _case(name, size=(4, 3)):
    rng = np.random.default_rng(512)
    inputs = {"sdf": rng.uniform(-1.3, 2.8, (size[1], size[0]))}
    foreground = [0.43, 0.28, 0.57, 0.71]
    background = [0.13, 0.21, 0.32, 0.27]
    if name == "clear_color":
        return {}, {"color": foreground}
    if name == "film_grain":
        return {}, {"time": 0.0}
    if name == "fill":
        params = dict(
            color=foreground,
            background=background,
            inflate=0.13,
            first=-1.57,
            second=1.81,
        )
    elif name == "outline":
        params = dict(outline=foreground, background=background, inflate=0.13)
    elif name == "fill_from_texture":
        inputs["origTex"] = rng.uniform(0.1, 0.9, (size[1], size[0], 4))
        params = dict(background=background, inflate=0.13)
    elif name == "isolines":
        params = dict(
            line_color=foreground,
            background=background,
            inflate=0.13,
            spacing=2.17,
            line_width=0.67,
            feather=0.31,
            phase=-3.29,
        )
    elif name == "light":
        params = dict(
            fgColor=foreground,
            bgColor=background,
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
    else:
        return (inputs if name == "partial_derivative" else {}), {}
    return inputs, params


def _check_gradients(name, *, overrides=None, tolerance=3e-5):
    context = _context()
    input_values, uniform_values = _case(name)
    uniform_values.update(overrides or {})
    inputs = {key: Parameter(key, value) for key, value in input_values.items()}
    # heightProfile is a discrete branch selector, not a continuous parameter.
    params = {
        key: Tensor(value) if key == "heightProfile" else Parameter(key, value)
        for key, value in uniform_values.items()
    }
    leaves = list(inputs.values()) + [
        v for v in params.values() if isinstance(v, Parameter)
    ]
    weights = np.random.default_rng(73).uniform(-0.7, 1.0, (3, 4, 4))

    def objective():
        value = RULES[name](context, inputs, params)
        return (value * weights + 0.03 * value * value).mean()

    objective().backward()
    analytic = [np.zeros(p.shape) if p.grad is None else p.grad.copy() for p in leaves]
    assert any(np.any(grad != 0) for grad in analytic)
    for parameter, grad in zip(leaves, analytic):
        original = parameter.numpy()
        numeric = np.zeros_like(original)
        for index in np.ndindex(original.shape):
            plus, minus = original.copy(), original.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            parameter.assign(plus)
            high = objective().item()
            parameter.assign(minus)
            low = objective().item()
            parameter.assign(original)
            numeric[index] = (high - low) / 2e-5
        np.testing.assert_allclose(
            grad,
            numeric,
            atol=tolerance,
            rtol=tolerance,
            err_msg=f"{name}: {parameter.name}",
        )
    return inputs, params


def test_rules_cover_exact_standalone_shader_descriptors():
    registry.ensure_loaded()
    descriptors = {
        descriptor.name
        for descriptor in registry.shader_files()
        if descriptor.path.startswith("core/plugins/shading/")
    }
    assert set(RULES) == descriptors
    for name, rule in RULES.items():
        inputs, params = _case(name)
        value = rule(
            _context(),
            {key: Tensor(v) for key, v in inputs.items()},
            {key: Tensor(v) for key, v in params.items()},
        )
        assert isinstance(value, Tensor)
        assert value.shape == (3, 4, 4)
        assert np.all(np.isfinite(value.numpy()))


@pytest.mark.parametrize(
    "name",
    [
        "clear_color",
        "fill",
        "outline",
        "fill_from_texture",
        "isolines",
        "partial_derivative",
    ],
)
def test_texture_and_uniform_gradients(name):
    inputs, params = _check_gradients(name)
    if name == "fill_from_texture":
        assert np.any(inputs["origTex"].grad != 0)
        assert np.any(inputs["sdf"].grad != 0)
        assert np.any(params["inflate"].grad != 0)


@pytest.mark.parametrize("profile", range(6))
def test_light_height_profiles_gradients(profile):
    inputs, params = _check_gradients(
        "light", overrides={"heightProfile": float(profile)}
    )
    assert np.any(inputs["sdf"].grad != 0)
    for key in ["lightDir", "heightGamma", "shininess", "normalStrength", "specular"]:
        assert np.any(params[key].grad != 0), key


@pytest.mark.parametrize(
    "overrides",
    [
        {"shadeBackground": 0.0},
        {"backgroundBevel": -0.3},
        {"specular": -0.1, "ambient": -0.2, "diffuse": -0.3},
    ],
)
def test_light_other_branches_gradients(overrides):
    _check_gradients("light", overrides=overrides)


@pytest.mark.parametrize("profile", range(6))
@pytest.mark.parametrize("distance", [-10.0, 10.0])
def test_light_flat_endpoints_have_finite_gradients(profile, distance):
    _, values = _case("light")
    values["heightProfile"] = float(profile)
    params = {
        key: Tensor(value) if key == "heightProfile" else Parameter(key, value)
        for key, value in values.items()
    }
    sdf = Parameter("sdf", np.full((1, 1), distance))
    value = RULES["light"](_context((1, 1)), {"sdf": sdf}, params)
    value.sum().backward()
    np.testing.assert_array_equal(sdf.grad, 0.0)
    for parameter in params.values():
        if parameter.grad is not None:
            assert np.all(np.isfinite(parameter.grad))


@pytest.mark.parametrize("size", [(1, 1), (1, 3), (4, 1), (4, 3)])
def test_partial_derivative_explicit_oob_zero_and_adjoint(size):
    width, height = size
    sdf = Parameter("sdf", np.arange(width * height).reshape(height, width) + 0.3)
    value = RULES["partial_derivative"](_context(size), {"sdf": sdf}, {})
    data = sdf.numpy()
    padded = np.pad(data, ((0, 1), (0, 1)))
    expected = np.stack(
        (
            data - padded[:height, 1:],
            data - padded[1:, :width],
            np.zeros_like(data),
            np.ones_like(data),
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(value.numpy(), expected)
    value.sum().backward()
    expected_grad = np.zeros_like(data)
    expected_grad[:, 0] += 1
    expected_grad[0, :] += 1
    np.testing.assert_array_equal(sdf.grad, expected_grad)


def test_uniform_tensor_containers_keep_their_graphs():
    red = Parameter("red", 0.37)
    value = RULES["clear_color"](_context(), {}, {"color": (red, 0.2, 0.4, 0.8)})
    value.mean().backward()
    assert red.grad is not None and np.isfinite(red.grad) and red.grad != 0


def test_isolines_invalid_distances_and_clamped_parameters():
    inputs, params = _case("isolines")
    params.update(spacing=-0.2, line_width=-0.3, feather=-0.7)
    _check_gradients("isolines", overrides=params)
    inputs["sdf"][:] = 1e21
    actual = RULES["isolines"](_context(), {"sdf": Tensor(inputs["sdf"])}, params)
    background = RULES["clear_color"](_context(), {}, {"color": params["background"]})
    np.testing.assert_array_equal(actual.numpy(), background.numpy())


@pytest.mark.parametrize("name", ["film_grain", "perlin_noise"])
def test_noise_is_fixed_deterministic_unclipped_and_composable(name):
    context = _context((301, 193))
    first = RULES[name](context, {}, {})
    second = RULES[name](context, {}, {"time": Parameter("time", 9.3)})
    assert not first.requires_grad and not second.requires_grad
    np.testing.assert_array_equal(first.numpy(), second.numpy())
    np.testing.assert_array_equal(first.numpy()[..., 0], first.numpy()[..., 1])
    np.testing.assert_array_equal(first.numpy()[..., 3], 1.0)
    if name == "perlin_noise":
        assert first.numpy()[..., 0].min() < 0.0
    gain = Parameter("gain", 0.7)
    (first * gain).mean().backward()
    assert gain.grad == pytest.approx(first.numpy().mean())


@pytest.fixture(scope="module")
def gl_context():
    import moderngl

    try:
        context = moderngl.create_standalone_context(require=430, backend="egl")
    except Exception as exc:
        pytest.skip(f"OpenGL 4.3 EGL context unavailable: {exc}")
    yield context
    context.release()


def _gl_run(context, name, inputs, params, *, size, floating):
    root = Path(__file__).resolve().parents[1] / "src/sdf_ui/core/plugins/shading"
    source = (root / name / "shader.glsl").read_text()
    if floating:
        source = source.replace("layout (rgba8,", "layout (rgba32f,")
    shader = context.compute_shader(source)
    textures = []
    try:
        output = context.texture(size, 4, dtype="f4" if floating else "f1")
        textures.append(output)
        output.bind_to_image(0, read=False, write=True)
        for key, value in inputs.items():
            binding = int(
                re.search(
                    r"binding = (\d+)\) readonly uniform image2D " + key + r";", source
                )[1]
            )
            scalar = value.ndim == 2
            dtype = "f4" if scalar or floating else "f1"
            data = (
                value.astype(np.float32)
                if dtype == "f4"
                else np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
            )
            texture = context.texture(
                size, 1 if scalar else 4, data.tobytes(), dtype=dtype
            )
            textures.append(texture)
            texture.bind_to_image(binding, read=True, write=False)
        for key, value in params.items():
            if key in shader:  # GLSL may optimize unused uniforms out.
                data = np.asarray(value)
                shader[key].value = float(data) if data.ndim == 0 else tuple(data)
        shader.run(group_x=(size[0] + 15) // 16, group_y=(size[1] + 15) // 16)
        context.memory_barrier()
        dtype = np.float32 if floating else np.uint8
        result = np.frombuffer(output.read(), dtype=dtype).reshape(size[1], size[0], 4)
        return result.copy() if floating else result / 255.0
    finally:
        for texture in textures:
            texture.release()
        shader.release()


@pytest.mark.parametrize("floating", [True, False], ids=["float32", "rgba8"])
@pytest.mark.parametrize("name", sorted(RULES))
def test_glsl_pixel_parity(gl_context, name, floating):
    size = (19, 13)  # Non-workgroup dimensions exercise the GLSL bounds guards.
    inputs, params = _case(name, size)
    for key, value in inputs.items():
        inputs[key] = (
            np.rint(value * 255) / 255
            if value.ndim == 3 and not floating
            else value.astype(np.float32).astype(float)
        )
    actual = RULES[name](
        _context(size), {k: Tensor(v) for k, v in inputs.items()}, params
    ).numpy()
    gpu = _gl_run(gl_context, name, inputs, params, size=size, floating=floating)
    if not floating:
        actual = np.clip(actual, 0.0, 1.0)
    tolerance = 3e-5 if floating else 0.5 / 255 + 3e-5
    # Trigonometric hash noise amplifies platform cos/dot rounding by 12345.
    if name == "film_grain":
        tolerance += 2.0 / 1024
    np.testing.assert_allclose(actual, gpu, atol=tolerance, rtol=0)


@pytest.mark.parametrize("profile", range(6))
@pytest.mark.parametrize("shade_background", [0.0, 0.73])
def test_light_glsl_profiles_and_background(gl_context, profile, shade_background):
    size = (9, 7)
    inputs, params = _case("light", size)
    params.update(
        heightProfile=float(profile),
        shadeBackground=shade_background,
        backgroundBevel=-0.3,
    )
    actual = RULES["light"](_context(size), {"sdf": Tensor(inputs["sdf"])}, params)
    gpu = _gl_run(gl_context, "light", inputs, params, size=size, floating=True)
    np.testing.assert_allclose(actual.numpy(), gpu, atol=3e-5, rtol=0)


def test_perlin_glsl_across_lattice_cells(gl_context):
    size = (401, 233)
    actual = RULES["perlin_noise"](_context(size), {}, {}).numpy()
    gpu = _gl_run(gl_context, "perlin_noise", {}, {}, size=size, floating=True)
    np.testing.assert_allclose(actual, gpu, atol=3e-5, rtol=0)
