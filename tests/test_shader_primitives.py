"""Raw GLSL uniform contracts, analytic VJPs, and direct compute-shader parity."""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui.backends.opengl.nodes.registry import registry
from sdf_ui.learning.autodiff import Parameter, Tensor
from sdf_ui.learning.shaders.primitives import RULES

SIZE = (7, 5)
PRIMITIVES = Path(__file__).resolve().parents[1] / "src/sdf_ui/backends/opengl/nodes/primitives"


def _context(size=SIZE):
    y, x = np.indices((size[1], size[0]), dtype=float)
    return SimpleNamespace(size=size, x=Tensor(x), y=Tensor(y))


CASES = [
    ("abs", {}),
    ("arc", dict(center=(2.31, 1.73), radius=2.17, start_angle=0.43, end_angle=4.71)),
    ("bezier", dict(a=(0.23, 0.41), b=(5.37, 5.71), c=(6.21, 0.73))),
    ("capsule", dict(a=(0.23, 0.41), b=(5.37, 3.71), radius=1.13)),
    ("circle", dict(offset=(2.31, 1.73), radius=2.17)),
    ("ellipse", dict(center=(2.31, 1.73), radii=(-3.17, 2.13))),
    ("grid", dict(grid_size=(3.17, 2.61), offset=(0.27, -0.43), angle=0.173)),
    ("half_plane", dict(point=(2.31, 1.73), normal=(0.43, -0.71))),
    ("interpolation", dict(t=0.37)),
    ("intersection", {}),
    ("line", dict(a=(0.23, 0.41), b=(5.37, 3.71))),
    ("masked_union", {}),
    (
        "rect",
        dict(
            offset=(2.31, 1.73),
            size=(2.13, 1.17),
            corner_radius=(0.31, 0.42, 0.23, 0.37),
            angle=0.173,
        ),
    ),
    ("repeat", dict(repeat=3.71)),
    ("ring", dict(center=(2.31, 1.73), radius=2.17, thickness=0.73)),
    (
        "sector",
        dict(center=(2.31, 1.73), radius=2.17, start_angle=0.43, end_angle=4.71),
    ),
    ("smooth_min", dict(smoothness=2.17)),
    ("subtract", {}),
    ("transform", dict(transform_kind=0.0, offset=(0.37, -0.29))),
    ("transform", dict(transform_kind=1.0, center=(2.31, 1.73), factor=-1.17)),
    ("transform", dict(transform_kind=2.0, center=(2.31, 1.73), angle=0.237)),
    ("transform", dict(transform_kind=3.0, center=(2.31, 1.73), skew=(0.173, -0.117))),
    ("triangle", dict(point0=(0.23, 0.41), point1=(5.37, 1.71), point2=(2.21, 4.73))),
    ("union", {}),
]


def _inputs(name, size=SIZE):
    rng = np.random.default_rng(123)
    shape = (size[1], size[0])
    names = {
        "abs": ("sdf0",),
        "repeat": ("sdf0",),
        "transform": ("sdf",),
        "interpolation": ("sdf0", "sdf1"),
        "intersection": ("sdf0", "sdf1"),
        "masked_union": ("sdf0", "sdf1"),
        "smooth_min": ("sdf0", "sdf1"),
        "subtract": ("sdf0", "sdf1"),
        "union": ("sdf0", "sdf1"),
    }.get(name, ())
    return {key: rng.normal(size=shape) for key in names}


def _outputs(result):
    return result if isinstance(result, tuple) else (result,)


def _loss(name, context, inputs, params):
    result = _outputs(RULES[name](context, inputs, params))
    rng = np.random.default_rng(901)
    return sum((value * rng.normal(size=value.shape)).sum() for value in result)


def _check_gradients(name, values, input_values=None, size=SIZE):
    context = _context(size)
    params = {key: Parameter(key, value) for key, value in values.items()}
    if input_values is None:
        input_values = _inputs(name, size)
    inputs = {key: Parameter(key, value) for key, value in input_values.items()}
    _loss(name, context, inputs, params).backward()
    for leaf in (*params.values(), *inputs.values()):
        assert leaf.grad is not None, f"detached {name}.{leaf.name}"
        analytic = leaf.grad.copy()
        initial = leaf.data
        numeric = np.empty(initial.shape)
        for index in np.ndindex(initial.shape):
            plus, minus = initial.copy(), initial.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            leaf.assign(plus)
            high = _loss(name, context, inputs, params).item()
            leaf.assign(minus)
            low = _loss(name, context, inputs, params).item()
            leaf.assign(initial)
            numeric[index] = (high - low) / 2e-5
        np.testing.assert_allclose(
            analytic,
            numeric,
            atol=2e-6,
            rtol=2e-5,
            err_msg=f"{name}.{leaf.name}",
        )


def test_rules_cover_every_standalone_primitive_shader_descriptor():
    names = {
        descriptor.name
        for descriptor in registry.shader_files()
        if "/primitives/" in descriptor.path
    }
    assert set(RULES) == names
    assert "smooth_min" in RULES
    assert "smooth_union" not in RULES


@pytest.mark.parametrize("name,params", CASES)
def test_parameter_and_every_input_pixel_vjp_finite_differences(name, params):
    _check_gradients(name, params)


@pytest.mark.parametrize("name", ["arc", "sector"])
def test_wrapped_angle_gradients(name):
    _check_gradients(
        name, dict(center=(2.31, 1.73), radius=2.17, start_angle=5.43, end_angle=7.71)
    )


def test_tensor_leaves_inside_vector_uniforms_keep_their_graphs():
    x, y, radius = Parameter("x", 2.31), Parameter("y", 1.73), Parameter("radius", 2.17)
    RULES["circle"](_context(), {}, dict(offset=(x, y), radius=radius)).sum().backward()
    assert x.grad is not None and y.grad is not None
    assert radius.grad == -SIZE[0] * SIZE[1]


def test_bezier_equal_endpoint_minima_split_without_duplicate_root_bias():
    a = Parameter("a", [-1.0, 2.0])
    b = Parameter("b", [0.0, 3.0])
    c = Parameter("c", [1.0, 2.0])
    output = RULES["bezier"](_context((1, 1)), {}, dict(a=a, b=b, c=c))
    np.testing.assert_allclose(output.data, [[np.sqrt(5.0)]])
    output.sum().backward()
    np.testing.assert_allclose(a.grad, np.array([-0.5, 1.0]) / np.sqrt(5.0))
    np.testing.assert_array_equal(b.grad, [0.0, 0.0])
    np.testing.assert_allclose(c.grad, np.array([0.5, 1.0]) / np.sqrt(5.0))


def test_bezier_interior_stationary_point_uses_envelope_gradients():
    # The nearest point is interior, near t=1/2. A tiny x curvature avoids the
    # shader's exact-zero nudge while keeping every control-point leaf active.
    _check_gradients(
        "bezier", dict(a=(-1.0, 2.0), b=(0.001, 1.0), c=(1.0, 2.0)), size=(1, 1)
    )


@pytest.mark.parametrize(
    "name,sign", [("union", 1), ("intersection", 1), ("subtract", -1)]
)
def test_hard_csg_ties_split_gradients_equally(name, sign):
    first, second = Parameter("first", [[2.0]]), Parameter("second", [[2.0 * sign]])
    RULES[name](_context((1, 1)), dict(sdf0=first, sdf1=second), {}).sum().backward()
    np.testing.assert_array_equal(first.grad, [[0.5]])
    np.testing.assert_array_equal(second.grad, [[0.5 * sign]])


def test_masked_union_has_scalar_distance_and_unquantized_rgba_mask():
    first = Parameter("first", [[-0.2, 0.7]])
    second = Parameter("second", [[0.1, -0.4]])
    distance, mask = RULES["masked_union"](
        _context((2, 1)), dict(sdf0=first, sdf1=second), {}
    )
    np.testing.assert_array_equal(distance.data, [[-0.2, -0.4]])
    np.testing.assert_allclose(mask.data, [[[0.216] * 3 + [1.0], [0.0] * 3 + [1.0]]])
    mask.sum().backward()
    np.testing.assert_allclose(first.grad, [[-3.78, 0.0]])
    np.testing.assert_allclose(second.grad, [[3.78, 0.0]])


def test_repeat_integer_addressing_zero_spacing_gradient_and_oob_zero():
    source = Parameter("source", -np.arange(1.0, 36.0).reshape(5, 7))
    spacing = Parameter("repeat", 3.71)
    result = RULES["repeat"](_context(), dict(sdf0=source), dict(repeat=spacing))
    x, y = np.meshgrid(np.arange(7), np.arange(5))
    ix, iy = np.rint(x / 3.71), np.rint(y / 3.71)
    ox, oy = np.sign(x - 3.71 * ix), np.sign(y - 3.71 * iy)
    samples = []
    for j in range(2):
        for i in range(2):
            xx = np.trunc(x - 3.71 * (ix + i * ox)).astype(int)
            yy = np.trunc(y - 3.71 * (iy + j * oy)).astype(int)
            valid = (xx >= 0) & (xx < 7) & (yy >= 0) & (yy < 5)
            samples.append(
                np.where(valid, source.data[np.clip(yy, 0, 4), np.clip(xx, 0, 6)], 0)
            )
    np.testing.assert_array_equal(result.data, np.minimum.reduce(samples))
    result.sum().backward()
    assert spacing.grad == 0.0
    assert source.grad.sum() == SIZE[0] * SIZE[1]
    positive = Tensor(np.ones((5, 7)))
    # At x=1 a neighbor truncates to -2 and must read zero, not the edge pixel.
    out = RULES["repeat"](_context(), dict(sdf0=positive), dict(repeat=3.71))
    assert out.data[0, 1] == 0.0


def test_transform_gathers_follow_source_size_and_accumulate_repeated_reads():
    source = Parameter("source", [[1.0, 3.0], [5.0, 9.0]])
    offset = Parameter("offset", [100.0, 100.0])
    result = RULES["transform"](
        _context(), dict(sdf=source), dict(transform_kind=0, offset=offset)
    )
    np.testing.assert_array_equal(result.data, np.ones((5, 7)))
    result.sum().backward()
    np.testing.assert_array_equal(source.grad, [[35.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(offset.grad, [0.0, 0.0])


def test_transform_bilinear_weights_and_coordinate_gradients():
    source = Parameter("source", [[1.0, 3.0], [5.0, 9.0]])
    offset = Parameter("offset", [-0.3, -0.4])
    result = RULES["transform"](
        _context((1, 1)), dict(sdf=source), dict(transform_kind=0, offset=offset)
    )
    np.testing.assert_allclose(result.data, [[3.44]])
    result.sum().backward()
    np.testing.assert_allclose(source.grad, [[0.42, 0.18], [0.28, 0.12]])
    np.testing.assert_allclose(offset.grad, [-2.8, -4.6])


@pytest.mark.parametrize(
    "name,params",
    [
        ("circle", dict(offset=(0.0, 0.0), radius=2.0)),
        ("ellipse", dict(center=(0.0, 0.0), radii=(0.0, -1.0))),
        ("capsule", dict(a=(0.0, 0.0), b=(0.0, 0.0), radius=1.0)),
        ("half_plane", dict(point=(0.0, 0.0), normal=(0.0, 0.0))),
    ],
)
def test_finite_subgradients_at_degenerate_norms(name, params):
    parameters = {key: Parameter(key, value) for key, value in params.items()}
    output = RULES[name](_context((1, 1)), {}, parameters)
    output.sum().backward()
    assert np.isfinite(output.data).all()
    for value in parameters.values():
        assert value.grad is not None
        assert np.isfinite(value.grad).all()


@pytest.mark.parametrize(
    "params",
    [
        dict(transform_kind=1.0, factor=0.0),
        dict(transform_kind=3.0, skew=(1.0, 1.0)),
    ],
)
def test_singular_transforms_fail_explicitly(params):
    with pytest.raises(ValueError, match="non-zero|singular"):
        RULES["transform"](_context(), dict(sdf=Tensor(np.ones((5, 7)))), params)


@pytest.fixture(scope="module")
def gl_context():
    import moderngl

    try:
        context = moderngl.create_standalone_context(require=430)
    except Exception:
        try:
            context = moderngl.create_standalone_context(require=430, backend="egl")
        except Exception as exc:
            pytest.skip(f"OpenGL 4.3 context unavailable: {exc}")
    yield context
    context.release()


def _gl_evaluate(gl, name, inputs, params, size=SIZE):
    """Dispatch the actual shader directly, independent of plugin mappings."""
    folder = {"smooth_min": "smooth_union", "interpolation": "interpolate"}.get(
        name, name
    )
    code = (PRIMITIVES / folder / "shader.glsl").read_text()
    program = gl.compute_shader(code)
    textures, outputs = [], []
    try:
        images = re.findall(
            r"layout\s*\(\s*(\w+),\s*binding\s*=\s*(\d+)\s*\)\s*"
            r"(writeonly|readonly)\s+uniform\s+image2D\s+(\w+)",
            code,
        )
        for image_format, binding, access, key in images:
            components, dtype = (4, "f1") if image_format == "rgba8" else (1, "f4")
            source = inputs.get(key)
            image_size = size if source is None else (source.shape[1], source.shape[0])
            payload = None if source is None else source.astype("f4").tobytes()
            texture = gl.texture(image_size, components, payload, dtype=dtype)
            textures.append(texture)
            writing = access == "writeonly"
            texture.bind_to_image(int(binding), read=not writing, write=writing)
            program[key].value = int(binding)
            if writing:
                outputs.append((texture, components, dtype))
        for key, value in params.items():
            if key in program:
                program[key].value = value
        program.run((size[0] + 15) // 16, (size[1] + 15) // 16, 1)
        gl.memory_barrier()
        result = []
        for texture, components, dtype in outputs:
            shape = (size[1], size[0]) + ((components,) if components > 1 else ())
            value = np.frombuffer(
                texture.read(), dtype="u1" if dtype == "f1" else "f4"
            ).reshape(shape)
            result.append(value / 255.0 if dtype == "f1" else value)
        return tuple(result)
    finally:
        for texture in textures:
            texture.release()
        program.release()


PARITY_CASES = (
    CASES
    + [
        (name, dict(center=(2.31, 1.73), radius=2.17, start_angle=5.43, end_angle=7.71))
        for name in ("arc", "sector")
    ]
    + [
        (
            "transform",
            dict(transform_kind=2.5, center=(2.31, 1.73), skew=(0.173, -0.117)),
        ),
        ("grid", dict(grid_size=(-3.17, 2.61), offset=(0.27, -0.43), angle=0.173)),
        ("smooth_min", dict(smoothness=-1.0)),
    ]
)


@pytest.mark.parametrize("name,params", PARITY_CASES)
def test_direct_glsl_parity(gl_context, name, params):
    inputs = _inputs(name)
    expected = _outputs(
        RULES[name](
            _context(), {key: Tensor(value) for key, value in inputs.items()}, params
        )
    )
    actual = _gl_evaluate(gl_context, name, inputs, params)
    assert len(actual) == len(expected)
    for reference, gpu in zip(expected, actual):
        assert reference.shape == gpu.shape
        tolerance = 0.5 / 255.0 + 1e-6 if gpu.ndim == 3 else 3e-5
        np.testing.assert_allclose(reference.data, gpu, atol=tolerance, rtol=2e-5)
