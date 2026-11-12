"""Compile every primitive dual and verify native GPU analytic adjoints."""

import re
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pytest

from sdf_ui import Canvas
from sdf_ui.backends.opengl.backend import GPUShaderBackend
from sdf_ui.backends.opengl.glsl import derivative_source
from sdf_ui.backends.opengl.nodes.registry import registry
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from sdf_ui.learning.shader_derivatives import shader_vjp
from sdf_ui.learning.shader_primitives import RULES
from sdf_ui.backends.opengl.shader_sources.primitives import SOURCES

SIZE = (7, 5)
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


def _contract(name):
    return next(s.differentiation for s in registry.shader_files() if s.name == name)


def _inputs(name, size):
    rng = np.random.default_rng(123)
    return {
        argument.name: rng.normal(size=(size[1], size[0])).astype("f4")
        for argument in _contract(name).inputs
    }


def _params(params):
    return {key: np.asarray(value, dtype="f4") for key, value in params.items()}


def _seeds(name, size):
    rng = np.random.default_rng(901)
    seeds = tuple(
        rng.normal(size=(size[1], size[0]) + argument.shape).astype("f4")
        for argument in _contract(name).outputs
    )
    return seeds[0] if len(seeds) == 1 else seeds


@pytest.fixture(scope="module")
def gl_context():
    import moderngl

    try:
        context = moderngl.create_standalone_context(require=430, backend="egl")
    except Exception as exc:
        pytest.skip(f"OpenGL 4.3 EGL context unavailable: {exc}")
    yield context
    context.release()


@pytest.fixture
def backend(gl_context):
    @contextmanager
    def open_backend(size=SIZE):
        with Canvas(size) as canvas, GPUShaderBackend(canvas) as runtime:
            yield runtime

    return open_backend


def _check_vjp(backend, name, params, *, size=SIZE, inputs=None, seeds=None):
    inputs = _inputs(name, size) if inputs is None else inputs
    inputs = {key: np.asarray(value, dtype="f4") for key, value in inputs.items()}
    params = _params(params)
    seeds = _seeds(name, size) if seeds is None else seeds
    expected = shader_vjp(_contract(name), size, inputs, params, seeds)
    with backend(size) as runtime:
        actual = runtime.vjp(name, inputs, params, seeds)
        assert runtime.stats["backward_dispatches"] == 1
    for field in ("inputs", "parameters"):
        result, reference = getattr(actual, field), getattr(expected, field)
        assert result.keys() == reference.keys()
        for key, value in reference.items():
            assert np.all(np.isfinite(result[key])), (name, key)
            np.testing.assert_allclose(
                result[key],
                value,
                atol=8e-5,
                rtol=4e-4,
                err_msg=f"{name}.{key}",
            )
    return actual


def test_sources_cover_all_21_actual_primitive_descriptor_names():
    descriptors = {
        descriptor.name
        for descriptor in registry.shader_files()
        if descriptor.path.startswith("backends/opengl/nodes/primitives/")
    }
    assert set(SOURCES) == descriptors == set(RULES)
    assert len(SOURCES) == 21
    assert "interpolation" in SOURCES and "interpolate" not in SOURCES
    assert "smooth_min" in SOURCES and "smooth_union" not in SOURCES
    for source in SOURCES.values():
        assert re.search(r"void\s+evaluate\(ivec2\s+pos,\s*out\s+D\s+o\[8\]\)", source)
        assert "#version" not in source
        assert "layout(" not in source
        assert "uniform " not in source


def test_all_shape_descriptors_compile_forward_and_backward_from_one_program():
    descriptors = {
        descriptor.name: descriptor
        for descriptor in registry.shader_files()
        if descriptor.path.startswith("backends/opengl/nodes/primitives/")
    }
    for name, descriptor in descriptors.items():
        operation = descriptor.operation
        program = operation.expression()
        assert isinstance(program, ShaderProgram), name
        assert program.body == SOURCES[name]
        if name in {
            "circle",
            "ring",
            "abs",
            "union",
            "intersection",
            "subtract",
            "interpolation",
        }:
            assert "seedKind" not in operation.backward_source()
        else:
            assert program.body in operation.backward_source()
        # Forward uses the same program with zero-tangent uniforms and inputs.
        assert "void evaluate(ivec2 pos,out D o[8])" in operation.forward_source()


def test_circle_reverse_gradient_uses_zero_subgradient_at_center(backend):
    gradient = _check_vjp(
        backend,
        "circle",
        {"offset": (0, 0), "radius": 1},
        size=(1, 1),
        seeds=np.ones((1, 1), dtype=np.float32),
    )
    np.testing.assert_array_equal(gradient.parameters["offset"], (0, 0))
    assert gradient.parameters["radius"] == -1


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_all_dual_bodies_compile_with_actual_shared_emitter(gl_context, name):
    program = gl_context.compute_shader(
        derivative_source(_contract(name), SOURCES[name])
    )
    program.release()


@pytest.mark.parametrize("name,params", CASES)
def test_all_gpu_vjps_match_independent_analytic_reference(backend, name, params):
    _check_vjp(backend, name, params)


@pytest.mark.parametrize("name,params", CASES)
def test_original_fp32_gpu_forward_matches_reference(backend, name, params):
    inputs, params = _inputs(name, SIZE), _params(params)
    reference = _contract(name).evaluate(SIZE, inputs, params)
    reference = reference if isinstance(reference, tuple) else (reference,)
    with backend() as runtime, ExitStack() as resources:
        images = {
            key: resources.enter_context(runtime.upload(value))
            for key, value in inputs.items()
        }
        outputs = [
            resources.enter_context(image)
            for image in runtime.forward(name, images, params)
        ]
        for image, expected in zip(outputs, reference):
            np.testing.assert_allclose(
                image.read(), expected.numpy(), atol=4e-5, rtol=3e-5
            )


@pytest.mark.parametrize("name,params", CASES)
def test_shared_primal_matches_original_glsl(backend, name, params):
    """Keep the old shader as an independent pixel-level conformance oracle."""
    contract = _contract(name)
    source = (
        Path(__file__).resolve().parents[1] / "src/sdf_ui" / contract.path
    ).read_text()
    source = source.replace("rgba8", "rgba32f")
    with backend() as runtime, ExitStack() as resources:
        inputs = {
            key: resources.enter_context(runtime.upload(value))
            for key, value in _inputs(name, SIZE).items()
        }
        actual = [
            resources.enter_context(value)
            for value in runtime.forward(name, inputs, _params(params))
        ]
        expected = [
            resources.enter_context(
                runtime.allocate(SIZE, 1 if arg.kind == "r32f" else 4)
            )
            for arg in contract.outputs
        ]
        program = runtime.ctx.compute_shader(source)
        try:
            for index, (arg, image) in enumerate(zip(contract.outputs, expected)):
                image.texture.bind_to_image(index, read=False, write=True)
                if arg.name in program:
                    program[arg.name].value = index
            for index, arg in enumerate(contract.inputs, start=len(expected)):
                inputs[arg.name].texture.bind_to_image(index, read=True, write=False)
                if arg.name in program:
                    program[arg.name].value = index
            for arg in contract.parameters:
                if arg.name in program:
                    program[arg.name].write(
                        runtime.uniforms(name, _params(params))[arg.name].tobytes()
                    )
            program.run(1, 1, 1)
            runtime.ctx.memory_barrier()
            for actual_image, original_image in zip(actual, expected):
                np.testing.assert_allclose(
                    actual_image.read(),
                    original_image.read(),
                    atol=1e-4,
                    rtol=1e-4,
                    err_msg=name,
                )
        finally:
            program.release()


@pytest.mark.parametrize("seed", range(4))
def test_shared_bezier_primal_matches_original_for_varied_control_points(backend, seed):
    points = np.random.default_rng(2026 + seed).uniform(-3, 9, size=(3, 2))
    test_shared_primal_matches_original_glsl(
        backend, "bezier", dict(zip(("a", "b", "c"), points))
    )


@pytest.mark.parametrize("name", ["arc", "sector"])
def test_wrapped_angle_gpu_vjps(backend, name):
    _check_vjp(
        backend,
        name,
        dict(
            center=(2.31, 1.73),
            radius=2.17,
            start_angle=5.43,
            end_angle=7.71,
        ),
    )


@pytest.mark.parametrize("kind", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
@pytest.mark.parametrize("size", [(1, 1), (1, 5), (7, 1), SIZE])
def test_transform_modes_boundaries_and_bilinear_gpu_scatter(backend, kind, size):
    gradients = _check_vjp(
        backend,
        "transform",
        dict(
            transform_kind=kind,
            center=(2.31, 1.73),
            factor=-1.17,
            offset=(0.37, -0.29),
            angle=0.237,
            skew=(0.173, -0.117),
        ),
        size=size,
    )
    np.testing.assert_array_equal(gradients.parameters["transform_kind"], 0.0)


@pytest.mark.parametrize("offset", [(100.0, 100.0), (-100.0, -100.0), (0.0, 0.0)])
def test_transform_clamped_coordinate_and_duplicate_texel_derivatives(backend, offset):
    _check_vjp(backend, "transform", dict(transform_kind=0.0, offset=offset))


@pytest.mark.parametrize("spacing", [3.71, -3.71, 0.3, 2.0])
@pytest.mark.parametrize("size", [(1, 1), SIZE])
def test_repeat_discrete_spacing_and_gpu_sample_routing(backend, spacing, size):
    # Negative values ensure an actual sampled texel can beat zero OOB reads.
    source = -np.arange(1, size[0] * size[1] + 1, dtype="f4").reshape(size[1], size[0])
    gradients = _check_vjp(
        backend, "repeat", dict(repeat=spacing), size=size, inputs={"sdf0": source}
    )
    np.testing.assert_array_equal(gradients.parameters["repeat"], 0.0)


def test_repeat_oob_zero_wins_and_receives_no_adjoint(backend):
    gradients = _check_vjp(
        backend,
        "repeat",
        dict(repeat=0.3),
        inputs={"sdf0": np.full((5, 7), 2.0, dtype="f4")},
    )
    assert np.count_nonzero(gradients.inputs["sdf0"]) < 35


@pytest.mark.parametrize(
    "name,sign", [("union", 1), ("intersection", 1), ("subtract", -1)]
)
def test_hard_csg_ties_split_gpu_adjoints(backend, name, sign):
    gradients = _check_vjp(
        backend,
        name,
        {},
        size=(1, 1),
        inputs={"sdf0": [[2.0]], "sdf1": [[2.0 * sign]]},
        seeds=np.ones((1, 1), "f4"),
    )
    np.testing.assert_array_equal(gradients.inputs["sdf0"], [[0.5]])
    np.testing.assert_array_equal(gradients.inputs["sdf1"], [[0.5 * sign]])


@pytest.mark.parametrize("channel", range(5))
def test_masked_union_flattened_distance_rgba_output_seeds(backend, channel):
    distance_seed = np.zeros((1, 2), "f4")
    rgba_seed = np.zeros((1, 2, 4), "f4")
    if channel == 0:
        distance_seed[:] = 1
    else:
        rgba_seed[..., channel - 1] = 1
    gradients = _check_vjp(
        backend,
        "masked_union",
        {},
        size=(2, 1),
        inputs={"sdf0": [[-0.2, 0.7]], "sdf1": [[0.1, -0.4]]},
        seeds=(distance_seed, rgba_seed),
    )
    if channel == 4:
        for value in gradients.inputs.values():
            np.testing.assert_array_equal(value, 0.0)


@pytest.mark.parametrize(
    "params",
    [
        dict(a=(-1.0, 2.0), b=(0.0, 3.0), c=(1.0, 2.0)),
        dict(a=(-1.0, 2.0), b=(0.001, 1.0), c=(1.0, 2.0)),
        dict(a=(2.0, 1.0), b=(3.0, 5.0), c=(4.0, 1.0)),
        dict(a=(-2.0, 0.0), b=(0.0, 3.0), c=(2.0, 1.0)),
        dict(a=(-1.0, 1.0), b=(0.0, 2.0), c=(3.0, 1.0)),
    ],
)
def test_bezier_endpoint_ties_interior_and_zero_quadratic_nudge(backend, params):
    _check_vjp(backend, "bezier", params, size=(1, 1), seeds=np.ones((1, 1), "f4"))


def test_triangle_equal_axis_edges_split_gpu_adjoints(backend):
    # Dyadic segment projections keep this tie exact in both FP32 and FP64.
    _check_vjp(
        backend,
        "triangle",
        dict(
            point0=(-1.0, -1.0),
            point1=(7.0, -1.0),
            point2=(-1.0, 7.0),
        ),
        size=(1, 1),
        seeds=np.ones((1, 1), "f4"),
    )


def test_sector_three_equal_candidates_share_gpu_radius_adjoint(backend):
    gradients = _check_vjp(
        backend,
        "sector",
        dict(center=(0.0, -1.0), radius=2.0, start_angle=0.0, end_angle=0.0),
        size=(1, 1),
        seeds=np.ones((1, 1), "f4"),
    )
    np.testing.assert_allclose(gradients.parameters["radius"], 1.0 / 3.0)


@pytest.mark.parametrize(
    "name,params",
    [case for case in CASES if case[0] in {"circle", "bezier", "repeat", "transform"}],
)
def test_native_gpu_vjp_does_not_invoke_cpu_primitive_rules(
    backend, monkeypatch, name, params
):
    inputs = _inputs(name, SIZE)
    params, seeds = _params(params), _seeds(name, SIZE)
    expected = shader_vjp(_contract(name), SIZE, inputs, params, seeds)

    def forbidden(*args, **kwargs):
        pytest.fail("GPU differentiation invoked a CPU primitive rule")

    monkeypatch.setitem(RULES, name, forbidden)
    with backend() as runtime:
        actual = runtime.vjp(name, inputs, params, seeds)
    for field in ("inputs", "parameters"):
        for key, reference in getattr(expected, field).items():
            np.testing.assert_allclose(
                getattr(actual, field)[key],
                reference,
                atol=8e-5,
                rtol=4e-4,
            )


@pytest.mark.parametrize(
    "name,params",
    [
        ("circle", dict(offset=(0.0, 0.0), radius=1.0)),
        ("ring", dict(center=(0.0, 0.0), radius=0.0, thickness=1.0)),
        ("capsule", dict(a=(0.0, 0.0), b=(0.0, 0.0), radius=1.0)),
        ("half_plane", dict(point=(0.0, 0.0), normal=(0.0, 0.0))),
        ("ellipse", dict(center=(0.0, 0.0), radii=(0.0, 0.0))),
        ("grid", dict(grid_size=(-3.17, 2.61), offset=(0.27, -0.43), angle=0.173)),
        ("smooth_min", dict(smoothness=-1.0)),
    ],
)
def test_gpu_origin_subgradients_and_epsilon_branches(backend, name, params):
    _check_vjp(backend, name, params, size=(1, 1))
