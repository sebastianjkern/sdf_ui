"""Every registered shader has an executable and auditable derivative contract."""

from pathlib import Path

import numpy as np
import pytest

from sdf_ui.gpu.nodes.base import ShaderFile
from sdf_ui.gpu.nodes.registry import registry
from sdf_ui.learning import Parameter

_CIRCLE_TEXT = "circle"

_OFFSET_TEXT = "offset"

_ORIGTEX_TEXT = "origTex"

_RADIUS_TEXT = "radius"
_MASK_SMOOTHSTEP_T = 0.9


def descriptor(name):
    return {shader.name: shader for shader in registry.shader_files()}[name]


def test_all_shader_files_have_contracts_and_executable_rules():
    shaders = {shader.name: shader for shader in registry.shader_files()}
    root = Path(__file__).resolve().parents[1] / "src/sdf_ui"
    glsl_leftovers = [
        str(path.relative_to(root.parent)) for path in root.rglob("*.glsl")
    ]
    assert glsl_leftovers == []
    for name, shader in shaders.items():
        _assert_shader_contract(name, shader, root)


def _assert_shader_contract(name, shader, root):
    contract = shader.differentiation
    assert contract is not None, name
    assert contract.name == name and contract.path == shader.path
    assert contract.outputs and contract.notes
    assert shader.interface is not None, name
    _assert_shader_executable(shader, name, root)
    assert callable(contract.reference), name
    assert [
        item.name for item in (shader.interface.output, *shader.interface.extra_outputs)
    ] == [item.name for item in contract.outputs]
    assert [item.name for item in shader.interface.inputs] == [
        item.name for item in contract.inputs
    ]
    assert [item.name for item in shader.interface.uniforms] == [
        item.name for item in contract.parameters
    ]
    assert [
        item.unit
        for item in (
            shader.interface.output,
            *shader.interface.extra_outputs,
            *shader.interface.inputs,
        )
    ] == list(range(len(contract.outputs) + len(contract.inputs)))
    assert set(contract.zero_gradient_parameters) <= {
        parameter.name for parameter in contract.parameters
    }


def _assert_shader_executable(shader, name, root):
    if shader.wgsl is not None:
        assert "fn compute_main" in shader.wgsl.forward()
        assert "fn compute_main" in shader.wgsl.backward
    else:
        assert (root / shader.path).is_file(), name


def test_circle_backward_reduces_pixels_and_does_not_mutate_callers():
    shader = descriptor(_CIRCLE_TEXT)
    radius = Parameter(_RADIUS_TEXT, 2.3)
    radius.grad = np.array(7.0)
    size = (7, 5)
    upstream = np.arange(35.0).reshape(5, 7) / 35
    params = {_OFFSET_TEXT: [2.3, 1.7], _RADIUS_TEXT: radius}
    gradients = shader.backward(size, {}, params, upstream)
    y, x = np.indices((5, 7))
    dx, dy = x - 23 / 10, y - 17 / 10
    norm = np.hypot(dx, dy)
    np.testing.assert_allclose(
        gradients.parameters[_OFFSET_TEXT],
        [-np.sum(upstream * dx / norm), -np.sum(upstream * dy / norm)],
    )
    assert gradients.parameters[_RADIUS_TEXT] == pytest.approx(-upstream.sum())
    assert radius.grad == 7
    assert gradients.inputs == {}
    with pytest.raises(TypeError):
        gradients.parameters[_RADIUS_TEXT] = 1
    with pytest.raises(ValueError):
        gradients.parameters[_OFFSET_TEXT][0] = 1


def test_contract_evaluation_preserves_existing_parameter_graph():
    radius = Parameter(_RADIUS_TEXT, 3.0)
    derived = radius * 2
    field = descriptor(_CIRCLE_TEXT).differentiation.evaluate(
        (4, 3), {}, {_OFFSET_TEXT: [1.3, 1.6], _RADIUS_TEXT: derived}
    )
    field.mean().backward()
    assert radius.grad == pytest.approx(-2)


def test_masked_union_accumulates_both_output_seeds():
    size, shape = (3, 2), (2, 3)
    inputs = {"sdf0": np.full(shape, -3 / 10), "sdf1": np.full(shape, 0.1)}
    mask_seed = np.zeros(shape + (4,))
    mask_seed[..., 0] = 2
    gradient = descriptor("masked_union").backward(
        size, inputs, {}, {"destTex": np.full(shape, 3.0), "maskTex": mask_seed}
    )
    # The mask's normalized smoothstep coordinate is (0.4 + 0.5) / 1 = 0.9.
    mask_gradient = 2 * 6 * _MASK_SMOOTHSTEP_T * (1 - _MASK_SMOOTHSTEP_T)
    np.testing.assert_allclose(gradient.inputs["sdf0"], 3 - mask_gradient)
    np.testing.assert_allclose(gradient.inputs["sdf1"], mask_gradient)

    eps = 1e-3
    _, mask_plus = descriptor("masked_union").differentiation.evaluate(
        size, {"sdf0": inputs["sdf0"] + eps, "sdf1": inputs["sdf1"]}
    )
    _, mask_minus = descriptor("masked_union").differentiation.evaluate(
        size, {"sdf0": inputs["sdf0"] - eps, "sdf1": inputs["sdf1"]}
    )
    # d mask / d sdf0 is -6t(1 - t) = -0.54 at t = 0.9, matching the analytic
    # contribution of mask_gradient for the seed of 2.
    np.testing.assert_allclose(
        -(mask_plus.numpy()[..., 0] - mask_minus.numpy()[..., 0]) / (2 * eps),
        mask_gradient / 2,
        atol=1e-5,
    )


def test_blur_vjp_is_transpose_of_actual_border_stencil():
    rng = np.random.default_rng(19)
    size, shape = (4, 3), (3, 4, 4)
    original, upstream = rng.normal(size=shape), rng.normal(size=shape)
    shader = descriptor("blur_hor_9")
    output = shader.differentiation.evaluate(size, {_ORIGTEX_TEXT: original})
    gradients = shader.backward(size, {_ORIGTEX_TEXT: original}, {}, upstream)
    assert np.sum(output.numpy() * upstream) == pytest.approx(
        np.sum(original * gradients.inputs[_ORIGTEX_TEXT]), abs=1e-12
    )
    np.testing.assert_array_equal(gradients.parameters[_OFFSET_TEXT], 0)
    assert _OFFSET_TEXT in shader.differentiation.zero_gradient_parameters
    assert shader.differentiation.sampling == "asymmetric_reflected_zero_outside"


@pytest.mark.parametrize("name,expected", [("dither_1bit", 0), ("dithering", 1)])
def test_hard_threshold_and_additive_dither_have_distinct_gradients(name, expected):
    inputs = {_ORIGTEX_TEXT: np.full((2, 3, 4), 0.514)}
    gradients = descriptor(name).backward((3, 2), inputs, {}, np.ones((2, 3, 4)))
    np.testing.assert_array_equal(gradients.inputs[_ORIGTEX_TEXT], expected)
    if name == "dither_1bit":
        assert descriptor(name).differentiation.behavior == "zero_almost_everywhere"


@pytest.mark.parametrize("name", ["film_grain", "perlin_noise"])
def test_fixed_noise_has_no_trainable_shader_arguments(name):
    shader = descriptor(name)
    assert shader.differentiation.behavior == "constant"
    assert shader.differentiation.continuous_parameters == ()
    result = shader.backward((3, 2), {}, {}, np.ones((2, 3, 4)))
    assert result.inputs == {}
    assert all(np.all(value == 0) for value in result.parameters.values())


def test_shader_file_exposes_only_its_explicit_differentiation_contract():
    shader = next(item for item in registry.shader_files() if item.name == _CIRCLE_TEXT)
    assert shader.differentiation is descriptor(_CIRCLE_TEXT).differentiation
    unknown = ShaderFile("custom", "custom.wgsl")
    assert unknown.differentiation is None
    with pytest.raises(ValueError, match="no differentiation"):
        unknown.backward((3, 2), {}, {}, np.ones((2, 3)))
    # Names and paths do not infer contracts: ownership must be explicit.
    assert (
        ShaderFile(
            _CIRCLE_TEXT, "gpu/nodes/primitives/rect/rect.wgsl"
        ).differentiation
        is None
    )


def test_ports_and_uniform_shapes_are_validated_before_backward():
    contract = descriptor(_CIRCLE_TEXT).differentiation
    for params in ({_OFFSET_TEXT: [1.0]}, {"typo": 3.0}):
        with pytest.raises(ValueError):
            contract.vjp((3, 2), {}, params, np.ones((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        contract.vjp((3, 2), {}, {}, np.ones((3, 2)))
    with pytest.raises(ValueError, match="inputs"):
        descriptor("union").backward((3, 2), {}, {}, np.ones((2, 3)))


@pytest.mark.parametrize(
    "name,params",
    [
        ("triangle", {"point0": [0, 0], "point1": [2, 2], "point2": [4, 4]}),
        ("bezier", {"a": [0, 0], "b": [1, 1], "c": [4, 4]}),
    ],
)
def test_degenerate_oriented_primitives_do_not_claim_valid_gradients(name, params):
    with pytest.raises(ValueError, match="non-collinear"):
        descriptor(name).backward((3, 2), {}, params, np.ones((2, 3)))
