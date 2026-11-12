"""Every registered shader has an executable and auditable derivative contract."""

from pathlib import Path

import numpy as np
import pytest

from sdf_ui.core.plugins.base import ShaderFile
from sdf_ui.core.plugins.registry import registry
from sdf_ui.core.shaders import ShaderLibrary
from sdf_ui.learning import Parameter


def descriptor(name):
    return {shader.name: shader for shader in registry.shader_files()}[name]


def test_all_shader_files_have_contracts_and_executable_rules():
    shaders = {shader.name: shader for shader in registry.shader_files()}
    root = Path(__file__).resolve().parents[1] / "src/sdf_ui"
    assert {
        str(path.relative_to(root)) for path in (root / "core/plugins").rglob("*.glsl")
    } == {shader.path for shader in shaders.values()}
    for name, shader in shaders.items():
        contract = shader.differentiation
        assert contract is not None, name
        assert contract.name == name and contract.path == shader.path
        assert contract.outputs and contract.notes
        assert shader.interface is not None, name
        assert shader.program is not None, name
        assert callable(contract.reference), name
        assert [item.name for item in (shader.interface.output, *shader.interface.extra_outputs)] == [
            item.name for item in contract.outputs
        ]
        assert [item.name for item in shader.interface.inputs] == [
            item.name for item in contract.inputs
        ]
        assert [item.name for item in shader.interface.uniforms] == [
            item.name for item in contract.parameters
        ]
        assert [item.unit for item in (
            shader.interface.output, *shader.interface.extra_outputs, *shader.interface.inputs
        )] == list(range(len(contract.outputs) + len(contract.inputs)))
        assert set(contract.zero_gradient_parameters) <= {
            p.name for p in contract.parameters
        }


def test_circle_backward_reduces_pixels_and_does_not_mutate_callers():
    shader = descriptor("circle")
    radius = Parameter("radius", 2.3)
    radius.grad = np.array(7.0)
    size = (7, 5)
    upstream = np.arange(35.0).reshape(5, 7) / 35
    params = {"offset": [2.3, 1.7], "radius": radius}
    gradients = shader.backward(size, {}, params, upstream)
    y, x = np.indices((5, 7))
    dx, dy = x - 2.3, y - 1.7
    norm = np.hypot(dx, dy)
    np.testing.assert_allclose(
        gradients.parameters["offset"],
        [-np.sum(upstream * dx / norm), -np.sum(upstream * dy / norm)],
    )
    assert gradients.parameters["radius"] == pytest.approx(-upstream.sum())
    assert radius.grad == 7
    assert gradients.inputs == {}
    with pytest.raises(TypeError):
        gradients.parameters["radius"] = 1
    with pytest.raises(ValueError):
        gradients.parameters["offset"][0] = 1


def test_contract_evaluation_preserves_existing_parameter_graph():
    radius = Parameter("radius", 3.0)
    derived = radius * 2
    field = descriptor("circle").differentiation.evaluate(
        (4, 3), {}, {"offset": [1.3, 1.6], "radius": derived}
    )
    field.mean().backward()
    assert radius.grad == pytest.approx(-2)


def test_masked_union_accumulates_both_output_seeds():
    size, shape = (3, 2), (2, 3)
    inputs = {"sdf0": np.full(shape, -0.3), "sdf1": np.full(shape, 0.1)}
    mask_seed = np.zeros(shape + (4,))
    mask_seed[..., 0] = 2
    gradient = descriptor("masked_union").backward(
        size, inputs, {}, {"destTex": np.full(shape, 3.0), "maskTex": mask_seed}
    )
    np.testing.assert_allclose(gradient.inputs["sdf0"], 3 - 2 * 6 * 0.4 * 0.6)
    np.testing.assert_allclose(gradient.inputs["sdf1"], 2 * 6 * 0.4 * 0.6)


def test_blur_vjp_is_transpose_of_actual_border_stencil():
    rng = np.random.default_rng(19)
    size, shape = (4, 3), (3, 4, 4)
    original, upstream = rng.normal(size=shape), rng.normal(size=shape)
    shader = descriptor("blur_hor_9")
    output = shader.differentiation.evaluate(size, {"origTex": original})
    gradients = shader.backward(size, {"origTex": original}, {}, upstream)
    assert np.sum(output.numpy() * upstream) == pytest.approx(
        np.sum(original * gradients.inputs["origTex"]), abs=1e-12
    )
    np.testing.assert_array_equal(gradients.parameters["offset"], 0)
    assert "offset" in shader.differentiation.zero_gradient_parameters
    assert shader.differentiation.sampling == "asymmetric_reflected_zero_outside"


@pytest.mark.parametrize("name,expected", [("dither_1bit", 0), ("dithering", 1)])
def test_hard_threshold_and_additive_dither_have_distinct_gradients(name, expected):
    inputs = {"origTex": np.full((2, 3, 4), 0.514)}
    gradients = descriptor(name).backward((3, 2), inputs, {}, np.ones((2, 3, 4)))
    np.testing.assert_array_equal(gradients.inputs["origTex"], expected)
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


def test_shader_library_exposes_contract_without_compiling_gl():
    library = ShaderLibrary(None)
    assert library.differentiation("circle") is descriptor("circle").differentiation
    with pytest.raises(KeyError, match="Unknown shader"):
        library.differentiation("unknown")
    unknown = ShaderFile("custom", "custom.glsl")
    assert unknown.differentiation is None
    with pytest.raises(ValueError, match="no differentiation"):
        unknown.backward((3, 2), {}, {}, np.ones((2, 3)))
    # Names and paths do not infer contracts: ownership must be explicit.
    assert (
        ShaderFile("circle", "core/plugins/primitives/rect/shader.glsl").differentiation
        is None
    )


def test_ports_and_uniform_shapes_are_validated_before_backward():
    contract = descriptor("circle").differentiation
    for params in ({"offset": [1.0]}, {"typo": 3.0}):
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
