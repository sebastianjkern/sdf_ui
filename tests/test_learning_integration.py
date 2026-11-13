"""Learning values remain usable by the existing expression/GPU APIs."""

from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui import Canvas, cos, param, percent_x, sdf, sin
from sdf_ui.core.expressions import evaluate_expr
from sdf_ui.core.texture import Renderer
from sdf_ui.learning.autodiff import Parameter
from sdf_ui.learning.layout import Bounds, Element
from sdf_ui.learning.optimizers import Adam
from sdf_ui.learning.parameters import ParameterStore


def test_reusable_expression_rebuilds_tape_after_parameter_update():
    parameters = ParameterStore()
    radius = parameters.define("radius", 2.0)
    expression = (param("radius") + 1.0) ** 2
    optimizer = Adam(parameters.parameters(), lr=0.1)
    context = SimpleNamespace(size=(32, 32))
    first = evaluate_expr(expression, context, parameters, track_grad=True).item()
    for _ in range(10):
        optimizer.zero_grad()
        loss = evaluate_expr(expression, context, parameters, track_grad=True)
        loss.backward()
        optimizer.step()
    assert loss.item() < first
    assert radius.item() < 2.0
    assert isinstance(evaluate_expr(expression, context, parameters), float)


def test_expression_gradients_and_gpu_uniform_resolution():
    value = Parameter("angle", 0.4)
    context = SimpleNamespace(
        size=(20, 10),
        percent=lambda x: x * 0.2,
        percent_x=lambda x: x * 0.2,
        percent_y=lambda x: x * 0.1,
        percent_of_min=lambda x: x * 0.1,
    )
    expression = sin(param("angle")) + cos(param("angle")) + percent_x(50)
    output = evaluate_expr(expression, context, {"angle": value}, track_grad=True)
    output.backward()
    np.testing.assert_allclose(value.grad, np.cos(0.4) - np.sin(0.4))
    renderer = Renderer(context, params={"angle": value})
    assert renderer._resolve((param("angle"), value)) == (0.4, 0.4)


@pytest.mark.parametrize("operation", ["relu", "softplus", "sigmoid", "tanh", "sqrt"])
def test_expression_nonlinearity_display_and_learning_agree(operation):
    value = Parameter("value", 0.7)
    expression = getattr(param("value"), operation)()
    context = SimpleNamespace(size=(20, 10))
    numeric = evaluate_expr(expression, context, {"value": value})
    differentiable = evaluate_expr(
        expression, context, {"value": value}, track_grad=True
    )
    assert numeric == pytest.approx(differentiable.item())
    differentiable.backward()
    assert np.isfinite(value.grad)


def test_parameter_store_checkpoint_validation_is_atomic(tmp_path):
    parameters = ParameterStore([Parameter("x", 1.0), Parameter("sizes", [2.0, 3.0])])
    checkpoint = tmp_path / "weights.json"
    parameters.save(checkpoint)
    parameters["x"].assign(9.0)
    parameters.load(checkpoint)
    assert parameters["x"].item() == 1.0
    invalid = parameters.state_dict()
    invalid["parameters"]["sizes"] = [1.0]
    with pytest.raises(ValueError, match="shape"):
        parameters.load_state_dict(invalid)
    assert parameters["x"].item() == 1.0
    invalid = parameters.state_dict()
    invalid["parameters"]["x"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        parameters.load_state_dict(invalid)
    with pytest.raises(ValueError, match="Duplicate"):
        parameters.define("x", 3.0)


def test_gpu_render_accepts_updated_trainable_parameters_and_float_backend():
    radius = Parameter("radius", 3.0)
    scene = sdf.circle((8, 8), param("radius"))
    with Canvas((16, 16)) as canvas:
        first = canvas.render(scene, params={"radius": radius})
        first_values = np.frombuffer(first.tex.read(), dtype=np.float32).copy()
        radius.assign(5.0)
        second = canvas.render(scene, params={"radius": radius})
        second_values = np.frombuffer(second.tex.read(), dtype=np.float32)
        np.testing.assert_allclose(second_values, first_values - 2.0, atol=0.03)
        floating = canvas.differentiate(scene, params={"radius": radius})
        floating.mean().backward()
        assert radius.grad == pytest.approx(-1.0)


def test_semantic_element_reuses_learned_bounds_in_gpu_and_float_renderer():
    x = Parameter("x", 3.0)
    scene = Element("button", Bounds(x, 4, 20, 12)).shape(corner_radius=3)
    with Canvas((40, 24)) as canvas:
        initial = canvas.render(scene)
        initial_values = np.frombuffer(initial.tex.read(), dtype=np.float32).reshape(
            24, 40
        )
        x.assign(8.0)
        moved = canvas.render(scene)
        moved_values = np.frombuffer(moved.tex.read(), dtype=np.float32).reshape(24, 40)
        floating = canvas.differentiate(scene)
        assert not np.array_equal(initial_values, moved_values)
        np.testing.assert_allclose(moved_values, floating.numpy(), atol=0.04)
