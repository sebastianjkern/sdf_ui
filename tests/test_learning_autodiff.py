"""Numerical derivatives, tape lifetime, and actual optimizer behavior."""

import json

import numpy as np
import pytest

from sdf_ui.learning import SGD, Adam, Parameter, Tensor, maximum, minimum, stack, where


def test_fixed_parameter_stays_out_of_autodiff_and_optimizers():
    trainable = Parameter("trainable", 2.0)
    fixed = Parameter("fixed", 3.0, trainable=False)
    loss = trainable * fixed
    loss.backward()
    assert trainable.grad.item() == 3.0
    assert fixed.grad is None
    with pytest.raises(ValueError, match="trainable parameters only"):
        SGD([fixed], lr=0.1)


def check_gradient(fn, values, tolerance=2e-5):
    parameters = [Parameter(f"p{i}", value) for i, value in enumerate(values)]
    output = fn(*parameters)
    output.backward()
    analytic = [parameter.grad.copy() for parameter in parameters]
    epsilon = 1e-5
    for parameter, expected in zip(parameters, analytic):
        original = parameter.numpy()
        numeric = np.zeros_like(original)
        for index in np.ndindex(original.shape):
            positive, negative = original.copy(), original.copy()
            positive[index] += epsilon
            negative[index] -= epsilon
            parameter.assign(positive)
            high = fn(*parameters).item()
            parameter.assign(negative)
            low = fn(*parameters).item()
            parameter.assign(original)
            numeric[index] = (high - low) / (2 * epsilon)
        np.testing.assert_allclose(expected, numeric, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x, y: ((x + y) * (x - y)).mean(),
        lambda x, y: (x / y).sum(),
        lambda x, y: (x**y).mean(),
        lambda x, y: maximum(x, y).sum() + minimum(x, y).mean(),
    ],
)
def test_binary_operations_and_broadcast_reductions(operation):
    check_gradient(
        operation,
        [np.array([[0.7, 1.4, 2.2], [1.1, 2.6, 3.4]]), np.array([1.2, 1.8, 2.5])],
    )


@pytest.mark.parametrize(
    "method",
    [
        "relu",
        "softplus",
        "sigmoid",
        "tanh",
        "log",
        "exp",
        "sqrt",
        "abs",
        "sin",
        "cos",
    ],
)
def test_unary_gradients(method):
    check_gradient(lambda x: getattr(x, method)().mean(), [np.array([0.3, 0.8, 1.7])])


@pytest.mark.parametrize(
    "shapes", [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 2)), ((2, 3), (3, 2))]
)
def test_matrix_and_vector_products(shapes):
    rng = np.random.default_rng(4)
    check_gradient(
        lambda a, b: ((a @ b) ** 2).sum(), [rng.normal(size=shape) for shape in shapes]
    )


def test_shared_graph_paths_accumulate_and_backward_uses_fresh_adjoints():
    p = Parameter("p", [2.0, 3.0])
    shared = p * p
    loss = (shared + shared * 2).sum()
    loss.backward()
    np.testing.assert_array_equal(p.grad, [12.0, 18.0])
    loss.backward()
    np.testing.assert_array_equal(p.grad, [24.0, 36.0])
    p.zero_grad()
    loss.backward()
    np.testing.assert_array_equal(p.grad, [12.0, 18.0])


def test_repeated_gather_and_mutable_indices_are_recorded_safely():
    p = Parameter("p", [2.0, 3.0, 4.0])
    indices = np.array([0, 0, 2])
    selected = p[indices]
    indices[:] = 1
    selected.sum().backward()
    np.testing.assert_array_equal(p.grad, [2.0, 0.0, 1.0])


def test_transpose_reshape_stack_and_reduction_axes():
    check_gradient(
        lambda x: (
            stack((x.T.reshape(3, 2), x.reshape(3, 2)), axis=0)
            .mean(axis=(0, 2), keepdims=True)
            .sum()
        ),
        [np.arange(6.0).reshape(2, 3)],
    )


def test_where_broadcasts_gradients_to_only_selected_branches():
    check_gradient(
        lambda x, y: where(np.array([[True], [False]]), x, y).sum(),
        [np.array([1.0, 2.0, 3.0]), np.array([[4.0], [5.0]])],
    )


def test_origin_subgradient_conventions_are_finite():
    p = Parameter("p", 0.0)
    (p.sqrt() + p.abs() + p.relu()).backward()
    assert p.grad == 0
    p.zero_grad()
    (maximum(p, 0.0) + minimum(p, 0.0)).backward()
    assert p.grad == 1


def test_parameter_mutation_invalidates_old_graph_without_changing_gradients():
    p = Parameter("p", 2.0)
    loss = p**2
    version = p.version
    p.assign(3.0)
    assert p.version == version + 1
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert p.grad is None
    with pytest.raises(RuntimeError, match="stale"):
        _ = loss + 1
    (p**2).backward()
    assert p.grad == 6


def test_tensor_data_access_cannot_mutate_live_parameters():
    p = Parameter("p", [1.0, 2.0])
    original = p.numpy()
    for value in (p.data, p.value, p.numpy()):
        value[:] = 9
    np.testing.assert_array_equal(p.numpy(), original)
    with pytest.raises(ValueError, match="shape"):
        p.assign([1.0])


def test_explicit_output_gradient_and_detach():
    p = Parameter("p", [2.0, 3.0])
    output = p * p
    with pytest.raises(ValueError, match="explicit"):
        output.backward()
    output.backward([2.0, 4.0])
    np.testing.assert_array_equal(p.grad, [8.0, 24.0])
    with pytest.raises(RuntimeError, match="detached"):
        output.detach().sum().backward()
    with pytest.raises(ValueError, match="shape"):
        output.backward([1.0])


@pytest.mark.parametrize("value", [np.nan, np.inf, [1, -np.inf]])
def test_nonfinite_input_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        Tensor(value)


@pytest.mark.parametrize(
    "operation", [lambda p: p / 0, lambda p: (-p).log(), lambda p: (p * 1e308).exp()]
)
def test_nonfinite_forward_is_rejected(operation):
    with pytest.raises(FloatingPointError, match="nonfinite"):
        operation(Parameter("p", 2.0))


@pytest.mark.parametrize(
    "optimizer_type,kwargs", [(SGD, {"lr": 0.05, "momentum": 0.8}), (Adam, {"lr": 0.1})]
)
def test_optimizer_convergence_and_exact_json_checkpoint_resume(optimizer_type, kwargs):
    p = Parameter("p", [4.0, -3.0])
    optimizer = optimizer_type([p], **kwargs)

    def update(parameter, opt):
        opt.zero_grad()
        loss = ((parameter - np.array([1.0, 2.0])) ** 2).mean()
        loss.backward()
        opt.step()

    for _ in range(100):
        update(p, optimizer)
    state = json.loads(json.dumps(optimizer.state_dict()))
    resumed_p = Parameter("p", p.numpy())
    resumed = optimizer_type([resumed_p], **kwargs)
    resumed.load_state_dict(state)
    for _ in range(100):
        update(p, optimizer)
        update(resumed_p, resumed)
    np.testing.assert_array_equal(p.numpy(), resumed_p.numpy())
    np.testing.assert_allclose(p.numpy(), [1.0, 2.0], atol=0.002)


@pytest.mark.parametrize("optimizer_type", [SGD, Adam])
def test_optimizer_nonfinite_update_is_atomic(optimizer_type):
    a, b = Parameter("a", 1.0), Parameter("b", 2.0)
    optimizer = optimizer_type([a, b])
    a.grad, b.grad = np.array(1.0), np.array(np.inf)
    state = optimizer.state_dict()
    with pytest.raises(ValueError, match="finite"):
        optimizer.step()
    assert a.item() == 1.0 and b.item() == 2.0
    assert optimizer.state_dict() == state


@pytest.mark.parametrize("optimizer_type", [SGD, Adam])
def test_optimizer_requires_gradients_and_unique_parameter_names(optimizer_type):
    p = Parameter("p", 1.0)
    with pytest.raises(ValueError, match="unique"):
        optimizer_type([p, Parameter("p", 2.0)])
    with pytest.raises(RuntimeError, match="gradients"):
        optimizer_type([p]).step()


def test_invalid_optimizer_checkpoint_does_not_change_state():
    p = Parameter("p", [1.0, 2.0])
    optimizer = Adam([p])
    (p**2).sum().backward()
    optimizer.step()
    original = optimizer.state_dict()
    invalid = json.loads(json.dumps(original))
    invalid["state"]["p"]["exp_avg_sq"] = [-1.0, 0.0]
    with pytest.raises(ValueError, match="variance"):
        optimizer.load_state_dict(invalid)
    assert optimizer.state_dict() == original
