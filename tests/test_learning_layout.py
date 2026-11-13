"""Semantic layout tests run without a GPU or rasterizing reward geometry."""

import gc
import json
import weakref
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui import param, percent_x, percent_y
from sdf_ui.core.expressions import Expr, evaluate_expr
from sdf_ui.learning.autodiff import Parameter, Tensor
from sdf_ui.learning.layout import Bounds, Element, Layout, LayoutTrainer
from sdf_ui.learning.optimizers import SGD, Adam


def test_bounds_use_full_extents_and_snapshot_reports_exact_geometry():
    a = Element("a", Bounds(-2, 1, 6, 4), role="button", minimum_size=(8, 5))
    b = Element("b", Bounds(2, 3, 5, 4))
    c = Element("c", Bounds(20, 20, 3, 2))
    snapshot = Layout([a, b, c], size=(10, 10)).snapshot()

    assert a.bounds.left == -2
    assert a.bounds.right == 4
    assert a.bounds.top == 1
    assert a.bounds.bottom == 5
    assert a.bounds.center == (1, 3)
    assert snapshot.bounds["a"] == a.bounds
    assert snapshot.overlap_area == 4
    assert snapshot.offscreen_area == 14  # 8 from a, all 6 units of c
    assert snapshot.minimum_size_loss == 5
    assert snapshot.elements["a"].role == "button"


def test_snapshot_is_detached_deeply_immutable_and_json_report_is_a_copy():
    x = Parameter("x", 2)
    width = Parameter("width", 4)
    child = Element(
        "child",
        Bounds(x, 3, width, 6),
        role="button",
        parent="panel",
        minimum_size=(width, 5),
        content_bounds=Bounds(x, 4, 2, 2),
    )
    layout = Layout([Element("panel", Bounds(0, 0, 20, 20)), child], (30, 30))
    snapshot = layout.snapshot()
    report = snapshot.as_dict()

    x.assign(10)
    width.assign(8)
    assert snapshot.bounds["child"].x == 2
    assert snapshot.bounds["child"].width == 4
    assert snapshot.elements["child"].content_bounds.x == 2
    assert snapshot.elements["child"].minimum_size == (4, 5)
    assert layout.snapshot().bounds["child"].x == 10
    with pytest.raises(TypeError):
        snapshot.bounds["new"] = Bounds(0, 0, 1, 1)
    with pytest.raises(TypeError):
        snapshot.elements["child"] = child
    with pytest.raises(FrozenInstanceError):
        snapshot.bounds["child"].x = 100
    with pytest.raises(FrozenInstanceError):
        snapshot.elements["child"].role = "changed"
    with pytest.raises(FrozenInstanceError):
        snapshot.offscreen_area = 99
    report["elements"]["child"]["bounds"]["x"] = 100
    report["size"][0] = 100
    assert snapshot.bounds["child"].x == 2
    assert snapshot.size == (30, 30)
    assert (
        json.loads(json.dumps(snapshot.as_dict()))["elements"]["child"]["parent"]
        == "panel"
    )


def test_parent_containment_is_excluded_but_sibling_overlap_is_counted():
    parent = Element("parent", Bounds(0, 0, 20, 20))
    a = Element("a", Bounds(2, 2, 5, 5), parent=parent)
    b = Element("b", Bounds(4, 4, 5, 5), parent="parent")
    grandchild = Element("nested", Bounds(3, 3, 1, 1), parent=a)
    layout = Layout([parent, a, b, grandchild], (30, 30))

    assert a.parent == "parent"
    assert layout.snapshot().overlap_area == 9
    assert layout.overlap_loss().item() == 9


def test_expr_and_percentage_bounds_resolve_against_each_execution():
    x = Parameter("x", 3)
    layout = Layout(
        [
            Element(
                "button",
                Bounds(
                    param("x") + percent_x(10), "10%y", percent_x(25), percent_y(20)
                ),
            ),
        ],
        (200, 100),
    )

    assert layout.snapshot({"x": x}).bounds["button"] == Bounds(23, 10, 50, 20)
    loss = layout.alignment_loss(["button"], target=30, params={"x": x})
    loss.backward()
    assert x.grad.item() == -14
    x.assign(10)
    assert layout.alignment_loss(["button"], target=30, params={"x": x}).item() == 0
    assert layout.snapshot({"x": x}).bounds["button"].left == 30


def test_persistent_shape_uses_lazy_center_half_extents_and_updated_parameters():
    x = Parameter("x", 4)
    width = Parameter("width", 10)
    element = Element("button", Bounds(x, 3, width, 8))
    shape = element.shape(corner_radius=2)
    values = dict(shape.params)
    ctx = SimpleNamespace(size=(100, 100))

    assert shape.op == "rect"
    assert all(isinstance(v, Expr) for v in values["center"] + values["size"])

    def resolve(key):
        return tuple(evaluate_expr(v, ctx, {}) for v in values[key])

    assert resolve("center") == (9, 7)
    assert resolve("size") == (5, 4)
    x.assign(40)
    width.assign(20)
    assert resolve("center") == (50, 7)
    assert resolve("size") == (10, 4)
    center_x = evaluate_expr(values["center"][0], ctx, {}, track_grad=True)
    center_x.backward()
    assert x.grad.item() == 1
    assert width.grad.item() == 0.5

    named = Element("named", Bounds(param("x"), 3, param("width"), 8)).to_sdf()
    center = dict(named.params)["center"][0]
    assert evaluate_expr(center, ctx, {"x": x, "width": width}) == 50


@pytest.mark.parametrize("symbolic", [False, True])
def test_training_three_buttons_fixes_overlap_offscreen_alignment_and_gaps(symbolic):
    parameters = {}
    elements = []
    for i, (x, y) in enumerate([(-12, 6), (7, 13), (155, 90)]):
        px, py = Parameter(f"x{i}", x), Parameter(f"y{i}", y)
        parameters.update({px.name: px, py.name: py})
        bx, by = (param(px.name), param(py.name)) if symbolic else (px, py)
        elements.append(
            Element(
                f"button{i}",
                Bounds(bx, by, 30, 20),
                role="button",
                minimum_size=(30, 20),
            )
        )
    layout = Layout(elements, size=(160, 80))
    names = list(layout.elements)
    initial = layout.snapshot(parameters)
    history = []

    def loss_fn():
        return (
            0.02 * layout.overlap_loss(parameters)
            + layout.offscreen_loss(parameters)
            + layout.minimum_size_loss(parameters)
            + layout.alignment_loss(names, axis="top", target=30, params=parameters)
            + layout.alignment_loss([names[0]], target=12, params=parameters)
            + layout.gap_loss(names, gap=8, params=parameters)
        )

    def callback(step, loss):
        assert isinstance(step, int) and isinstance(loss, float)
        history.append(layout.snapshot(parameters))

    trainer = LayoutTrainer(parameters, loss_fn, SGD(parameters, lr=0.04))
    losses = trainer.fit(600, callback=callback)
    final = layout.snapshot(parameters)

    assert initial.overlap_area > 0
    assert initial.offscreen_area > 0
    assert losses[-1] < losses[0] * 1e-6
    assert final.overlap_area == 0
    assert final.offscreen_area == 0
    for i, name in enumerate(names):
        bounds = final.bounds[name]
        assert bounds.left == pytest.approx(12 + i * 38, abs=0.02)
        assert bounds.top == pytest.approx(30, abs=0.02)
    assert history[0].bounds[names[0]].x != final.bounds[names[0]].x
    assert initial.bounds[names[0]].x == -12


def test_minimum_and_target_size_losses_train_dimensions():
    width, height = Parameter("width", 3), Parameter("height", 4)
    parameters = [width, height]
    layout = Layout(
        [Element("button", Bounds(5, 5, width, height), minimum_size=(20, 12))],
        (100, 100),
    )
    initial = layout.snapshot()
    trainer = LayoutTrainer(
        parameters,
        lambda: layout.minimum_size_loss() + layout.size_loss({"button": (24, 16)}),
        SGD(parameters, lr=0.1),
    )
    losses = trainer.fit(80)

    assert initial.minimum_size_loss == 353
    assert losses[-1] < 1e-10
    assert layout.snapshot().minimum_size_loss == 0
    assert width.item() == pytest.approx(24, abs=1e-5)
    assert height.item() == pytest.approx(16, abs=1e-5)


def test_geometric_loss_gradients_match_finite_differences():
    p = Parameter("p", -2.5)
    layout = Layout(
        [
            Element("a", Bounds(p, 2, 10, 7)),
            Element("b", Bounds(3, 4, 8, 9)),
        ],
        (40, 40),
    )

    def loss_fn():
        return (
            layout.overlap_loss()
            + layout.offscreen_loss()
            + layout.gap_loss(["a", "b"], 4)
            + layout.alignment_loss(["a", "b"], axis="center_x")
        )

    loss_fn().backward()
    analytic = p.grad.item()
    epsilon = 1e-5
    p.assign(-2.5 + epsilon)
    plus = loss_fn().item()
    p.assign(-2.5 - epsilon)
    minus = loss_fn().item()
    assert analytic == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-6)


def test_nonoverlap_has_no_attraction_but_target_gap_has_useful_gradient():
    x = Parameter("x", 60)
    layout = Layout(
        [Element("a", Bounds(0, 0, 10, 10)), Element("b", Bounds(x, 0, 10, 10))],
        (100, 100),
    )
    layout.overlap_loss().backward()
    assert x.grad.item() == 0
    x.zero_grad()
    layout.gap_loss(["a", "b"], 5).backward()
    assert x.grad.item() > 0  # gradient descent moves b toward the target row


def test_fully_offscreen_element_receives_inward_gradient():
    x = Parameter("x", 30)
    layout = Layout([Element("a", Bounds(x, 1, 5, 5))], (10, 10))
    assert layout.snapshot().offscreen_area == 25
    layout.offscreen_loss().backward()
    assert x.grad.item() == 50


@pytest.mark.parametrize(
    "bounds,expected",
    [
        (Bounds(12.000330946826029, 29.99999999999998, 30, 20), 0),
        (Bounds(-2, -3, 8, 7), 32),  # 14 side-strip + 18 top-strip
        (Bounds(-4, -4, 18, 18), 224),  # spans every viewport edge
        (Bounds(-20, 2, 5, 5), 25),
        (Bounds(2, -20, 5, 5), 25),
        (Bounds(2, 20, 5, 5), 25),
        (Bounds(20, 20, 0, 5), 0),
    ],
)
def test_offscreen_area_counts_strips_stably(bounds, expected):
    layout = Layout([Element("a", bounds)], (10, 10))
    # The first case uses a larger viewport to expose cancellation for an
    # entirely visible box with coordinates produced by gradient descent.
    if expected == 0 and bounds.width == 30:
        layout = Layout([Element("a", bounds)], (160, 80))
    assert layout.snapshot().offscreen_area == expected
    assert layout.offscreen_loss().item() >= 0


def test_vertical_gaps_and_alignment_targets():
    layout = Layout(
        [Element("a", Bounds(3, 2, 8, 10)), Element("b", Bounds(5, 17, 6, 4))], (50, 50)
    )
    assert layout.gap_loss(["a", "b"], 5, axis="y").item() == 0
    assert layout.alignment_loss(["a", "b"], axis="right").item() == 0
    assert layout.alignment_loss(["a", "b"], axis="center_y", target=10).item() == 90


def test_trainer_uses_fresh_loss_resets_gradients_and_retains_no_graphs():
    p = Parameter("p", 4)
    refs = []

    def loss_fn():
        loss = (p - 1) ** 2
        refs.append(weakref.ref(loss))
        return loss

    trainer = LayoutTrainer([p], loss_fn, SGD([p], lr=0.1))
    assert trainer.step() == 9
    assert p.item() == pytest.approx(3.4)
    assert trainer.step() == pytest.approx(5.76)
    assert p.item() == pytest.approx(2.92)
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert trainer.fit(0) == []
    assert isinstance(LayoutTrainer([p], loss_fn).optimizer, Adam)


@pytest.mark.parametrize("optimizer_type", [Adam, SGD])
def test_trainer_rejects_mismatched_optimizer_parameter_identities(optimizer_type):
    p = Parameter("same_name", 1)
    other = Parameter("same_name", 2)
    with pytest.raises(ValueError, match="same parameter identities"):
        LayoutTrainer([p], lambda: p**2, optimizer_type([other]))


def test_trainer_rejects_disconnected_loss_even_with_preexisting_gradients():
    p, other = Parameter("p", 1), Parameter("other", 2)
    (p**2).backward()  # stale gradient must not trick the connectivity check
    (other**2).backward()
    trainer = LayoutTrainer([p], lambda: other**2, SGD([p]))
    with pytest.raises(ValueError, match="at least one trainer parameter"):
        trainer.step()
    assert p.item() == 1


def test_trainer_allows_individually_disconnected_leaves():
    p, other = Parameter("p", 2), Parameter("unused", 3)
    trainer = LayoutTrainer([p, other], lambda: p**2, SGD([p, other], lr=0.1))
    assert trainer.step() == 4
    assert p.item() == pytest.approx(1.6)
    assert other.item() == 3


@pytest.mark.parametrize(
    "loss_fn,error",
    [
        (lambda p: 1.0, TypeError),
        (lambda p: Tensor(1), ValueError),
        (lambda p: p * np.ones(2), ValueError),
    ],
)
def test_trainer_requires_differentiable_scalar_tensor(loss_fn, error):
    p = Parameter("p", 1)
    with pytest.raises(error):
        LayoutTrainer([p], lambda: loss_fn(p)).step()


@pytest.mark.parametrize(
    "bounds",
    [
        (np.nan, 0, 1, 1),
        (0, np.inf, 1, 1),
        (0, 0, -1, 1),
        (0, 0, 1, -1),
        ([1], 0, 1, 1),
        (0, 0, np.inf, 1),
    ],
)
def test_bounds_reject_invalid_numeric_geometry(bounds):
    with pytest.raises(ValueError):
        Bounds(*bounds)


@pytest.mark.parametrize("size", [(0, 10), (-1, 10), (10, np.inf), (10,), ([10], 10)])
def test_layout_rejects_invalid_viewport(size):
    with pytest.raises(ValueError):
        Layout([], size)


def test_symbolic_dimensions_are_validated_at_execution():
    layout = Layout([Element("a", Bounds(0, 0, param("w"), 2))], (10, 10))
    for width in (-1, np.inf, np.nan, [2]):
        with pytest.raises(ValueError):
            layout.snapshot({"w": width})
        with pytest.raises(ValueError):
            layout.offscreen_loss({"w": width})
    assert layout.snapshot({"w": 0}).bounds["a"].area == 0


def test_layout_rejects_duplicate_names_unknown_parents_and_cycles():
    a = Element("a", Bounds(0, 0, 1, 1))
    with pytest.raises(ValueError, match="Duplicate"):
        Layout([a, a], (10, 10))
    with pytest.raises(ValueError, match="Unknown parent"):
        Layout([Element("a", a.bounds, parent="missing")], (10, 10))
    with pytest.raises(ValueError, match="cycle"):
        Layout(
            [Element("a", a.bounds, parent="b"), Element("b", a.bounds, parent="a")],
            (10, 10),
        )


def test_empty_layout_losses_and_snapshot_are_well_defined():
    layout = Layout([], (10, 10))
    assert layout.overlap_loss().item() == 0
    assert layout.offscreen_loss().item() == 0
    assert layout.minimum_size_loss().item() == 0
    assert layout.size_loss({}).item() == 0
    assert layout.alignment_loss([]).item() == 0
    assert layout.gap_loss([], 5).item() == 0
    assert layout.snapshot().offscreen_area == 0


def test_loss_arguments_validate_axes_names_and_target_sizes():
    layout = Layout([Element("a", Bounds(0, 0, 1, 1))], (10, 10))
    with pytest.raises(ValueError):
        layout.alignment_loss(["a"], axis="diagonal")
    with pytest.raises(ValueError):
        layout.gap_loss(["a"], 1, axis="z")
    with pytest.raises(KeyError):
        layout.alignment_loss(["missing"])
    with pytest.raises(ValueError):
        layout.size_loss({"a": (-1, 2)})
    with pytest.raises(ValueError):
        layout.gap_loss(["a"], -1)
