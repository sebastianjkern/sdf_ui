"""Semantic, CPU-only layout geometry and direct gradient-based training.

Coordinates use a top-left origin and full widths/heights. Bounds may contain
numbers, Tensor leaves, or reusable ``core.expressions.Expr`` objects. For a
persistent layout, declare derived geometry with Expr arithmetic, e.g.
``Bounds(param('x'), 20, param('width'), 30)`` and pass its parameter mapping to
each loss/snapshot. Direct Parameter leaves also work. Eager derived Tensors
(such as ``parameter + 10``) must be rebuilt inside the loss callback: their
forward values belong to the execution that created them.

Losses resolve expressions afresh and preserve gradients. Snapshots detach all
geometry into immutable numeric records suitable for reward callbacks; neither
path creates a GPU context. A shape shares the original bounds expressions,
with lazy center and half-size calculations for the existing SDF rect API.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..core.expressions import Expr, evaluate_expr
from .autodiff import Parameter, Tensor, as_tensor, maximum, minimum
from .optimizers import Adam

__all__ = ["Bounds", "Element", "Layout", "LayoutSnapshot", "LayoutTrainer"]


def _number(value, label):
    """Inspect scalar geometry without changing the differentiable value."""
    if isinstance(value, Tensor):
        value = value.numpy()
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite scalar") from exc
    if array.ndim != 0 or not np.isfinite(array).item():
        raise ValueError(f"{label} must be a finite scalar")
    return float(array)


def _validate_scalar(value, label, nonnegative=False):
    # Expr and percentage strings need a viewport/parameter mapping first.
    if isinstance(value, (Expr, str)):
        return
    numeric = _number(value, label)
    if nonnegative and numeric < 0:
        raise ValueError(f"{label} must be nonnegative")


def _pair(value, label):
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must contain two scalars") from exc
    if len(pair) != 2:
        raise ValueError(f"{label} must contain two scalars")
    return pair


@dataclass(frozen=True)
class Bounds:
    """An axis-aligned box with full width and height (zero sizes are valid).

    Literal and Tensor coordinates must be finite scalars; sizes must also be
    nonnegative. Symbolic values are validated when a Layout resolves them.
    ``center`` is the ``(center_x, center_y)`` pair. Properties use ordinary
    arithmetic, so keep derived Tensor properties inside a fresh loss callback
    or use Expr leaves when storing them for later executions.
    """

    x: object
    y: object
    width: object
    height: object

    def __post_init__(self):
        for name in ("x", "y", "width", "height"):
            _validate_scalar(getattr(self, name), name, name in {"width", "height"})

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def center_x(self):
        return self.x + self.width / 2

    @property
    def center_y(self):
        return self.y + self.height / 2

    @property
    def center(self):
        return self.center_x, self.center_y

    @property
    def area(self):
        return self.width * self.height


@dataclass(frozen=True)
class Element:
    """A named semantic element; all bounds use absolute layout coordinates.

    ``parent`` is an Element or its name and is stored as a name. Parent-child
    containment is intentional, so ancestors are excluded from overlap pairs.
    ``minimum_size`` is an optional (width, height) pair. ``content_bounds``
    describes content geometry for rewards; it does not impose a constraint.
    """

    name: str
    bounds: Bounds
    role: str = "element"
    parent: object = None
    minimum_size: object = None
    content_bounds: object = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Element name must be a nonempty string")
        if not isinstance(self.bounds, Bounds):
            raise TypeError("Element bounds must be Bounds")
        if not isinstance(self.role, str):
            raise TypeError("Element role must be a string")
        if isinstance(self.parent, Element):
            object.__setattr__(self, "parent", self.parent.name)
        if self.parent is not None and not isinstance(self.parent, str):
            raise TypeError("Element parent must be an Element or name")
        if self.minimum_size is not None:
            pair = _pair(self.minimum_size, "minimum_size")
            for value in pair:
                _validate_scalar(value, "minimum_size", nonnegative=True)
            object.__setattr__(self, "minimum_size", pair)
        if self.content_bounds is not None and not isinstance(
            self.content_bounds, Bounds
        ):
            raise TypeError("content_bounds must be Bounds")

    def shape(self, corner_radius=0.0):
        """Build a persistent SDF rect sharing these raw geometry leaves.

        Lazy Expr wrapping happens *before* center/half-extent arithmetic, so
        direct Parameter bounds update an already built scene. Precomputed
        eager Tensor expressions cannot be refreshed; declare those as Expr.
        ``corner_radius`` follows the SDF API (a scalar or four radii).
        """
        from .. import sdf

        x, y, width, height = (
            Expr("add", (value, 0.0))
            for value in (
                self.bounds.x,
                self.bounds.y,
                self.bounds.width,
                self.bounds.height,
            )
        )
        return sdf.rect(
            center=(x + width / 2, y + height / 2),
            size=(width / 2, height / 2),
            corner_radius=corner_radius,
        )

    def to_sdf(self, corner_radius=0.0):
        """Alias for :meth:`shape`."""
        return self.shape(corner_radius=corner_radius)


@dataclass(frozen=True)
class _LayoutContext:
    size: tuple
    inputs: Mapping = field(default_factory=dict)

    def percent(self, value):
        return self.percent_x(value)

    def percent_x(self, value):
        return value * self.size[0] / 100

    def percent_y(self, value):
        return value * self.size[1] / 100

    def percent_of_min(self, value):
        return value * min(self.size) / 100


def _resolve(value, ctx, params, track_grad, label, nonnegative=False):
    result = evaluate_expr(value, ctx, params, track_grad=track_grad)
    numeric = _number(result, label)
    if nonnegative and numeric < 0:
        raise ValueError(f"{label} must be nonnegative")
    return as_tensor(result) if track_grad else numeric


def _resolve_bounds(bounds, ctx, params, track_grad):
    return Bounds(
        *(
            _resolve(
                getattr(bounds, name),
                ctx,
                params,
                track_grad,
                name,
                nonnegative=name in {"width", "height"},
            )
            for name in ("x", "y", "width", "height")
        )
    )


def _overlap(a, b):
    width = (minimum(a.right, b.right) - maximum(a.left, b.left)).relu()
    height = (minimum(a.bottom, b.bottom) - maximum(a.top, b.top)).relu()
    return width * height


def _outside_length(start, length, limit):
    # Each edge spill is bounded by the element's length, including when the
    # element lies wholly outside. A fully inside interval gives exact zero.
    return min(
        length,
        min(length, max(0.0, -start)) + min(length, max(0.0, start + length - limit)),
    )


def _validated_parameters(parameters):
    source = parameters.values() if isinstance(parameters, Mapping) else parameters
    values = tuple(source)
    if not values or not all(isinstance(value, Parameter) for value in values):
        raise TypeError("parameters must be a nonempty collection of Parameters")
    if len({id(value) for value in values}) != len(values):
        raise ValueError("parameters must not contain duplicates")
    return values


def _serialize_layout_identity(identity):
    if identity is None:
        return None
    try:
        return json.dumps(
            identity, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Layout identity must be finite JSON-serializable data"
        ) from error


def _validate_layout_checkpoint(trainer, state):
    if (
        not isinstance(state, Mapping)
        or state.get("format_version") != 2
        or "parameter_schema" not in state
    ):
        raise ValueError("Invalid layout trainer checkpoint")
    if trainer.identity is None or state.get("identity") != trainer.identity:
        raise ValueError("Checkpoint layout loss identity differs")
    schema = (
        None if trainer._parameter_store is None else trainer._parameter_store.schema()
    )
    if state.get("parameter_schema") != schema:
        raise ValueError("Checkpoint parameter schema differs")


def _validated_checkpoint_parameters(parameters, values, steps):
    if (
        not isinstance(values, Mapping)
        or set(values) != {parameter.name for parameter in parameters}
        or len(values) != len(parameters)
    ):
        raise ValueError("Checkpoint parameter names differ")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Invalid checkpoint step count")
    checked = {}
    for parameter in parameters:
        value = np.asarray(values[parameter.name], dtype=float)
        if value.shape != parameter.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint parameter '{parameter.name}'")
        checked[parameter.name] = value
    return checked


def _validate_parent_forest(elements):
    for element in elements.values():
        seen = {element.name}
        parent = element.parent
        while parent is not None:
            if parent not in elements:
                raise ValueError(f"Unknown parent: {parent}")
            if parent in seen:
                raise ValueError("Element parents must not form a cycle")
            seen.add(parent)
            parent = elements[parent].parent


@dataclass(frozen=True)
class LayoutSnapshot:
    """Detached reward input with read-only mappings and frozen numeric bounds.

    ``overlap_area`` sums pairwise intersection areas (excluding ancestors).
    ``offscreen_area`` sums each element's area outside the viewport, including
    fully offscreen elements. These are sums, not union areas. All values are
    in layout units; ``minimum_size_loss`` is a squared size deficit.
    """

    elements: Mapping
    size: tuple
    overlap_area: float
    offscreen_area: float
    minimum_size_loss: float
    bounds: Mapping = field(init=False)

    def __post_init__(self):
        elements = dict(self.elements)
        object.__setattr__(self, "elements", MappingProxyType(elements))
        object.__setattr__(
            self, "bounds", MappingProxyType({k: v.bounds for k, v in elements.items()})
        )
        object.__setattr__(self, "size", tuple(self.size))

    def as_dict(self):
        """Return a mutable, JSON-compatible copy for reporting/checkpoints."""

        def box(bounds):
            return {key: getattr(bounds, key) for key in ("x", "y", "width", "height")}

        return {
            "size": list(self.size),
            "elements": {
                name: {
                    "bounds": box(element.bounds),
                    "role": element.role,
                    "parent": element.parent,
                    "minimum_size": element.minimum_size,
                    "content_bounds": (
                        box(element.content_bounds)
                        if element.content_bounds is not None
                        else None
                    ),
                }
                for name, element in self.elements.items()
            },
            "overlap_area": self.overlap_area,
            "offscreen_area": self.offscreen_area,
            "minimum_size_loss": self.minimum_size_loss,
        }


@dataclass(frozen=True)
class Layout:
    """A named element collection in a fixed, finite, positive viewport.

    Accepts an iterable of Elements or a name-to-Element mapping. Names must
    be unique and parent references must form a forest within this layout.
    Loss methods return scalar Tensors, including zero for an empty layout.
    """

    elements: object
    size: tuple
    inputs: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.inputs, Mapping):
            raise TypeError("Layout inputs must be a mapping")
        elements = self._validate_layout_elements(self.elements)
        size = tuple(_number(v, "viewport size") for v in _pair(self.size, "size"))
        if any(v <= 0 for v in size):
            raise ValueError("Viewport size must be positive")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "elements", MappingProxyType(elements))

    @staticmethod
    def _validate_layout_elements(source):
        source = source.values() if isinstance(source, Mapping) else source
        elements = {}
        for element in source:
            if not isinstance(element, Element):
                raise TypeError("Layout accepts Element instances")
            if element.name in elements:
                raise ValueError(f"Duplicate element name: {element.name}")
            elements[element.name] = element
        _validate_parent_forest(elements)
        return elements

    def _pairs(self):
        ancestors = {}
        for name, element in self.elements.items():
            ancestors[name] = set()
            parent = element.parent
            while parent is not None:
                ancestors[name].add(parent)
                parent = self.elements[parent].parent
        return (
            (a, b)
            for a, b in combinations(self.elements, 2)
            if a not in ancestors[b] and b not in ancestors[a]
        )

    def _bounds(self, params):
        ctx = _LayoutContext(self.size, self.inputs)
        return {
            name: _resolve_bounds(element.bounds, ctx, params, True)
            for name, element in self.elements.items()
        }

    def snapshot(self, params=None):
        """Validate and freeze numeric geometry without retaining a Tensor graph."""
        params = {} if params is None else params
        ctx = _LayoutContext(self.size, self.inputs)
        elements = {}
        for name, element in self.elements.items():
            minimum_size = (
                tuple(
                    _resolve(v, ctx, params, False, "minimum_size", True)
                    for v in element.minimum_size
                )
                if element.minimum_size is not None
                else None
            )
            elements[name] = Element(
                name,
                _resolve_bounds(element.bounds, ctx, params, False),
                role=element.role,
                parent=element.parent,
                minimum_size=minimum_size,
                content_bounds=(
                    _resolve_bounds(element.content_bounds, ctx, params, False)
                    if element.content_bounds is not None
                    else None
                ),
            )
        bounds = {name: element.bounds for name, element in elements.items()}
        overlap_area = sum(
            max(
                0.0,
                min(bounds[a].right, bounds[b].right)
                - max(bounds[a].left, bounds[b].left),
            )
            * max(
                0.0,
                min(bounds[a].bottom, bounds[b].bottom)
                - max(bounds[a].top, bounds[b].top),
            )
            for a, b in self._pairs()
        )
        offscreen_area = 0.0
        minimum_size_loss = 0.0
        for element in elements.values():
            b = element.bounds
            outside_width = _outside_length(b.x, b.width, self.size[0])
            outside_height = _outside_length(b.y, b.height, self.size[1])
            # Count full-height side strips, then top/bottom strips across the
            # remaining width. Avoid area-minus-visible-area cancellation.
            offscreen_area += outside_width * b.height
            offscreen_area += (b.width - outside_width) * outside_height
            if element.minimum_size is not None:
                mw, mh = element.minimum_size
                minimum_size_loss += max(0.0, mw - b.width) ** 2
                minimum_size_loss += max(0.0, mh - b.height) ** 2
        return LayoutSnapshot(
            elements,
            self.size,
            float(overlap_area),
            float(offscreen_area),
            float(minimum_size_loss),
        )

    def overlap_loss(self, params=None):
        """Sum pairwise overlap areas; gradients vanish for separated boxes.

        Combine with alignment and target-gap losses to attract distant boxes
        into a useful arrangement rather than relying on non-overlap alone.
        Ancestor/descendant pairs are excluded; siblings are included.
        """
        bounds = self._bounds({} if params is None else params)
        return sum(
            (_overlap(bounds[a], bounds[b]) for a, b in self._pairs()), as_tensor(0.0)
        )

    def offscreen_loss(self, params=None):
        """Sum squared edge overflow, giving gradients even fully offscreen.

        This is a distance penalty, not the snapshot's clipped area metric.
        """
        bounds = self._bounds({} if params is None else params)
        result = as_tensor(0.0)
        for b in bounds.values():
            for overflow in (
                -b.left,
                -b.top,
                b.right - self.size[0],
                b.bottom - self.size[1],
            ):
                result = result + overflow.relu() ** 2
        return result

    def minimum_size_loss(self, params=None):
        """Sum squared width/height deficits against declared minimum sizes."""
        params = {} if params is None else params
        bounds = self._bounds(params)
        ctx = _LayoutContext(self.size, self.inputs)
        result = as_tensor(0.0)
        for name, element in self.elements.items():
            if element.minimum_size is None:
                continue
            mw, mh = (
                _resolve(v, ctx, params, True, "minimum_size", True)
                for v in element.minimum_size
            )
            b = bounds[name]
            result = result + (mw - b.width).relu() ** 2
            result = result + (mh - b.height).relu() ** 2
        return result

    @staticmethod
    def _containment(inner, outer):
        return sum(
            (
                edge.relu() ** 2
                for edge in (
                    outer.left - inner.left,
                    outer.top - inner.top,
                    inner.right - outer.right,
                    inner.bottom - outer.bottom,
                )
            ),
            as_tensor(0.0),
        )

    def containment_loss(self, params=None):
        """Squared edge overflow for children outside their immediate parents."""
        bounds = self._bounds({} if params is None else params)
        return sum(
            (
                self._containment(bounds[name], bounds[element.parent])
                for name, element in self.elements.items()
                if element.parent is not None
            ),
            as_tensor(0.0),
        )

    def content_overflow_loss(self, params=None):
        """Squared edge overflow of declared content outside its element."""
        params = {} if params is None else params
        bounds = self._bounds(params)
        ctx = _LayoutContext(self.size, self.inputs)
        return sum(
            (
                self._containment(
                    _resolve_bounds(element.content_bounds, ctx, params, True),
                    bounds[name],
                )
                for name, element in self.elements.items()
                if element.content_bounds is not None
            ),
            as_tensor(0.0),
        )

    def violations(self, params=None, *, tolerance=0.0):
        """Return measured constraint violations exceeding ``tolerance``.

        Losses are soft constraints; callers can use this method to reject a
        candidate before presenting it. Tolerance is in each metric's units.
        """
        tolerance = _number(tolerance, "tolerance")
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        snapshot = self.snapshot(params)
        metrics = {
            "overlap_area": snapshot.overlap_area,
            "offscreen_area": snapshot.offscreen_area,
            # Area alone misses zero-sized bounds positioned outside the page.
            "offscreen_edge_loss": self.offscreen_loss(params).item(),
            "minimum_size_loss": snapshot.minimum_size_loss,
            "containment_loss": self.containment_loss(params).item(),
            "content_overflow_loss": self.content_overflow_loss(params).item(),
        }
        return {name: value for name, value in metrics.items() if value > tolerance}

    def prevent_overlap(self, params=None):
        """Return a soft overlap penalty; use ``assert_valid`` for validation."""
        return self.overlap_loss(params)

    def keep_inside_parent(self, params=None):
        """Return a soft parent-containment penalty."""
        return self.containment_loss(params)

    def is_valid(self, params=None, *, tolerance=0.0):
        """Check the measured geometric constraints without updating the layout."""
        return not self.violations(params, tolerance=tolerance)

    def assert_valid(self, params=None, *, tolerance=0.0):
        """Reject a layout with measured violations above the given tolerance."""
        violations = self.violations(params, tolerance=tolerance)
        if violations:
            raise ValueError(f"Layout constraint violations: {violations}")

    def size_loss(self, targets, params=None):
        """Sum squared size errors for ``{name: (width, height)}`` targets."""
        params = {} if params is None else params
        bounds = self._bounds(params)
        ctx = _LayoutContext(self.size, self.inputs)
        result = as_tensor(0.0)
        for name, target in targets.items():
            width, height = (
                _resolve(v, ctx, params, True, "target size", True)
                for v in _pair(target, "target size")
            )
            b = bounds[name]
            result = result + (b.width - width) ** 2 + (b.height - height) ** 2
        return result

    def alignment_loss(self, names, axis="left", target=None, params=None):
        """Sum squared coordinate errors against a target, or the group mean.

        ``axis`` is left/right/top/bottom/center_x/center_y. An explicit target
        anchors the group; mean alignment alone permits translation.
        """
        if axis not in {"left", "right", "top", "bottom", "center_x", "center_y"}:
            raise ValueError("Unknown alignment axis")
        params = {} if params is None else params
        bounds = self._bounds(params)
        values = [getattr(bounds[name], axis) for name in names]
        if not values:
            return as_tensor(0.0)
        coordinate = (
            sum(values, as_tensor(0.0)) / len(values)
            if target is None
            else _resolve(
                target, _LayoutContext(self.size, self.inputs), params, True, "target"
            )
        )
        return sum(((value - coordinate) ** 2 for value in values), as_tensor(0.0))

    def gap_loss(self, names, gap, axis="x", params=None):
        """Sum squared errors of consecutive ordered edge gaps to ``gap``.

        For x: next.left - previous.right; for y: next.top - previous.bottom.
        Name order is intentional; elements are never sorted by position.
        Targets must be nonnegative. Fewer than two names give zero.
        """
        if axis not in {"x", "y"}:
            raise ValueError("Gap axis must be 'x' or 'y'")
        params = {} if params is None else params
        bounds = self._bounds(params)
        ordered = [bounds[name] for name in names]
        target = _resolve(
            gap, _LayoutContext(self.size, self.inputs), params, True, "gap", True
        )
        result = as_tensor(0.0)
        for before, after in zip(ordered, ordered[1:]):
            distance = (
                after.left - before.right if axis == "x" else after.top - before.bottom
            )
            result = result + (distance - target) ** 2
        return result


class LayoutTrainer:
    """Optimize a scalar Tensor loss built fresh by ``loss_fn()`` each step.

    Parameters may be an iterable or a mapping of Parameter objects. The
    default optimizer is Adam; custom optimizers implement zero_grad()/step().
    ``loss_fn`` must return a differentiable scalar Tensor, never a Python
    float or a detached Tensor. It should recompute eager derived geometry or
    resolve a persistent Expr layout on every call. The trainer retains no
    loss Tensor or execution graph. Callbacks receive only numeric values.
    """

    def __init__(self, parameters, loss_fn, optimizer=None, *, identity=None):
        from .parameters import ParameterStore

        self._parameter_store = (
            parameters if isinstance(parameters, ParameterStore) else None
        )
        self.parameters = _validated_parameters(parameters)
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        identity = _serialize_layout_identity(identity)
        self.identity = identity
        self.loss_fn = loss_fn
        self.steps = 0
        self.optimizer = Adam(self.parameters) if optimizer is None else optimizer
        if not all(
            callable(getattr(self.optimizer, method, None))
            for method in ("zero_grad", "step")
        ):
            raise TypeError("optimizer must implement zero_grad() and step()")
        owned = getattr(self.optimizer, "parameters", None)
        if owned is not None:
            owned = tuple(owned.values() if isinstance(owned, Mapping) else owned)
            if len(owned) != len(self.parameters) or {id(p) for p in owned} != {
                id(p) for p in self.parameters
            }:
                raise ValueError("optimizer must own the same parameter identities")

    def step(self):
        """Run a fresh forward/backward/update and return the pre-update loss."""
        self.optimizer.zero_grad()
        for parameter in self.parameters:
            parameter.zero_grad()
        loss = self.loss_fn()
        if not isinstance(loss, Tensor):
            raise TypeError("loss_fn must return a scalar Tensor, not a float")
        if not loss.requires_grad:
            raise ValueError("loss must preserve gradients to trainable parameters")
        numeric = _number(loss, "loss")
        loss.backward()
        if not any(parameter.grad is not None for parameter in self.parameters):
            raise ValueError("loss must depend on at least one trainer parameter")
        for parameter in self.parameters:
            if parameter.grad is not None and not np.all(np.isfinite(parameter.grad)):
                raise ValueError(f"Nonfinite gradient for parameter '{parameter.name}'")
        self.optimizer.step()
        self.steps += 1
        return numeric

    def evaluate(self):
        """Evaluate the current scalar loss without performing an update."""
        return {"loss": _number(self.loss_fn(), "loss")}

    def state_dict(self):
        if self.identity is None:
            raise ValueError(
                "Layout trainer checkpoints require an explicit loss identity"
            )
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("Checkpoint parameters must have unique names")
        if not callable(getattr(self.optimizer, "state_dict", None)):
            raise TypeError("Optimizer does not support checkpoints")
        return {
            "format_version": 2,
            "steps": self.steps,
            "parameters": {p.name: p.numpy().tolist() for p in self.parameters},
            "parameter_schema": None
            if self._parameter_store is None
            else self._parameter_store.schema(),
            "identity": self.identity,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        _validate_layout_checkpoint(self, state)
        values, steps = state.get("parameters"), state.get("steps")
        checked = _validated_checkpoint_parameters(self.parameters, values, steps)
        if not callable(getattr(self.optimizer, "load_state_dict", None)):
            raise TypeError("Optimizer does not support checkpoints")
        self.optimizer.load_state_dict(state.get("optimizer"))
        for p in self.parameters:
            p.assign(checked[p.name])
        self.steps = steps

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    def load(self, path):
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def fit(
        self,
        steps,
        callback=None,
        *,
        validation=None,
        patience=None,
        min_delta=0.0,
        restore_best=True,
        validation_interval=1,
    ):
        """Return numeric losses; call callback(zero_based_step, loss) after updates."""
        from .training import fit_steps

        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        index = 0

        def step():
            return {"step": self.steps + 1, "loss": self.step()}

        def notify(result):
            nonlocal index
            if callback is not None:
                callback(index, result["loss"])
            index += 1

        history = fit_steps(
            self,
            steps,
            notify,
            validation=validation,
            patience=patience,
            min_delta=min_delta,
            restore_best=restore_best,
            validation_interval=validation_interval,
            step=step,
        )
        return [result["loss"] for result in history]
