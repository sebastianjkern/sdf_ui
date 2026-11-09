"""Named trainable values shared by layout expressions and render graphs."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .autodiff import Parameter

_UNSET = object()


class ParameterStore(Mapping):
    """A named collection usable directly as ``render(..., params=store)``.

    Checkpoints contain values, rather than execution tapes or GPU resources.
    Load them into an identically declared store before the next execution.
    """

    def __init__(self, parameters=()):
        self._parameters = {}
        self._scalars = {}
        for parameter in parameters:
            self.add(parameter)

    def add(self, parameter):
        if not isinstance(parameter, Parameter):
            raise TypeError("ParameterStore accepts Parameter instances")
        if parameter.name in self._parameters:
            raise ValueError(f"Duplicate parameter name: {parameter.name}")
        self._parameters[parameter.name] = parameter
        return parameter

    def define(self, name, initial):
        """Create, register, and return a trainable parameter."""
        return self.add(Parameter(name, initial))

    def bind(self, name, initial=_UNSET):
        """Return a reusable scene expression, optionally defining its leaf.

        Unlike an eager Tensor expression, the returned expression is resolved
        afresh after each optimizer step. Existing parameters need no initial value.
        """
        from ..core.expressions import param

        if name not in self:
            if initial is _UNSET:
                raise KeyError(f"Unknown parameter '{name}'; supply an initial value")
            self.define(name, initial)
        elif initial is not _UNSET:
            raise ValueError(f"Parameter '{name}' already exists")
        return param(name)

    def scalar(self, name, *, initial, min=None, max=None):
        """Declare a scalar expression, with optional smooth open bounds.

        ``initial`` is the actual rendered value. A bounded value must be
        strictly inside its bounds; the bounds are approached asymptotically.
        Use :meth:`value` and :meth:`assign_value` for physical values.
        """
        bounds = (min, max)
        if any(
            v is not None
            and (isinstance(v, bool) or not np.isscalar(v) or not np.isfinite(v))
            for v in bounds
        ):
            raise ValueError("Bounds must be finite scalar values")
        if min is not None and max is not None and min >= max:
            raise ValueError("Minimum must be less than maximum")
        raw = self._inverse_scalar(initial, bounds)
        self.define(name, raw)
        self._scalars[name] = bounds
        return self.scalar_expression(name)

    def scalar_expression(self, name):
        """Return the lazy expression for an existing scalar declaration."""
        from ..core.expressions import param

        if name not in self._scalars:
            raise KeyError(f"'{name}' is not a declared scalar")
        min, max = self._scalars[name]
        expr = param(name)
        if min is not None and max is not None:
            return min + (max - min) * expr.sigmoid()
        if min is not None:
            return min + expr.softplus()
        if max is not None:
            return max - expr.softplus()
        return expr

    @staticmethod
    def _inverse_scalar(value, bounds):
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError("Scalar value must be finite")
        value = float(value)
        lower, upper = bounds
        if (lower is not None and value <= lower) or (
            upper is not None and value >= upper
        ):
            raise ValueError("Scalar value must be strictly inside its bounds")
        if lower is not None and upper is not None:
            return math.log(value - lower) - math.log(upper - value)
        distance = (
            value - lower
            if lower is not None
            else upper - value
            if upper is not None
            else None
        )
        return (
            value if distance is None else distance + math.log(-math.expm1(-distance))
        )

    def assign_value(self, name, value):
        """Assign a declared scalar in its rendered units."""
        if name not in self._scalars:
            raise KeyError(f"'{name}' is not a declared scalar")
        self[name].assign(self.raw_value(name, value))
        return self

    def is_scalar(self, name):
        """Whether a name was declared through :meth:`scalar`."""
        return name in self._scalars

    def scalar_name(self, expression):
        """Find the declaration corresponding to an expression, if any."""
        from ..core.expressions import Expr

        if not isinstance(expression, Expr):
            return None
        return next(
            (
                name
                for name in self._scalars
                if expression == self.scalar_expression(name)
            ),
            None,
        )

    def raw_value(self, name, value):
        """Validate a physical scalar and return its stored raw value."""
        if name not in self._scalars:
            raise KeyError(f"'{name}' is not a declared scalar")
        return self._inverse_scalar(value, self._scalars[name])

    def value(self, name):
        """Return a declared scalar in its rendered units as a Python float."""
        if name not in self._scalars:
            raise KeyError(f"'{name}' is not a declared scalar")
        raw = self[name].item()
        lower, upper = self._scalars[name]
        if lower is not None and upper is not None:
            sigmoid = (
                1 / (1 + math.exp(-raw))
                if raw >= 0
                else math.exp(raw) / (1 + math.exp(raw))
            )
            return float(lower + (upper - lower) * sigmoid)
        softplus = max(raw, 0) + math.log1p(math.exp(-abs(raw)))
        return float(
            lower + softplus
            if lower is not None
            else upper - softplus
            if upper is not None
            else raw
        )

    def __getitem__(self, key):
        return self._parameters[key]

    def __iter__(self):
        return iter(self._parameters)

    def __len__(self):
        return len(self._parameters)

    def parameters(self):
        return tuple(self._parameters.values())

    def state_dict(self):
        return {
            "format_version": 1,
            "schema": self.schema(),
            "parameters": {
                name: parameter.numpy().tolist() for name, parameter in self.items()
            },
        }

    def schema(self):
        """Return JSON-compatible shapes and declared scalar transforms."""
        return {
            name: {
                "shape": list(parameter.shape),
                "scalar_bounds": [
                    None if bound is None else float(bound)
                    for bound in self._scalars[name]
                ]
                if name in self._scalars
                else None,
            }
            for name, parameter in self.items()
        }

    def load_state_dict(self, state):
        """Validate the entire checkpoint before assigning any values."""
        if not isinstance(state, Mapping) or state.get("format_version") != 1:
            raise ValueError("Invalid parameter checkpoint format")
        if state.get("schema") != self.schema():
            raise ValueError("Checkpoint parameter schema differs from the store")
        raw = state.get("parameters")
        if not isinstance(raw, Mapping) or set(raw) != set(self):
            raise ValueError("Checkpoint parameter names do not match the store")
        values = {}
        for name, parameter in self.items():
            value = np.asarray(raw[name], dtype=float)
            if value.shape != parameter.shape:
                raise ValueError(f"Checkpoint shape mismatch for '{name}'")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Nonfinite checkpoint value for '{name}'")
            values[name] = value
        for name, value in values.items():
            self[name].assign(value)

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    def load(self, path):
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
