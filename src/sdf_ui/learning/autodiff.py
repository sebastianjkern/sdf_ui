"""NumPy reverse-mode autodiff tensors and named parameter leaves.

Values are finite float64 arrays. Public value access returns copies; use
``Parameter.assign`` to mutate a parameter and invalidate existing graphs.
Backward accumulates gradients; clear them through an optimizer between
training iterations. Non-scalar outputs require an explicit gradient seed.
"""

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Tensor",
    "Parameter",
    "as_tensor",
    "sum",
    "mean",
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
    "maximum",
    "minimum",
    "stack",
    "matmul",
    "where",
    "reshape",
    "transpose",
]


def _array(value: Any, name: str = "value") -> np.ndarray:
    if isinstance(value, Tensor):
        value = value._data
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise TypeError("{} must contain real numbers".format(name))
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("{} must be finite".format(name))
    return result


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("nonfinite {}".format(name))
    return value


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A real-valued array with an optional reverse-mode computation graph."""

    __array_priority__ = 1000

    def __init__(self, data: Any, requires_grad: bool = False):
        self._data = _array(data)
        self._data.flags.writeable = False
        self.requires_grad = bool(requires_grad)
        self.grad = None  # type: Optional[np.ndarray]
        self._parents = ()  # type: Tuple[Tensor, ...]
        self._pullback = None  # type: Optional[Callable]
        self._versions = {}  # type: Dict[Parameter, int]

    @classmethod
    def _result(
        cls, data: Any, parents: Sequence["Tensor"], pullback: Callable
    ) -> "Tensor":
        _finite(data, "forward value")
        result = Tensor(data, any(p.requires_grad for p in parents))
        if result.requires_grad:
            result._parents = tuple(parents)
            result._pullback = pullback
            for parent in parents:
                parent._check_versions()
                result._versions.update(parent._versions)
        return result

    def _check_versions(self) -> None:
        for parameter, version in self._versions.items():
            if parameter.version != version:
                raise RuntimeError(
                    "stale computation graph: parameter {!r} changed after forward".format(
                        parameter.name
                    )
                )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def data(self) -> np.ndarray:
        return self.numpy()

    @property
    def value(self) -> np.ndarray:
        return self.numpy()

    def numpy(self) -> np.ndarray:
        return self._data.copy()

    def item(self) -> float:
        return self._data.item()

    def detach(self) -> "Tensor":
        return Tensor(self._data)

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        return "Tensor({!r}, requires_grad={!r})".format(self._data, self.requires_grad)

    def __bool__(self) -> bool:
        raise TypeError("Tensor truth is ambiguous; use item() explicitly")

    def backward(self, grad: Any = None) -> None:
        """Accumulate a vector-Jacobian product; reject stale or invalid graphs.

        Each call uses fresh intermediate adjoints. All resulting gradients are
        validated before committing, including accumulation with existing grads.
        """
        if not self.requires_grad:
            raise RuntimeError("cannot backward through a detached tensor")
        self._check_versions()
        if grad is None:
            if self.size != 1:
                raise ValueError("non-scalar backward requires an explicit gradient")
            seed = np.ones_like(self._data)
        else:
            seed = _array(grad, "gradient")
            if seed.shape != self.shape:
                raise ValueError("gradient shape must match output shape")
        order = []
        visited = set()
        pending = [(self, False)]
        while pending:
            node, expanded = pending.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                pending.append((node, True))
                pending.extend((p, False) for p in node._parents if p.requires_grad)
        adjoints = {self: seed}
        with np.errstate(all="ignore"):
            for node in reversed(order):
                upstream = adjoints.get(node)
                if upstream is None or node._pullback is None:
                    continue
                contributions = node._pullback(upstream)
                for parent, contribution in zip(node._parents, contributions):
                    if not parent.requires_grad or contribution is None:
                        continue
                    contribution = _finite(contribution, "gradient")
                    if contribution.shape != parent.shape:
                        raise RuntimeError("internal gradient shape mismatch")
                    if parent in adjoints:
                        contribution = _finite(
                            adjoints[parent] + contribution, "gradient"
                        )
                    adjoints[parent] = contribution
            updates = {}
            for node, contribution in adjoints.items():
                if node.grad is not None:
                    previous = _array(node.grad, "existing gradient")
                    if previous.shape != node.shape:
                        raise ValueError("existing gradient shape mismatch")
                    contribution = previous + contribution
                updates[node] = _finite(contribution, "accumulated gradient").copy()
        for node, contribution in updates.items():
            node.grad = contribution

    def _binary(self, other: Any, forward: Callable, derivative: Callable) -> "Tensor":
        other = as_tensor(other)
        a, b = self._data, other._data
        with np.errstate(all="ignore"):
            value = forward(a, b)

        def pullback(g: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
            da, db = derivative(g, a, b)
            return (
                _unbroadcast(da, self.shape) if self.requires_grad else None,
                _unbroadcast(db, other.shape) if other.requires_grad else None,
            )

        return Tensor._result(value, (self, other), pullback)

    def __add__(self, other: Any) -> "Tensor":
        return self._binary(other, np.add, lambda g, a, b: (g, g))

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __sub__(self, other: Any) -> "Tensor":
        return self._binary(other, np.subtract, lambda g, a, b: (g, -g))

    def __rsub__(self, other: Any) -> "Tensor":
        return as_tensor(other) - self

    def __mul__(self, other: Any) -> "Tensor":
        return self._binary(other, np.multiply, lambda g, a, b: (g * b, g * a))

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __truediv__(self, other: Any) -> "Tensor":
        return self._binary(other, np.divide, lambda g, a, b: (g / b, -g * a / b**2))

    def __rtruediv__(self, other: Any) -> "Tensor":
        return as_tensor(other) / self

    def __pow__(self, other: Any) -> "Tensor":
        other = as_tensor(other)
        a = self._data
        if other.requires_grad and np.any(a <= 0):
            raise ValueError("a differentiable exponent requires positive bases")

        def derivative(
            g: np.ndarray, a: np.ndarray, b: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            shape = np.broadcast(a, b).shape
            # x**0 is constant, including at zero; avoid 0 * (0**-1).
            power = np.zeros(shape)
            np.power(a, b - 1, out=power, where=np.broadcast_to(b != 0, shape))
            da = g * b * power if self.requires_grad else g
            db = g * np.power(a, b) * np.log(a) if other.requires_grad else g
            return da, db

        return self._binary(other, np.power, derivative)

    def __rpow__(self, other: Any) -> "Tensor":
        return as_tensor(other) ** self

    def __neg__(self) -> "Tensor":
        return Tensor._result(-self._data, (self,), lambda g: (-g,))

    def __abs__(self) -> "Tensor":
        return self.abs()

    def _unary(self, forward: Callable, derivative: Callable) -> "Tensor":
        a = self._data
        with np.errstate(all="ignore"):
            value = forward(a)
        return Tensor._result(value, (self,), lambda g: (g * derivative(a, value),))

    def relu(self) -> "Tensor":
        return self._unary(lambda a: np.maximum(a, 0), lambda a, y: a > 0)

    def softplus(self) -> "Tensor":
        return self._unary(lambda a: np.logaddexp(0, a), lambda a, y: _sigmoid(a))

    def sigmoid(self) -> "Tensor":
        return self._unary(_sigmoid, lambda a, y: y * (1 - y))

    def tanh(self) -> "Tensor":
        return self._unary(np.tanh, lambda a, y: 1 - y * y)

    def log(self) -> "Tensor":
        return self._unary(np.log, lambda a, y: 1 / a)

    def exp(self) -> "Tensor":
        return self._unary(np.exp, lambda a, y: y)

    def sqrt(self) -> "Tensor":
        """Square root with the finite subgradient convention sqrt'(0) = 0."""
        return self._unary(
            np.sqrt,
            lambda a, y: np.divide(0.5, y, out=np.zeros_like(y), where=y != 0),
        )

    def abs(self) -> "Tensor":
        return self._unary(np.abs, lambda a, y: np.sign(a))

    def sin(self) -> "Tensor":
        return self._unary(np.sin, lambda a, y: np.cos(a))

    def cos(self) -> "Tensor":
        return self._unary(np.cos, lambda a, y: -np.sin(a))

    def maximum(self, other: Any) -> "Tensor":
        return self._binary(
            other,
            np.maximum,
            lambda g, a, b: (
                g * ((a > b) + 0.5 * (a == b)),
                g * ((b > a) + 0.5 * (a == b)),
            ),
        )

    def minimum(self, other: Any) -> "Tensor":
        return self._binary(
            other,
            np.minimum,
            lambda g, a, b: (
                g * ((a < b) + 0.5 * (a == b)),
                g * ((b < a) + 0.5 * (a == b)),
            ),
        )

    def sum(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        value = (
            self._data.sum(axis=axis, keepdims=True)
            if keepdims
            else self._data.sum(axis=axis, keepdims=False)
        )
        axes = _axes(axis, self.ndim)

        def pullback(g: np.ndarray) -> Tuple[np.ndarray]:
            if not keepdims:
                for dimension in sorted(axes):
                    g = np.expand_dims(g, dimension)
            return (np.broadcast_to(g, self.shape),)

        return Tensor._result(value, (self,), pullback)

    def mean(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        axes = _axes(axis, self.ndim)
        count = int(np.prod([self.shape[a] for a in axes], dtype=np.int64))
        if count == 0:
            raise ValueError("mean of an empty reduction is undefined")
        return self.sum(axis=axis, keepdims=keepdims) / count

    def reshape(self, *shape: Any) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        value = self._data.reshape(*shape) if shape else self._data.reshape(())
        return Tensor._result(value, (self,), lambda g: (g.reshape(self.shape),))

    def transpose(self, *axes: Any) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        elif axes == (None,):
            axes = ()
        permutation = tuple(range(self.ndim - 1, -1, -1)) if not axes else axes
        value = self._data.transpose(permutation)
        normalized = tuple(a % self.ndim for a in permutation)
        inverse = tuple(np.argsort(normalized))
        return Tensor._result(value, (self,), lambda g: (g.transpose(inverse),))

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def matmul(self, other: Any) -> "Tensor":
        other = as_tensor(other)
        a, b = self._data, other._data
        if a.ndim not in (1, 2) or b.ndim not in (1, 2):
            raise ValueError("matmul supports vectors and two-dimensional matrices")
        with np.errstate(all="ignore"):
            value = a @ b

        def pullback(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if a.ndim == b.ndim == 1:
                return g * b, g * a
            if a.ndim == 2 and b.ndim == 1:
                return np.outer(g, b), a.T @ g
            if a.ndim == 1 and b.ndim == 2:
                return g @ b.T, np.outer(a, g)
            return g @ b.T, a.T @ g

        return Tensor._result(value, (self, other), pullback)

    def __matmul__(self, other: Any) -> "Tensor":
        return self.matmul(other)

    def __rmatmul__(self, other: Any) -> "Tensor":
        return as_tensor(other).matmul(self)

    def __getitem__(self, index: Any) -> "Tensor":
        # Freeze mutable advanced indices so later caller edits cannot reroute grads.
        def freeze(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, list):
                return np.array(value, copy=True)
            if isinstance(value, tuple):
                return tuple(freeze(part) for part in value)
            return value

        index = freeze(index)
        value = self._data[index]

        def pullback(g: np.ndarray) -> Tuple[np.ndarray]:
            result = np.zeros_like(self._data)
            np.add.at(result, index, g)
            return (result,)

        return Tensor._result(value, (self,), pullback)


class Parameter(Tensor):
    """A named graph weight; assignments preserve shape and increment version."""

    def __init__(self, name: str, initial: Any, *, trainable: bool = True):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("parameter name must be a nonempty string")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a bool")
        super().__init__(initial, requires_grad=trainable)
        self._trainable = trainable
        self.name = name
        self._version = 0
        self._versions = {self: self._version}

    @property
    def version(self) -> int:
        return self._version

    @property
    def trainable(self) -> bool:
        return self._trainable

    def assign(self, value: Any) -> "Parameter":
        data = _array(value, "parameter value")
        if data.shape != self.shape:
            raise ValueError("assignment must preserve parameter shape")
        data.flags.writeable = False
        self._data = data
        self._version += 1
        self._versions = {self: self._version}
        return self


def as_tensor(value: Any, requires_grad: bool = False) -> Tensor:
    """Preserve tensor identity and graphs; wrap other values as new tensors."""
    if isinstance(value, Tensor):
        if requires_grad and not value.requires_grad:
            return Tensor(value._data, requires_grad=True)
        return value
    return Tensor(value, requires_grad=requires_grad)


def _axes(axis: Any, ndim: int) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    normalized = []
    for a in axes:
        if not isinstance(a, (int, np.integer)):
            raise TypeError("reduction axes must be integers")
        if not -ndim <= a < ndim:
            # NumPy accepts scalar sum(axis=0), including axis=-1.
            if ndim == 0 and a in (0, -1) and not isinstance(axis, tuple):
                return ()
            raise ValueError("reduction axis out of bounds")
        a = int(a) % ndim
        if a in normalized:
            raise ValueError("duplicate reduction axis")
        normalized.append(a)
    return tuple(normalized)


def _sigmoid(a: np.ndarray) -> np.ndarray:
    e = np.exp(-np.abs(a))
    return np.where(a >= 0, 1 / (1 + e), e / (1 + e))


def sum(value: Any, axis: Any = None, keepdims: bool = False) -> Tensor:
    return as_tensor(value).sum(axis, keepdims)


def mean(value: Any, axis: Any = None, keepdims: bool = False) -> Tensor:
    return as_tensor(value).mean(axis, keepdims)


def relu(value: Any) -> Tensor:
    return as_tensor(value).relu()


def softplus(value: Any) -> Tensor:
    return as_tensor(value).softplus()


def sigmoid(value: Any) -> Tensor:
    return as_tensor(value).sigmoid()


def tanh(value: Any) -> Tensor:
    return as_tensor(value).tanh()


def log(value: Any) -> Tensor:
    return as_tensor(value).log()


def exp(value: Any) -> Tensor:
    return as_tensor(value).exp()


def sqrt(value: Any) -> Tensor:
    return as_tensor(value).sqrt()


def abs(value: Any) -> Tensor:
    return as_tensor(value).abs()


def sin(value: Any) -> Tensor:
    return as_tensor(value).sin()


def cos(value: Any) -> Tensor:
    return as_tensor(value).cos()


def maximum(left: Any, right: Any) -> Tensor:
    return as_tensor(left).maximum(right)


def minimum(left: Any, right: Any) -> Tensor:
    return as_tensor(left).minimum(right)


def matmul(left: Any, right: Any) -> Tensor:
    return as_tensor(left).matmul(right)


def reshape(value: Any, shape: Any) -> Tensor:
    return as_tensor(value).reshape(shape)


def transpose(value: Any, axes: Any = None) -> Tensor:
    return as_tensor(value).transpose(axes)


def stack(values: Iterable[Any], axis: int = 0) -> Tensor:
    tensors = tuple(as_tensor(value) for value in values)
    value = np.stack([t._data for t in tensors], axis=axis)
    return Tensor._result(
        value,
        tensors,
        lambda g: tuple(np.take(g, i, axis=axis) for i in range(len(tensors))),
    )


def where(condition: Any, x: Any, y: Any) -> Tensor:
    condition = np.array(condition, copy=True)
    if condition.dtype.kind != "b":
        raise TypeError("where condition must be a boolean array")
    x, y = as_tensor(x), as_tensor(y)
    return Tensor._result(
        np.where(condition, x._data, y._data),
        (x, y),
        lambda g: (
            _unbroadcast(np.where(condition, g, 0), x.shape),
            _unbroadcast(np.where(condition, 0, g), y.shape),
        ),
    )
