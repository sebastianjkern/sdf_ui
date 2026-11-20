"""Functional operations over the autodiff Tensor type."""

from typing import TYPE_CHECKING, Any, Iterable, Tuple

import numpy as np

if TYPE_CHECKING:
    from .autodiff import Tensor, as_tensor


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


def sum(value: Any, axis: Any = None, keepdims: bool = False):
    return as_tensor(value).sum(axis, keepdims)


def mean(value: Any, axis: Any = None, keepdims: bool = False):
    return as_tensor(value).mean(axis, keepdims)


def relu(value: Any):
    return as_tensor(value).relu()


def softplus(value: Any):
    return as_tensor(value).softplus()


def sigmoid(value: Any):
    return as_tensor(value).sigmoid()


def tanh(value: Any):
    return as_tensor(value).tanh()


def log(value: Any):
    return as_tensor(value).log()


def exp(value: Any):
    return as_tensor(value).exp()


def sqrt(value: Any):
    return as_tensor(value).sqrt()


def abs(value: Any):
    return as_tensor(value).abs()


def sin(value: Any):
    return as_tensor(value).sin()


def cos(value: Any):
    return as_tensor(value).cos()


def maximum(left: Any, right: Any):
    return as_tensor(left).maximum(right)


def minimum(left: Any, right: Any):
    return as_tensor(left).minimum(right)


def matmul(left: Any, right: Any):
    return as_tensor(left).matmul(right)


def reshape(value: Any, shape: Any):
    return as_tensor(value).reshape(shape)


def transpose(value: Any, axes: Any = None):
    return as_tensor(value).transpose(axes)


def stack(values: Iterable[Any], axis: int = 0):
    tensors = tuple(as_tensor(value) for value in values)
    value = np.stack([t._data for t in tensors], axis=axis)
    return Tensor._result(
        value,
        tensors,
        lambda g: tuple(np.take(g, i, axis=axis) for i in range(len(tensors))),
    )
