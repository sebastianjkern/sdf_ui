"""Boolean selection for autodiff tensors."""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .autodiff import Tensor, _unbroadcast, as_tensor


def where(condition: Any, x: Any, y: Any):
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
