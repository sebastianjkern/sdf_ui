from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from numpy.typing import NDArray

from ...gpu.nodes.differentiation import ShaderDifferentiation
from ...learning.autodiff import Tensor

class ShaderGradients:
    inputs: Mapping[str, NDArray[Any]]
    parameters: Mapping[str, NDArray[Any]]
    def __init__(
        self, inputs: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> None: ...

def evaluate_shader(
    contract: ShaderDifferentiation,
    size: tuple[int, int],
    inputs: Mapping[str, Any] | None = ...,
    params: Mapping[str, Any] | None = ...,
) -> Tensor | tuple[Tensor, ...]: ...
def shader_vjp(
    contract: ShaderDifferentiation,
    size: tuple[int, int],
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
    output_gradients: Any,
) -> ShaderGradients: ...
