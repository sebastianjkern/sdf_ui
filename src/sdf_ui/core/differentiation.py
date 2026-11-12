"""Shader derivative schemas for real math before texture-store quantization."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ShaderArgument:
    name: str
    shape: tuple
    kind: str


@dataclass(frozen=True)
class ShaderDifferentiation:
    name: str
    path: str
    inputs: Tuple[ShaderArgument, ...]
    parameters: Tuple[ShaderArgument, ...]
    outputs: Tuple[ShaderArgument, ...]
    behavior: str = "piecewise"
    zero_gradient_parameters: tuple = ()
    notes: str = ""
    sampling: str = "pointwise"
    reference: object = None
    defaults: object = None

    @property
    def continuous_parameters(self):
        return tuple(
            p.name
            for p in self.parameters
            if p.name not in self.zero_gradient_parameters
        )

    def evaluate(self, size, inputs=None, params=None):
        """Evaluate unquantized math, preserving Tensor arguments' graphs.

        Ports use GLSL names. Single outputs return a Tensor; multiple outputs
        return a tuple in output declaration order. Execution is CPU reference.
        """
        from ..learning.shader_derivatives import evaluate_shader

        return evaluate_shader(self, size, inputs, params)

    def vjp(self, size, inputs, params, output_gradients):
        """Compute pixel and reduced uniform adjoints without altering callers.

        The local CPU reference recomputes the forward expression. Upstream
        gradients are one array, or a tuple/name mapping for multiple outputs.
        """
        from ..learning.shader_derivatives import shader_vjp

        return shader_vjp(self, size, inputs, params, output_gradients)
