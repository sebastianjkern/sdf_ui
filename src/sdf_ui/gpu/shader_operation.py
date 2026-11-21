"""Shared description of a shader and its independent derivative implementations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShaderOperation:
    """Typed ports, CPU oracle and GPU derivative knowledge for one operation.

    The CPU oracle remains independent of the packaged WGSL kernels.
    """

    name: str
    path: str
    contract: object = None
    gpu_derivative: object = None

    def reference_rule(self):
        if self.contract is None:
            raise ValueError(f"Shader '{self.name}' has no differentiation contract")
        if self.contract.reference is not None:
            if not callable(self.contract.reference):
                raise TypeError("Shader reference implementation must be callable")
            return self.contract.reference, getattr(self.contract, "defaults", None) or {}
        raise ValueError(f"Shader '{self.name}' has no extension-owned reference rule")
