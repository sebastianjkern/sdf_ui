"""Shared description of a shader and its independent derivative implementations."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShaderOperation:
    """Forward source, typed ports, CPU oracle and GPU VJP for one operation.

    Builtin forward and backward shaders share a program. The CPU oracle
    remains independent. Custom descriptors may supply both rules.
    """

    name: str
    path: str
    contract: object = None
    gpu_derivative: object = None
    program: object = None

    def expression(self):
        if self.program is not None:
            from .shader_expression import ShaderProgram

            if (
                not isinstance(self.program, ShaderProgram)
                or self.program.name != self.name
            ):
                raise ValueError(
                    "Shared shader program must have the descriptor's name"
                )
            if self.gpu_derivative is not None:
                raise ValueError(
                    "Specify either a shared shader program or a custom GPU derivative"
                )
            if self.contract is None:
                raise ValueError(
                    "A shared shader program requires a differentiation contract"
                )
            return self.program
        return None

    def forward_source(
        self, *, sdf_format="r32f", color_format="rgba8", base_path=None
    ):
        root = (
            Path(base_path)
            if base_path is not None
            else Path(__file__).resolve().parents[1]
        )
        source = (root / self.path).read_text(encoding="utf-8")
        expression = self.expression()
        if expression is not None:
            return expression.forward(
                self.contract,
                sdf_format=sdf_format,
                color_format=color_format,
                original_source=source,
            )
        if sdf_format != "r32f":
            source = source.replace("layout (r32f", f"layout ({sdf_format}")
        if color_format != "rgba8":
            source = source.replace("rgba8", color_format)
        return source

    def reference_rule(self):
        if self.contract is None:
            raise ValueError(f"Shader '{self.name}' has no differentiation contract")
        if self.contract.reference is not None:
            if not callable(self.contract.reference):
                raise TypeError("Shader reference implementation must be callable")
            return self.contract.reference, getattr(self.contract, "defaults", None) or {}
        raise ValueError(f"Shader '{self.name}' has no extension-owned reference rule")

    @staticmethod
    def workgroup_size(source):
        layout = re.search(r"layout\s*\(([^)]*local_size[^)]*)\)\s*in\s*;", source)
        if layout is None:
            raise ValueError(
                "Forward compute shaders must declare a fixed local workgroup size"
            )
        sizes = []
        for axis in ("x", "y", "z"):
            match = re.search(r"local_size_" + axis + r"\s*=\s*(\w+)", layout.group(1))
            value = "1" if match is None else match.group(1)
            if not value.isdigit():
                macro = re.search(
                    r"#define\s+" + re.escape(value) + r"\s+(\d+)", source
                )
                if macro is None:
                    raise ValueError(
                        "Workgroup dimensions must be integer literals or defines"
                    )
                value = macro.group(1)
            sizes.append(int(value))
        if any(v <= 0 for v in sizes) or sizes[2] != 1:
            raise ValueError(
                "SDF graph shaders require positive 2D workgroups with local_size_z=1"
            )
        return tuple(sizes)

    def backward_source(self):
        if self.contract is None:
            raise ValueError(f"Shader '{self.name}' has no differentiation contract")
        body = self.gpu_derivative
        expression = self.expression()
        if expression is not None and expression.reverse_expression is not None:
            return expression.reverse_expression.reverse_source(self.contract)
        if body is None and expression is not None:
            body = expression.body
        if body is None:
            raise ValueError(
                f"Shader '{self.name}' needs an explicit native GPU backward body"
            )
        from ..backends.opengl.glsl import derivative_source

        return derivative_source(self.contract, body)
