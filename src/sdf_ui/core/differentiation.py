"""Shader derivative schemas for real math before texture-store quantization."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple


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


_PRIMITIVES = {
    "abs",
    "arc",
    "bezier",
    "capsule",
    "circle",
    "ellipse",
    "grid",
    "half_plane",
    "interpolation",
    "intersection",
    "line",
    "masked_union",
    "rect",
    "repeat",
    "ring",
    "sector",
    "smooth_min",
    "subtract",
    "transform",
    "triangle",
    "union",
}
_SHADING = {
    "clear_color",
    "fill",
    "fill_from_texture",
    "film_grain",
    "isolines",
    "light",
    "outline",
    "partial_derivative",
    "perlin_noise",
}
_EFFECTS = {
    "overlay",
    "layer_mask",
    "multiply",
    "transparency",
    "invert",
    "to_rgb",
    "to_lab",
    "dithering",
    "dither_1bit",
    "blur_hor_9",
    "blur_ver_9",
    "blur_hor_13",
    "blur_ver_13",
}


def shader_rule_module(name):
    if name in _PRIMITIVES:
        return "shader_primitives"
    if name in _SHADING:
        return "shader_shading"
    if name in _EFFECTS:
        return "shader_effects"
    return None


_NOTES = {
    "abs": "Zero subgradient at zero.",
    "arc": "Active radial or endpoint branch; angular selection is fixed away from boundaries.",
    "bezier": "Closest-point envelope derivative with active root/endpoint fixed; final abs removes sign.",
    "grid": "Modulo cell indices fixed; spacing/rotation differentiate within each cell.",
    "line": "Closest point; coincident endpoints are outside the finite shader domain.",
    "masked_union": "Distance and mask adjoints both contribute to source fields.",
    "repeat": "Integer imageLoad addresses: spacing derivative zero away from jumps, source gradients accumulate.",
    "transform": "Bilinear source gradients scatter-add; transform_kind is a discrete mode.",
    "triangle": "Active edges/signs fixed; degenerate triangles outside shader domain.",
    "film_grain": "Fixed pixel coordinates/hash constants; declared time uniform is unused and has zero derivative.",
    "perlin_noise": "Fixed pixel coordinates/noise constants; no trainable shader inputs.",
    "partial_derivative": "Transpose of the forward finite-difference stencil, including border reads.",
    "light": "Sobel stencil, normals/material/light differentiate; heightProfile is discrete.",
    "dithering": "Additive threshold derivative zero; original color derivative identity away from jumps.",
    "dither_1bit": "True zero derivative almost everywhere; undefined at thresholds. No straight-through estimator.",
}


@lru_cache(maxsize=None)
def shader_differentiation(name: str, path: str) -> Optional[ShaderDifferentiation]:
    """Infer ports from builtin GLSL; unknown and third-party shaders opt out."""
    if shader_rule_module(name) is None:
        return None
    if name in _PRIMITIVES:
        directory = {"interpolation": "interpolate", "smooth_min": "smooth_union"}.get(
            name, name
        )
        expected_path = f"core/plugins/primitives/{directory}/shader.glsl"
    elif name in _SHADING:
        expected_path = f"core/plugins/shading/{name}/shader.glsl"
    elif name in {"overlay", "layer_mask", "multiply", "transparency"}:
        directory = "alpha_overlay" if name == "overlay" else name
        expected_path = f"core/plugins/layer/{directory}/shader.glsl"
    elif name.startswith("blur_"):
        _, direction, taps = name.split("_")
        suffix = "hor" if direction == "hor" else "vert"
        expected_path = f"core/plugins/postprocessing/blur/blur{taps}_{suffix}.glsl"
    else:
        expected_path = f"core/plugins/postprocessing/{name}/shader.glsl"
    source_path = Path(__file__).resolve().parents[1] / path
    if path != expected_path or not source_path.is_file():
        return None
    source = source_path.read_text(encoding="utf-8")
    images = re.findall(
        r"layout\s*\(\s*(r32f|rgba8)\s*,[^)]*\)\s*"
        r"(readonly|writeonly)\s+uniform\s+image2D\s+(\w+)\s*;",
        source,
    )
    inputs: List[ShaderArgument] = []
    outputs: List[ShaderArgument] = []
    for storage, access, uniform in images:
        argument = ShaderArgument(uniform, (() if storage == "r32f" else (4,)), storage)
        (inputs if access == "readonly" else outputs).append(argument)
    parameters = []
    for dtype, uniform, count in re.findall(
        r"^uniform\s+(float|vec[234]|int|bool)\s+(\w+)"
        r"(?:\[(\w+)\])?[^;]*;",
        source,
        re.MULTILINE,
    ):
        shape: tuple = () if dtype in {"float", "int", "bool"} else (int(dtype[-1]),)
        if count:
            if not count.isdigit():
                match = re.search(
                    r"#define\s+" + re.escape(count) + r"\s+(\d+)", source
                )
                if match is None:
                    raise ValueError(f"Unknown uniform array size {count!r} in {path}")
                count = match.group(1)
            shape = (int(count),) + shape
        parameters.append(ShaderArgument(uniform, shape, dtype))
    zero = tuple(p.name for p in parameters if p.kind in {"int", "bool"})
    zero += {
        "transform": ("transform_kind",),
        "light": ("heightProfile",),
        "repeat": ("repeat",),
        "film_grain": ("time",),
    }.get(name, ())
    if name.startswith("blur_"):
        zero += ("offset",)
    behavior = "piecewise"
    if name in {"film_grain", "perlin_noise"}:
        behavior = "constant"
    elif name == "dither_1bit":
        behavior = "zero_almost_everywhere"
    elif name in {
        "interpolation",
        "multiply",
        "transparency",
        "invert",
        "layer_mask",
        "partial_derivative",
    } or name.startswith("blur_"):
        behavior = "smooth_for_fixed_sampling"
    notes = _NOTES.get(
        name, "Piecewise derivatives; min/max ties split adjoints equally."
    )
    if name.startswith("blur_"):
        notes = "Transpose the actual reflected stencil; weights differentiate, truncated offsets have zero gradients."
    sampling = "pointwise"
    if name == "transform":
        sampling = "clamped_bilinear"
    elif name in {"repeat", "partial_derivative"}:
        sampling = "integer_zero_outside"
    elif name.startswith("blur_"):
        sampling = "asymmetric_reflected_zero_outside"
    elif name == "light":
        sampling = "clamped_sobel"
    return ShaderDifferentiation(
        name,
        path,
        tuple(inputs),
        tuple(parameters),
        tuple(outputs),
        behavior,
        zero,
        notes,
        sampling,
    )
