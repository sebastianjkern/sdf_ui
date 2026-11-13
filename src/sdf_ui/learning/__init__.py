"""Parameters, objectives, autodiff, optimizers, policies and CPU reference tools.

Compiled render graph types live in :mod:`sdf_ui.graph`.
Public objects are imported on first use.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.opengl.optim import GPUSGD, GPUAdam  # noqa: F401
    from .autodiff import (  # noqa: F401
        Parameter,
        Tensor,
        as_tensor,
        maximum,
        minimum,
        stack,
        where,
    )
    from .layout import Bounds, Element, Layout, LayoutSnapshot, LayoutTrainer  # noqa: F401
    from .objective import GeometryLoss, Objective, PixelLoss  # noqa: F401
    from .optimizers import SGD, Adam  # noqa: F401
    from .parameters import ParameterStore  # noqa: F401
    from .policy import (  # noqa: F401
        CategoricalPolicy,
        Episode,
        GaussianPolicy,
        PolicyGradientTrainer,
        RandomSource,
    )
    from .render import DifferentiableRenderer, UnsupportedGradientError  # noqa: F401
    from .shader_derivatives import ShaderGradients  # noqa: F401


_MODULES = {
    "autodiff": (
        "Tensor",
        "Parameter",
        "as_tensor",
        "stack",
        "maximum",
        "minimum",
        "where",
    ),
    "optimizers": ("Adam", "SGD"),
    "parameters": ("ParameterStore",),
    "layout": ("Bounds", "Element", "Layout", "LayoutSnapshot", "LayoutTrainer"),
    "policy": (
        "GaussianPolicy",
        "CategoricalPolicy",
        "PolicyGradientTrainer",
        "RandomSource",
        "Episode",
    ),
    "render": ("DifferentiableRenderer", "UnsupportedGradientError"),
    "shader_derivatives": ("ShaderGradients",),
    "objective": ("Objective", "PixelLoss", "GeometryLoss"),
    "..backends.opengl.optim": ("GPUAdam", "GPUSGD"),
}
__all__ = [name for names in _MODULES.values() for name in names]


def __getattr__(name):
    for module, names in _MODULES.items():
        if name in names:
            path = module if module.startswith(".") else "." + module
            value = getattr(import_module(path, __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
