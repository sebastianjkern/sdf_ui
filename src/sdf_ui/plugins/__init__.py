"""Extension API for registering and lowering shader-backed operations."""

from ..core.differentiation import ShaderArgument, ShaderDifferentiation
from ..core.plugins.base import Plugin, PluginFamily, ShaderFile, TextureKind
from ..core.plugins.common import shader
from ..core.plugins.registry import PluginRegistry, registry
from ..core.shader_expression import Expression, ShaderProgram
from ..core.shader_operation import ShaderOperation
from ..graph.lowering import GraphPlan, Pass, Port, lower_graph

__all__ = [
    "Plugin",
    "PluginFamily",
    "ShaderFile",
    "TextureKind",
    "ShaderArgument",
    "ShaderDifferentiation",
    "PluginRegistry",
    "registry",
    "shader",
    "GraphPlan",
    "Pass",
    "Port",
    "lower_graph",
    "Expression",
    "ShaderProgram",
    "ShaderOperation",
]
