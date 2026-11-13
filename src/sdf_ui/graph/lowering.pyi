from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..core.texture import TextureNode
from ..gpu.nodes.base import ShaderFile

class Port:
    pass_index: int
    output_index: int
    kind: str
    mode: str | None
    def __init__(
        self,
        pass_index: int,
        output_index: int = 0,
        kind: str = "sdf",
        mode: str | None = None,
    ) -> None: ...

class Pass:
    shader: ShaderFile
    inputs: tuple[Port, ...]
    uniforms: Callable[..., Any]
    contract: Any
    def __init__(
        self, shader: ShaderFile, inputs: tuple[Port, ...], uniforms: Callable[..., Any]
    ) -> None: ...

class GraphPlan:
    passes: tuple[Pass, ...]
    outputs: Mapping[str, Port]
    fingerprint: str
    parameter_names: tuple[str, ...]
    sources: tuple[Any, ...]
    topology_controls: tuple[Any, ...]
    required_parameters: tuple[str, ...]
    def __init__(
        self,
        passes: tuple[Pass, ...],
        outputs: Mapping[str, Port],
        fingerprint: str = "",
        parameter_names: tuple[str, ...] = (),
        sources: tuple[Any, ...] = (),
        topology_controls: tuple[Any, ...] = (),
        required_parameters: tuple[str, ...] = (),
    ) -> None: ...

def lower_graph(outputs: Mapping[str, TextureNode]) -> GraphPlan: ...

__all__: list[str]
