from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .lowering import GraphPlan

def inspect_graph(
    outputs: GraphPlan | Mapping[str, Any], params: Mapping[str, Any] | None = None
) -> dict[str, Any]: ...
