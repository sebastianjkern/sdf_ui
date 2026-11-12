"""Shared graph compilation, execution and inspection."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import (
        GraphCase as GraphCase,
    )
    from .dataset import (
        GraphDatasetTrainer as GraphDatasetTrainer,
    )
    from .environment import NamedRenderEnvironment as NamedRenderEnvironment
    from .environment import NamedValues as NamedValues
    from .environment import RenderEnvironment as RenderEnvironment
    from .inspection import inspect_graph as inspect_graph
    from .lowering import GraphPlan as GraphPlan
    from .lowering import Port as Port
    from .lowering import lower_graph as lower_graph
    from .runtime import (
        GraphExecution as GraphExecution,
    )
    from .runtime import (
        GraphTrainer as GraphTrainer,
    )
    from .runtime import (
        RenderGraph as RenderGraph,
    )
    from .session import TrainingSession as TrainingSession

_EXPORTS = {
    "TrainingSession": "session",
    "GraphCase": "dataset",
    "GraphDatasetTrainer": "dataset",
    "RenderEnvironment": "environment",
    "NamedRenderEnvironment": "environment",
    "NamedValues": "environment",
    "GraphPlan": "lowering",
    "Port": "lowering",
    "lower_graph": "lowering",
    "RenderGraph": "runtime",
    "GraphExecution": "runtime",
    "GraphTrainer": "runtime",
    "inspect_graph": "inspection",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
