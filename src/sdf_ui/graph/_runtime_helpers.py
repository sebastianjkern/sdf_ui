"""Dependency adapters used by the render graph runtime."""

from collections.abc import Mapping

from ..core.expressions import Expr
from ..core.image import Image as BackendImage
from ..core.texture import MultiOutputResult, TextureNode
from ..learning.autodiff import Parameter, Tensor
from ..learning.objective import Objective
from .lowering import GraphPlan, lower_graph


def _normalize_graph_outputs(outputs):
    if isinstance(outputs, (TextureNode, MultiOutputResult)):
        outputs = {"output": outputs}
    output_nodes = (
        {id(node): name for name, node in outputs.items()}
        if isinstance(outputs, Mapping)
        else {}
    )
    plan = outputs if isinstance(outputs, GraphPlan) else lower_graph(outputs)
    return output_nodes, plan


def _is_parameter(value):
    return isinstance(value, Parameter)


def _needs_expression_evaluation(value):
    return isinstance(value, (Expr, str, Tensor))


def _is_backend_image(value):
    return isinstance(value, BackendImage)


def _create_execution(graph, objective, loss):
    from .runtime import GraphExecution

    return GraphExecution(graph, objective, loss=loss)


def _validate_run_request(graph, objective, loss):
    if graph._running or (graph._active is not None and not graph._active.closed):
        raise RuntimeError("Close the previous graph execution before starting another")
    if objective is not None and loss is not None:
        raise ValueError("Pass either an Objective or a loss callback, not both")
    if objective is not None and not isinstance(objective, Objective):
        raise TypeError("Graph objective must be an Objective")
    if loss is not None and not callable(loss):
        raise TypeError("Graph loss must be callable")


def _evaluate_reference(plan, size, params, progress, inputs, sources):
    from ..learning.render import DifferentiableRenderer
    from .execution import execute_reference

    renderer = DifferentiableRenderer(size, params, progress=progress)
    renderer.inputs = inputs
    return execute_reference(plan, renderer, sources=sources)


def _create_environment(graph, **kwargs):
    from .environment import NamedRenderEnvironment

    return NamedRenderEnvironment(graph, **kwargs)


def _fit_result(history):
    from .runtime import FitResult

    return FitResult(history)


def _is_objective(value):
    return isinstance(value, Objective)
