"""Display and CPU-reference executors for the shared typed render plan."""

from ..core.expressions import Expr, evaluate_expr
from ..graph.lowering import GraphPlan
from ..learning.autodiff import Tensor, as_tensor
from ..util import hex_col


def _resolve(value, context, params):
    if isinstance(value, (Expr, str, Tensor)):
        value = evaluate_expr(value, context, params, track_grad=True)
    if isinstance(value, str) and value.startswith("#"):
        if len(value) == 4:
            value = "#" + "".join(c * 2 for c in value[1:])
        return hex_col(value)
    return value


def execute_reference(plan, renderer, sources=None):
    """Evaluate a GraphPlan with the independent analytic CPU oracle."""
    if not isinstance(plan, GraphPlan):
        raise TypeError("Expected a GraphPlan")
    if plan.sources and sources is None:
        raise ValueError("CPU reference plans do not accept materialized GPU sources")
    sources = () if sources is None else tuple(as_tensor(source) for source in sources)
    if len(sources) != len(plan.sources):
        raise ValueError("Reference source count differs from the plan")
    results = []

    def get(port):
        return (
            sources[-port.pass_index - 1]
            if port.pass_index < 0
            else results[port.pass_index][port.output_index]
        )

    for operation in plan.passes:
        contract = operation.contract
        inputs = {
            arg.name: get(port) for arg, port in zip(contract.inputs, operation.inputs)
        }
        params = operation.uniforms(lambda v: _resolve(v, renderer, renderer.params))
        value = contract.evaluate(renderer.size, inputs, params)
        results.append(value if isinstance(value, tuple) else (value,))
    return {name: get(port) for name, port in plan.outputs.items()}


def execute_display(
    plan, canvas, params=None, sources=None, *, expression_context=None
):
    """Execute the shared plan with the WebGPU display backend."""
    from ..gpu.plan import execute_display as gpu_execute_display

    return gpu_execute_display(
        plan,
        canvas,
        params,
        sources,
        expression_context=expression_context,
    )
