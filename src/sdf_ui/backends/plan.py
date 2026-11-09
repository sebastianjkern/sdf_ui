"""Display and CPU-reference executors for the shared typed render plan."""

import numpy as np

from ..core.color import ColorTexture
from ..core.expressions import Expr, evaluate_expr
from ..core.sdf import SDFTexture
from ..graph.lowering import GraphPlan
from ..learning.autodiff import Tensor, as_tensor
from ..learning.shader_derivatives import _implementation
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
    """Execute a GraphPlan using normal r32f/rgba8 display shader storage."""
    if not isinstance(plan, GraphPlan):
        raise TypeError("Expected a GraphPlan")
    if canvas._closed:
        raise RuntimeError("Canvas is closed")
    params = {} if params is None else params
    sources = plan.sources if sources is None else tuple(sources)
    if len(sources) != len(plan.sources):
        raise ValueError("Display source count differs from the plan")
    for source in sources:
        if source.context is not canvas or tuple(source.tex.size) != tuple(canvas.size):
            raise ValueError(
                "Plan sources must belong to the canvas and match its size"
            )
    results = []

    def get(port):
        if port.pass_index < 0:
            return sources[-port.pass_index - 1]
        return results[port.pass_index][port.output_index]

    for operation in plan.passes:
        contract = operation.contract
        program = canvas.get_shader(operation.shader.name)
        outputs = []
        for argument in contract.outputs:
            if argument.kind == "r32f":
                outputs.append(SDFTexture(tex=canvas.r32f(), context=canvas))
            else:
                # Select the output mode from the plan's ports where available.
                mode = next(
                    (
                        p.mode
                        for p in plan.outputs.values()
                        if p.pass_index == len(results)
                        and p.output_index == len(outputs)
                    ),
                    None,
                )
                outputs.append(
                    ColorTexture(tex=canvas.rgba8(), context=canvas, mode=mode or "LAB")
                )
        location = 0
        for argument, texture in zip(contract.outputs, outputs):
            texture.tex.bind_to_image(location, read=False, write=True)
            if argument.name in program:
                program[argument.name].value = location
            location += 1
        for argument, port in zip(contract.inputs, operation.inputs):
            get(port).tex.bind_to_image(location, read=True, write=False)
            if argument.name in program:
                program[argument.name].value = location
            location += 1
        scope = canvas if expression_context is None else expression_context
        uniforms = operation.uniforms(lambda v: _resolve(v, scope, params))
        _, defaults = _implementation(contract)
        for argument in contract.parameters:
            if argument.name not in program:
                continue
            value = uniforms.get(
                argument.name, defaults.get(argument.name, np.zeros(argument.shape))
            )
            if isinstance(value, Tensor):
                value = value.numpy()
            dtype = np.int32 if argument.kind in {"int", "bool"} else np.float32
            data = np.asarray(value, dtype=dtype)
            if data.shape != argument.shape or not np.all(np.isfinite(data)):
                raise ValueError(f"Invalid uniform '{argument.name}'")
            program[argument.name].write(data.tobytes())
        shader = operation.shader.operation
        local_x, local_y, _ = shader.workgroup_size(shader.forward_source())
        program.run(
            (canvas.size[0] + local_x - 1) // local_x,
            (canvas.size[1] + local_y - 1) // local_y,
            1,
        )
        canvas._mgl_ctx.memory_barrier()
        results.append(tuple(outputs))
    return {name: get(port) for name, port in plan.outputs.items()}
