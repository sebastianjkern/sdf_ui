"""Display executor for typed render plans on WebGPU."""

from __future__ import annotations

from typing import Any, cast

from ..core.color import ColorTexture
from ..core.sdf import SDFTexture
from ..graph.execution import _resolve
from ..graph.lowering import GraphPlan


def execute_display(
    plan: GraphPlan,
    canvas,
    params=None,
    sources=None,
    *,
    expression_context=None,
):
    """Execute every graph-plan pass into caller-owned display textures."""
    if not isinstance(plan, GraphPlan):
        raise TypeError("Expected a GraphPlan")
    canvas._check_open()
    params = {} if params is None else params
    sources = plan.sources if sources is None else tuple(sources)
    if len(sources) != len(plan.sources):
        raise ValueError("Display source count differs from the plan")
    for source in sources:
        if source.context is not canvas or tuple(source.tex.size) != tuple(canvas.size):
            raise ValueError("Plan sources must belong to the canvas and match its size")

    results: list[tuple[Any, ...]] = []

    def get(port):
        if port.pass_index < 0:
            return sources[-port.pass_index - 1]
        return results[port.pass_index][port.output_index]

    for operation in plan.passes:
        contract = operation.contract
        scope = canvas if expression_context is None else expression_context
        uniforms = operation.uniforms(lambda value: _resolve(value, scope, params))
        outputs = tuple(
            canvas.r32f()
            if argument.kind == "r32f"
            else canvas.rgba8()
            for argument in contract.outputs
        )
        input_images = {
            argument.name: get(port).tex
            for argument, port in zip(contract.inputs, operation.inputs)
        }
        try:
            canvas.run_node_shader(
                operation.shader,
                outputs[0],
                input_images,
                uniforms,
                extra_outputs=outputs[1:],
            )
        except BaseException:
            for texture in outputs:
                texture.close()
            raise
        modes = {
            port.output_index: port.mode
            for port in plan.outputs.values()
            if port.pass_index == len(results)
        }
        results.append(
            tuple(
                cast(Any, SDFTexture)(tex=texture, context=canvas)
                if argument.kind == "r32f"
                else cast(Any, ColorTexture)(
                    tex=texture,
                    context=canvas,
                    mode=modes.get(index) or "LAB",
                )
                for index, (argument, texture) in enumerate(zip(contract.outputs, outputs))
            )
        )

    return {name: get(port) for name, port in plan.outputs.items()}
