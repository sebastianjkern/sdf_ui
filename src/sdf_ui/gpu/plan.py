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
    _validate_display_sources(canvas, sources)

    results: list[tuple[Any, ...]] = []
    for operation in plan.passes:
        results.append(
            _execute_display_operation(
                operation,
                plan,
                canvas,
                params,
                sources,
                results,
                expression_context,
            )
        )
    return {
        name: _display_port(port, results, sources)
        for name, port in plan.outputs.items()
    }


def _validate_display_sources(canvas, sources):
    for source in sources:
        full_canvas = tuple(source.tex.size) == tuple(canvas.size) and tuple(
            getattr(source.tex, "origin", (0, 0))
        ) == (0, 0)
        cropped_canvas = (
            canvas._active_evaluation_rect is not None
            and tuple(getattr(source.tex, "canvas_size", ())) == tuple(canvas.size)
            and tuple(getattr(source.tex, "origin", ()))
            == canvas._active_evaluation_rect[:2]
            and tuple(source.tex.size) == canvas._active_evaluation_rect[2:]
        )
        if source.context is not canvas or not (full_canvas or cropped_canvas):
            raise ValueError(
                "Plan sources must belong to the canvas and match its size"
            )


def _execute_display_operation(
    operation, plan, canvas, params, sources, results, expression_context
):
    contract = operation.contract
    scope = canvas if expression_context is None else expression_context
    uniforms = operation.uniforms(lambda value: _resolve(value, scope, params))
    outputs = _allocate_display_outputs(contract.outputs, canvas)
    inputs = {
        argument.name: _display_port(port, results, sources).tex
        for argument, port in zip(contract.inputs, operation.inputs)
    }
    try:
        canvas.run_node_shader(
            operation.shader,
            outputs[0],
            inputs,
            uniforms,
            extra_outputs=outputs[1:],
        )
    except BaseException:
        for texture in outputs:
            texture.close()
        raise
    modes = _output_modes(plan, len(results))
    return _wrap_display_outputs(contract.outputs, outputs, modes, canvas)


def _display_port(port, results, sources):
    if port.pass_index < 0:
        return sources[-port.pass_index - 1]
    return results[port.pass_index][port.output_index]


def _allocate_display_outputs(arguments, canvas):
    return tuple(
        canvas.r32f() if argument.kind == "r32f" else canvas.rgba8()
        for argument in arguments
    )


def _output_modes(plan, pass_index):
    return {
        port.output_index: port.mode
        for port in plan.outputs.values()
        if port.pass_index == pass_index
    }


def _wrap_display_outputs(arguments, outputs, modes, canvas):
    wrapped = []
    for index, (argument, texture) in enumerate(zip(arguments, outputs)):
        if argument.kind == "r32f":
            wrapped.append(cast(Any, SDFTexture)(tex=texture, context=canvas))
        else:
            wrapped.append(
                cast(Any, ColorTexture)(
                    tex=texture,
                    context=canvas,
                    mode=modes.get(index) or "LAB",
                )
            )
    return tuple(wrapped)
