"""Read-only diagnostics for shared render graphs."""

from .lowering import GraphPlan, lower_graph


def inspect_graph(outputs, params=None):
    """Describe a plan, or report why lowering is unsupported without GPU work."""
    try:
        plan = outputs if isinstance(outputs, GraphPlan) else lower_graph(outputs)
        if params is not None:
            missing = sorted(
                name for name in plan.required_parameters if name not in params
            )
            if missing:
                raise ValueError("Missing render parameters: " + ", ".join(missing))
        for op in plan.passes:
            if op.contract.name != op.shader.name or op.contract.path != op.shader.path:
                raise ValueError(
                    f"Shader '{op.shader.name}' contract name or path does not match"
                )
            op.shader.operation.forward_source(color_format="rgba32f")
            op.shader.operation.backward_source()
    except (ValueError, TypeError, KeyError, OSError) as error:
        return {
            "supported": False,
            "error": str(error),
            "error_type": type(error).__name__,
        }
    return {
        "supported": True,
        "outputs": {
            name: {"kind": port.kind, "mode": port.mode}
            for name, port in plan.outputs.items()
        },
        "parameter_names": list(plan.parameter_names),
        "required_parameters": list(plan.required_parameters),
        "sources": len(plan.sources),
        "topology_controls": [
            {"operation": name, "fields": list(fields)}
            for name, fields in plan.topology_controls
        ],
        "operations": [
            {
                "shader": op.shader.name,
                "behavior": op.contract.behavior,
                "sampling": op.contract.sampling,
                "zero_gradient_uniforms": list(op.contract.zero_gradient_parameters),
                "notes": op.contract.notes,
            }
            for op in plan.passes
        ],
    }
