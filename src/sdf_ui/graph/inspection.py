"""Read-only diagnostics for shared render graphs."""

from pathlib import Path

from .lowering import GraphPlan, lower_graph


def _validate_plan(plan, params):
    if params is not None:
        missing = sorted(
            name for name in plan.required_parameters if name not in params
        )
        if missing:
            raise ValueError("Missing render parameters: " + ", ".join(missing))
    shader_dir = Path(__file__).resolve().parents[1] / "gpu" / "shaders"
    for operation in plan.passes:
        descriptor = operation.shader
        if (
            operation.contract.name != descriptor.name
            or operation.contract.path != descriptor.path
        ):
            raise ValueError(
                f"Shader '{descriptor.name}' contract name or path does not match"
            )
        forward = shader_dir / f"{descriptor.name}_rgba32f.wgsl"
        if not forward.is_file():
            forward = shader_dir / f"{descriptor.name}.wgsl"
        backward = shader_dir / f"{descriptor.name}_backward.wgsl"
        if not forward.is_file() or not backward.is_file():
            raise ValueError(
                f"WebGPU kernels for shader '{descriptor.name}' are incomplete"
            )


def _unsupported(error):
    return {
        "supported": False,
        "error": str(error),
        "error_type": type(error).__name__,
    }


def inspect_graph(outputs, params=None):
    """Describe a plan, or report why lowering is unsupported without GPU work."""
    try:
        plan = outputs if isinstance(outputs, GraphPlan) else lower_graph(outputs)
        _validate_plan(plan, params)
    except (ValueError, TypeError, KeyError, OSError) as error:
        return _unsupported(error)
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
