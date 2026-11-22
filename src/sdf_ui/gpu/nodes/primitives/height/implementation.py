"""Height surface reference operations and differentiation contracts."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation


def _ramp(context, inputs, params):
    from sdf_ui.learning.render import _tensor

    origin = _tensor(params["origin"])
    slope = _tensor(params["slope"])
    return (context.x - origin[0]) * slope[0] + (context.y - origin[1]) * slope[1]


def _gaussian(context, inputs, params):
    from sdf_ui.learning.render import _tensor

    center = _tensor(params["center"])
    sigma = _tensor(params["sigma"])
    dx = (context.x - center[0]) / sigma[0]
    dy = (context.y - center[1]) / sigma[1]
    return _tensor(params["amplitude"]) * (-(dx * dx + dy * dy) * 0.5).exp()


def _add(context, inputs, params):
    return inputs["height0"] + inputs["height1"]


def _contract(name):
    if name == "height_ramp":
        parameters = (
            ShaderArgument("origin", (2,), "vec2"),
            ShaderArgument("slope", (2,), "vec2"),
        )
        reference = _ramp
    elif name == "height_gaussian":
        parameters = (
            ShaderArgument("center", (2,), "vec2"),
            ShaderArgument("sigma", (2,), "vec2"),
            ShaderArgument("amplitude", (), "float"),
        )
        reference = _gaussian
    else:
        parameters = ()
        reference = _add
    inputs = (
        ()
        if name != "height_add"
        else (
            ShaderArgument("height0", (), "r32f", port_kind="height"),
            ShaderArgument("height1", (), "r32f", port_kind="height"),
        )
    )
    return ShaderDifferentiation(
        name,
        f"gpu/nodes/primitives/height/{name}.wgsl",
        inputs,
        parameters,
        (ShaderArgument("destTex", (), "r32f", port_kind="height"),),
        behavior="smooth_for_fixed_sampling",
        notes="Pointwise height operation with continuous parameter gradients.",
        reference=reference,
    )


def _interface(name):
    images = [ImageBinding("destTex", 0, write=True)]
    uniforms = ()
    if name == "height_add":
        images.extend(
            (
                ImageBinding("height0", 1, read=True),
                ImageBinding("height1", 2, read=True),
            )
        )
    elif name == "height_ramp":
        uniforms = (Uniform("origin", "vec2"), Uniform("slope", "vec2"))
    else:
        uniforms = (
            Uniform("center", "vec2"),
            Uniform("sigma", "vec2"),
            Uniform("amplitude", "float"),
        )
    return ShaderInterface(images[0], tuple(images[1:]), uniforms)


CONTRACTS = {
    name: _contract(name) for name in ("height_ramp", "height_gaussian", "height_add")
}
INTERFACES = {name: _interface(name) for name in CONTRACTS}
