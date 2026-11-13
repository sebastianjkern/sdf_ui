__docformat__ = "google"

from math import isfinite

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import lower_composite, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def _render_transform(renderer, source, uniforms):
    from sdf_ui.core.sdf import SDFTexture

    ctx = renderer.ctx
    tex = ctx.r32f()

    from sdf_ui.gpu.nodes.registry import registry

    descriptor = next(
        item for item in registry.shader_files() if item.name == "transform"
    )
    try:
        ctx.run_node_shader(descriptor, tex, (source.tex,), uniforms)
    except BaseException:
        tex.close()
        raise
    return SDFTexture(tex=tex, context=ctx)


def _render_translate(renderer, inputs, params):
    return _render_transform(
        renderer,
        inputs[0],
        {"transform_kind": 0.0, "offset": params["offset"]},
    )


def _render_scale(renderer, inputs, params):
    factor = float(params["factor"])
    if not isfinite(factor) or abs(factor) < 1e-12:
        raise ValueError("scale factor must be finite and non-zero")

    return _render_transform(
        renderer,
        inputs[0],
        {
            "transform_kind": 1.0,
            "center": params["center"],
            "factor": factor,
        },
    )


def _render_rotate(renderer, inputs, params):
    return _render_transform(
        renderer,
        inputs[0],
        {
            "transform_kind": 2.0,
            "center": params["center"],
            "angle": float(params["angle"]),
        },
    )


def _render_skew(renderer, inputs, params):
    skew_x, skew_y = params["skew"]
    skew_x = float(skew_x)
    skew_y = float(skew_y)
    det = 1.0 - skew_x * skew_y
    if not isfinite(det) or abs(det) < 1e-12:
        raise ValueError("skew values produce a singular transform")

    return _render_transform(
        renderer,
        inputs[0],
        {
            "transform_kind": 3.0,
            "center": params["center"],
            "skew": (skew_x, skew_y),
        },
    )


def register_plugins(registry):
    descriptor = shader("transform", "primitives/transform/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE)
    registry.register(
        Plugin(
            "translate",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("offset",),
            shader=descriptor,
            render_func=_render_translate,
            lowering_func=lower_composite,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )
    registry.register(
        Plugin(
            "scale",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("factor", "center"),
            defaults={"center": (0.0, 0.0)},
            shader=descriptor,
            render_func=_render_scale,
            lowering_func=lower_composite,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )
    registry.register(
        Plugin(
            "rotate",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("angle", "center"),
            defaults={"center": (0.0, 0.0)},
            shader=descriptor,
            render_func=_render_rotate,
            lowering_func=lower_composite,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )
    registry.register(
        Plugin(
            "skew",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("skew", "center"),
            defaults={"center": (0.0, 0.0)},
            shader=descriptor,
            render_func=_render_skew,
            lowering_func=lower_composite,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )
