"""GPU resampling of SDFs and color images by vector fields."""

__docformat__ = "google"

from sdf_ui.gpu.nodes.base import (
    ImageBinding,
    Plugin,
    PluginFamily,
    ShaderInterface,
    TextureKind,
    Uniform,
)
from sdf_ui.gpu.nodes.common import shader

from .implementation import COLOR_CONTRACT, SDF_CONTRACT


def _steps(params):
    steps = params["steps"]
    if isinstance(steps, bool) or int(steps) != steps or not 1 <= steps <= 32:
        raise ValueError("steps must be an integer from 1 to 32")
    import math

    strength = float(params["strength"])
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")
    return {"strength": strength, "steps": int(steps)}


def register_plugins(registry):
    sdf_descriptor = shader(
        "warp_sdf", "primitives/warp_sdf/shader.glsl", derivative_contract=SDF_CONTRACT,
        interface=ShaderInterface(
            ImageBinding("destTex", 0, write=True),
            (ImageBinding("sourceTex", 1, read=True), ImageBinding("fieldTex", 2, read=True)),
            (Uniform("strength", "float"), Uniform("steps", "int")),
        ),
    )
    color_descriptor = shader(
        "warp_color", "shading/warp_color/shader.glsl", derivative_contract=COLOR_CONTRACT,
        interface=ShaderInterface(
            ImageBinding("destTex", 0, write=True),
            (ImageBinding("sourceTex", 1, read=True), ImageBinding("fieldTex", 2, read=True)),
            (Uniform("strength", "float"), Uniform("steps", "int")),
        ),
    )
    registry.register(
        Plugin(
            "warp_sdf", PluginFamily.PRIMITIVE, TextureKind.SDF,
            (TextureKind.SDF, TextureKind.FIELD), params=("strength", "steps"),
            defaults={"strength": 1.0, "steps": 1}, shader=sdf_descriptor,
            input_uniforms=("sourceTex", "fieldTex"), make_uniforms=_steps,
            method_of=(TextureKind.SDF,),
        )
    )
    registry.register(
        Plugin(
            "warp_color", PluginFamily.SHADING, TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.FIELD), params=("strength", "steps"),
            defaults={"strength": 1.0, "steps": 1}, shader=color_descriptor,
            input_uniforms=("sourceTex", "fieldTex"), make_uniforms=_steps,
            method_of=(TextureKind.COLOR,),
        )
    )
