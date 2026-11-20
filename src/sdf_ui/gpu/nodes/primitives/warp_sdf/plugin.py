"""GPU resampling of SDFs and color images by vector fields."""




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

_STEPS_TEXT = 'steps'

_STRENGTH_TEXT = 'strength'

__docformat__ = "google"

def _steps(params):
    steps = params[_STEPS_TEXT]
    if isinstance(steps, bool) or int(steps) != steps or not 1 <= steps <= 32:
        raise ValueError("steps must be an integer from 1 to 32")
    import math

    strength = float(params[_STRENGTH_TEXT])
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")
    return {_STRENGTH_TEXT: strength, _STEPS_TEXT: int(steps)}


def register_plugins(registry):
    sdf_descriptor = shader(
        "warp_sdf", "primitives/warp_sdf/shader.glsl", derivative_contract=SDF_CONTRACT,
        interface=ShaderInterface(
            ImageBinding("destTex", 0, write=True),
            (ImageBinding("sourceTex", 1, read=True), ImageBinding("fieldTex", 2, read=True)),
            (Uniform(_STRENGTH_TEXT, "float"), Uniform(_STEPS_TEXT, "int")),
        ),
    )
    color_descriptor = shader(
        "warp_color", "shading/warp_color/shader.glsl", derivative_contract=COLOR_CONTRACT,
        interface=ShaderInterface(
            ImageBinding("destTex", 0, write=True),
            (ImageBinding("sourceTex", 1, read=True), ImageBinding("fieldTex", 2, read=True)),
            (Uniform(_STRENGTH_TEXT, "float"), Uniform(_STEPS_TEXT, "int")),
        ),
    )
    registry.register(
        Plugin(
            "warp_sdf", PluginFamily.PRIMITIVE, TextureKind.SDF,
            (TextureKind.SDF, TextureKind.FIELD), params=(_STRENGTH_TEXT, _STEPS_TEXT),
            defaults={_STRENGTH_TEXT: 1.0, _STEPS_TEXT: 1}, shader=sdf_descriptor,
            input_uniforms=("sourceTex", "fieldTex"), make_uniforms=_steps,
            method_of=(TextureKind.SDF,),
        )
    )
    registry.register(
        Plugin(
            "warp_color", PluginFamily.SHADING, TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.FIELD), params=(_STRENGTH_TEXT, _STEPS_TEXT),
            defaults={_STRENGTH_TEXT: 1.0, _STEPS_TEXT: 1}, shader=color_descriptor,
            input_uniforms=("sourceTex", "fieldTex"), make_uniforms=_steps,
            method_of=(TextureKind.COLOR,),
        )
    )
