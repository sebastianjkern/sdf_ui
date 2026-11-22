"""Height surface plugin registrations."""

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from ..vector_field.plugin import _finite_uniform, _scalar_value, _vector_uniform
from .implementation import CONTRACTS, INTERFACES


def _ramp_uniforms(params):
    return {
        "origin": _vector_uniform(params["origin"], "origin"),
        "slope": _vector_uniform(params["slope"], "slope"),
    }


def _gaussian_uniforms(params):
    sigma = _vector_uniform(params["sigma"], "sigma")
    axes = sigma.numpy() if hasattr(sigma, "numpy") else sigma
    if any(_scalar_value(axis) <= 0 for axis in axes):
        raise ValueError("sigma must contain two positive values")
    return {
        "center": _vector_uniform(params["center"], "center"),
        "sigma": sigma,
        "amplitude": _finite_uniform(params["amplitude"], "amplitude"),
    }


def register_plugins(registry):
    registry.register(
        Plugin(
            "height_ramp",
            PluginFamily.PRIMITIVE,
            TextureKind.HEIGHT,
            params=("origin", "slope"),
            shader=shader(
                "height_ramp",
                "primitives/height/height_ramp.wgsl",
                derivative_contract=CONTRACTS["height_ramp"],
                interface=INTERFACES["height_ramp"],
            ),
            make_uniforms=_ramp_uniforms,
            public=True,
        )
    )
    registry.register(
        Plugin(
            "height_gaussian",
            PluginFamily.PRIMITIVE,
            TextureKind.HEIGHT,
            params=("center", "sigma", "amplitude"),
            shader=shader(
                "height_gaussian",
                "primitives/height/height_gaussian.wgsl",
                derivative_contract=CONTRACTS["height_gaussian"],
                interface=INTERFACES["height_gaussian"],
            ),
            make_uniforms=_gaussian_uniforms,
            public=True,
        )
    )
    registry.register(
        Plugin(
            "height_add",
            PluginFamily.PRIMITIVE,
            TextureKind.HEIGHT,
            (TextureKind.HEIGHT, TextureKind.HEIGHT),
            shader=shader(
                "height_add",
                "primitives/height/height_add.wgsl",
                derivative_contract=CONTRACTS["height_add"],
                interface=INTERFACES["height_add"],
            ),
            input_uniforms=("height0", "height1"),
            method_of=(TextureKind.HEIGHT,),
        )
    )
