"""GPU vector-field factories and composition."""

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


def _field_shader(name, uniforms=(), input_names=()):
    from .implementation import contract_for

    return shader(
        name,
        f"primitives/vector_field/{name}.glsl",
        derivative_contract=contract_for(name),
        interface=ShaderInterface(
            ImageBinding("destTex", 0, write=True),
            tuple(ImageBinding(input_name, index + 1, read=True) for index, input_name in enumerate(input_names)),
            uniforms=tuple(uniforms),
        ),
    )


def _center_uniforms(params):
    import numpy as np

    center = np.asarray(params["center"], dtype=np.float32)
    radius, strength = float(params["radius"]), float(params["strength"])
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must contain two finite values")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(strength):
        raise ValueError("strength must be finite")
    return {"center": center, "radius": radius, "strength": strength}


def _constant_uniforms(params):
    import numpy as np

    vector = np.asarray(params["vector"], dtype=np.float32)
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError("vector must contain two finite values")
    return {"vector": vector}


def _scale_uniforms(params):
    import math

    amount = float(params["amount"])
    if not math.isfinite(amount):
        raise ValueError("amount must be finite")
    return {"amount": amount}


def _turbulence_uniforms(params):
    import math

    for name in ("octaves", "seed"):
        value = params[name]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be an integer")

    values = {
        "frequency": float(params["frequency"]),
        "amplitude": float(params["amplitude"]),
        "octaves": int(params["octaves"]),
        "lacunarity": float(params["lacunarity"]),
        "gain": float(params["gain"]),
        "seed": int(params["seed"]),
    }
    if not all(math.isfinite(values[key]) for key in ("frequency", "amplitude", "lacunarity", "gain")):
        raise ValueError("turbulence parameters must be finite")
    if values["frequency"] <= 0 or values["lacunarity"] <= 0 or values["octaves"] not in range(1, 9):
        raise ValueError("frequency and lacunarity must be positive; octaves must be 1..8")
    return values


def register_plugins(registry):
    field_type = TextureKind.FIELD
    for name, uniforms, defaults, make_uniforms in (
        (
            "constant_field",
            (Uniform("vector", "vec2"),),
            {"vector": (0.0, 0.0)},
            _constant_uniforms,
        ),
        (
            "radial_field",
            (Uniform("center", "vec2"), Uniform("radius", "float"), Uniform("strength", "float")),
            {"strength": 1.0},
            _center_uniforms,
        ),
        (
            "vortex_field",
            (Uniform("center", "vec2"), Uniform("radius", "float"), Uniform("strength", "float")),
            {"strength": 1.0},
            _center_uniforms,
        ),
        (
            "turbulence_field",
            (
                Uniform("frequency", "float"), Uniform("amplitude", "float"),
                Uniform("octaves", "int"), Uniform("lacunarity", "float"),
                Uniform("gain", "float"), Uniform("seed", "int"),
            ),
            {"frequency": 0.012, "amplitude": 24.0, "octaves": 4, "lacunarity": 2.0, "gain": 0.5, "seed": 0},
            _turbulence_uniforms,
        ),
    ):
        params = {
            "constant_field": ("vector",),
            "radial_field": ("center", "radius", "strength"),
            "vortex_field": ("center", "radius", "strength"),
            "turbulence_field": ("frequency", "amplitude", "octaves", "lacunarity", "gain", "seed"),
        }[name]
        registry.register(
            Plugin(
                name, PluginFamily.PRIMITIVE, field_type, params=params,
                defaults=defaults,
                shader=_field_shader(
                    name,
                    uniforms,
                    ("field0", "field1") if name == "field_add" else ("field0",) if name == "field_scale" else (),
                ),
                make_uniforms=make_uniforms,
                public=True,
            )
        )

    for name, params, uniforms, inputs, defaults in (
        ("field_add", (), (), (field_type, field_type), {}),
        ("field_scale", ("amount",), (Uniform("amount", "float"),), (field_type,), {"amount": 1.0}),
    ):
        registry.register(
            Plugin(
                name, PluginFamily.PRIMITIVE, field_type, inputs, params=params,
                defaults=defaults,
                shader=_field_shader(name, uniforms, ("field0", "field1") if name == "field_add" else ("field0",)),
                input_uniforms=("field0", "field1") if name == "field_add" else ("field0",),
                make_uniforms=_scale_uniforms if name == "field_scale" else None,
                method_of=(field_type,),
            )
        )
