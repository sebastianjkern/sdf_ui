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


def _scalar_value(value):
    return value.item() if hasattr(value, "item") else value


def _finite_uniform(value, name, *, positive=False):
    import math

    result = float(_scalar_value(value))
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return value if hasattr(value, "requires_grad") else result


def _vector_uniform(value, name):
    import numpy as np

    if hasattr(value, "numpy"):
        values = np.asarray(value.numpy(), dtype=np.float32)
        output = value
    else:
        try:
            items = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain two finite values") from exc
        values = np.asarray([_scalar_value(item) for item in items], dtype=np.float32)
        output = items if any(hasattr(item, "requires_grad") for item in items) else values
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain two finite values")
    return output


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
    center = _vector_uniform(params["center"], "center")
    radius = _finite_uniform(params["radius"], "radius", positive=True)
    strength = _finite_uniform(params["strength"], "strength")
    return {"center": center, "radius": radius, "strength": strength}


def _constant_uniforms(params):
    vector = _vector_uniform(params["vector"], "vector")
    return {"vector": vector}


def _scale_uniforms(params):
    amount = _finite_uniform(params["amount"], "amount")
    return {"amount": amount}


def _turbulence_uniforms(params):
    for name in ("octaves", "seed"):
        value = params[name]
        scalar = _scalar_value(value)
        if isinstance(scalar, bool) or int(scalar) != scalar:
            raise ValueError(f"{name} must be an integer")

    frequency = _finite_uniform(params["frequency"], "frequency", positive=True)
    amplitude = _finite_uniform(params["amplitude"], "amplitude")
    lacunarity = _finite_uniform(params["lacunarity"], "lacunarity", positive=True)
    gain = _finite_uniform(params["gain"], "gain")
    octaves, seed = int(_scalar_value(params["octaves"])), int(_scalar_value(params["seed"]))
    if octaves not in range(1, 9):
        raise ValueError("frequency and lacunarity must be positive; octaves must be 1..8")
    return {
        "frequency": frequency,
        "amplitude": amplitude,
        "octaves": octaves,
        "lacunarity": lacunarity,
        "gain": gain,
        "seed": seed,
    }


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
