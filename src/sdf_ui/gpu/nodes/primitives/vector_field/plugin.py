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

_CENTER = "center"
_RADIUS = "radius"
_STRENGTH = "strength"
_VECTOR = "vector"
_AMOUNT = "amount"
_OCTAVES = "octaves"
_FREQUENCY = "frequency"
_AMPLITUDE = "amplitude"
_LACUNARITY = "lacunarity"
_FLOAT = "float"
_FIELD0 = "field0"


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
    center = _vector_uniform(params[_CENTER], _CENTER)
    radius = _finite_uniform(params[_RADIUS], _RADIUS, positive=True)
    strength = _finite_uniform(params[_STRENGTH], _STRENGTH)
    return {_CENTER: center, _RADIUS: radius, _STRENGTH: strength}


def _constant_uniforms(params):
    vector = _vector_uniform(params[_VECTOR], _VECTOR)
    return {_VECTOR: vector}


def _scale_uniforms(params):
    amount = _finite_uniform(params[_AMOUNT], _AMOUNT)
    return {_AMOUNT: amount}


def _turbulence_uniforms(params):
    for name in (_OCTAVES, "seed"):
        value = params[name]
        scalar = _scalar_value(value)
        if isinstance(scalar, bool) or int(scalar) != scalar:
            raise ValueError(f"{name} must be an integer")

    frequency = _finite_uniform(params[_FREQUENCY], _FREQUENCY, positive=True)
    amplitude = _finite_uniform(params[_AMPLITUDE], _AMPLITUDE)
    lacunarity = _finite_uniform(params[_LACUNARITY], _LACUNARITY, positive=True)
    gain = _finite_uniform(params["gain"], "gain")
    octaves, seed = int(_scalar_value(params[_OCTAVES])), int(_scalar_value(params["seed"]))
    if octaves not in range(1, 9):
        raise ValueError("frequency and lacunarity must be positive; octaves must be 1..8")
    return {
        _FREQUENCY: frequency,
        _AMPLITUDE: amplitude,
        _OCTAVES: octaves,
        _LACUNARITY: lacunarity,
        "gain": gain,
        "seed": seed,
    }


def register_plugins(registry):
    field_type = TextureKind.FIELD
    for spec in (
        (
            "constant_field",
            (Uniform(_VECTOR, "vec2"),),
            {_VECTOR: (0.0, 0.0)},
            _constant_uniforms,
        ),
        (
            "radial_field",
            (Uniform(_CENTER, "vec2"), Uniform(_RADIUS, _FLOAT), Uniform(_STRENGTH, _FLOAT)),
            {_STRENGTH: 1.0},
            _center_uniforms,
        ),
        (
            "vortex_field",
            (Uniform(_CENTER, "vec2"), Uniform(_RADIUS, _FLOAT), Uniform(_STRENGTH, _FLOAT)),
            {_STRENGTH: 1.0},
            _center_uniforms,
        ),
        (
            "turbulence_field",
            (
                Uniform(_FREQUENCY, _FLOAT), Uniform(_AMPLITUDE, _FLOAT),
                Uniform(_OCTAVES, "int"), Uniform(_LACUNARITY, _FLOAT),
                Uniform("gain", _FLOAT), Uniform("seed", "int"),
            ),
            {"frequency": 0.012, "amplitude": 24.0, "octaves": 4, "lacunarity": 2.0, "gain": 0.5, "seed": 0},
            _turbulence_uniforms,
        ),
    ):
        _register_source_field(registry, field_type, spec)
    _register_field_combinators(registry, field_type)


def _register_source_field(registry, field_type, spec):
    name, uniforms, defaults, make_uniforms = spec
    parameters = {
        "constant_field": (_VECTOR,),
        "radial_field": (_CENTER, _RADIUS, _STRENGTH),
        "vortex_field": (_CENTER, _RADIUS, _STRENGTH),
        "turbulence_field": (_FREQUENCY, _AMPLITUDE, _OCTAVES, _LACUNARITY, "gain", "seed"),
    }[name]
    registry.register(Plugin(
        name, PluginFamily.PRIMITIVE, field_type, params=parameters,
        defaults=defaults,
        shader=_field_shader(name, uniforms),
        make_uniforms=make_uniforms,
        public=True,
    ))


def _register_field_combinators(registry, field_type):
    for name, params, uniforms, inputs, defaults in (
        ("field_add", (), (), (field_type, field_type), {}),
        ("field_scale", (_AMOUNT,), (Uniform(_AMOUNT, _FLOAT),), (field_type,), {_AMOUNT: 1.0}),
    ):
        input_names = (_FIELD0, "field1") if name == "field_add" else (_FIELD0,)
        registry.register(Plugin(
            name, PluginFamily.PRIMITIVE, field_type, inputs, params=params,
            defaults=defaults,
            shader=_field_shader(name, uniforms, input_names),
            input_uniforms=input_names,
            make_uniforms=_scale_uniforms if name == "field_scale" else None,
            method_of=(field_type,),
        ))
