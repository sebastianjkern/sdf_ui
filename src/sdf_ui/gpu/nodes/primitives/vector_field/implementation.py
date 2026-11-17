"""Reference behavior and shader contracts for floating-point vector fields."""

import numpy as np

from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation

_FIELD = ShaderArgument("field0", (4,), "rgba32float")


def _reference(name, context, inputs, params):
    from sdf_ui.learning.autodiff import Tensor, stack
    from sdf_ui.learning.render import _tensor

    if name == "constant_field":
        vector = params["vector"]
        x, y = context.x * 0 + vector[0], context.y * 0 + vector[1]
    elif name in {"radial_field", "vortex_field"}:
        center = params["center"]
        dx, dy = context.x - center[0], context.y - center[1]
        radius = _tensor(params["radius"])
        attenuation = (-(dx * dx + dy * dy) / (2 * radius * radius)).exp()
        if name == "radial_field":
            x = dx / radius * params["strength"] * attenuation
            y = dy / radius * params["strength"] * attenuation
        else:
            x = -dy / radius * params["strength"] * attenuation
            y = dx / radius * params["strength"] * attenuation
    elif name == "turbulence_field":
        value = _noise_array(context.x.shape, context.x.data, context.y.data, params)
        return Tensor(value)
    elif name == "field_add":
        return inputs["field0"] + inputs["field1"]
    elif name == "field_scale":
        return inputs["field0"] * _tensor(params["amount"])
    else:
        raise ValueError(f"Unknown field operation {name}")
    return stack((x, y, x * 0, x * 0), axis=-1)


def _hash(x, y, seed):
    x, y = np.asarray(x, dtype=np.int32), np.asarray(y, dtype=np.int32)
    with np.errstate(over="ignore"):
        value = x.astype(np.uint32) * np.uint32(0x8DA6B343)
        value ^= y.astype(np.uint32) * np.uint32(0xD8163841)
        value ^= np.uint32(seed & 0xFFFFFFFF) * np.uint32(0xCB1AB31F)
        value ^= value >> np.uint32(16)
        value *= np.uint32(0x7FEB352D)
        value ^= value >> np.uint32(15)
        value *= np.uint32(0x846CA68B)
        value ^= value >> np.uint32(16)
    return (value & np.uint32(0x00FFFFFF)).astype(np.float32) / np.float32(16777215.0)


def _value_noise(x, y, seed):
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = x - ix.astype(np.float32), y - iy.astype(np.float32)
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    a = _hash(ix, iy, seed)
    b = _hash(ix + 1, iy, seed)
    c = _hash(ix, iy + 1, seed)
    d = _hash(ix + 1, iy + 1, seed)
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy) * 2 - 1


def _noise_array(shape, x, y, params):
    result_x = np.zeros(shape, dtype=np.float32)
    result_y = np.zeros(shape, dtype=np.float32)
    frequency = np.float32(params["frequency"].item())
    amplitude = np.float32(params["amplitude"].item())
    octaves = int(params["octaves"].item())
    lacunarity = np.float32(params["lacunarity"].item())
    gain = np.float32(params["gain"].item())
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    seed = int(params["seed"].item())
    weight = 1.0
    total = 0.0
    for octave in range(octaves):
        f = np.float32(frequency * lacunarity**octave)
        a = weight
        result_x += _value_noise(x * f, y * f, seed + octave) * a
        result_y += _value_noise((x + 37.1) * f, (y - 19.7) * f, seed + octave + 101) * a
        total += a
        weight *= gain
    result_x *= amplitude / total
    result_y *= amplitude / total
    return np.stack((result_x, result_y, np.zeros_like(result_x), np.zeros_like(result_x)), axis=-1)


def _params_for(name):
    return {
        "constant_field": ("vector",),
        "radial_field": ("center", "radius", "strength"),
        "vortex_field": ("center", "radius", "strength"),
        "turbulence_field": ("frequency", "amplitude", "octaves", "lacunarity", "gain", "seed"),
        "field_add": (),
        "field_scale": ("amount",),
    }[name]


def contract_for(name):
    inputs = ()
    if name == "field_add":
        inputs = (ShaderArgument("field0", (4,), "rgba32float"), ShaderArgument("field1", (4,), "rgba32float"))
    elif name == "field_scale":
        inputs = (_FIELD,)
    params = tuple(
        ShaderArgument(item, (), "int" if item in {"octaves", "seed"} else "float")
        for item in _params_for(name)
    )
    if name == "constant_field":
        params = (ShaderArgument("vector", (2,), "vec2"),)
    if name in {"radial_field", "vortex_field"}:
        params = (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radius", (), "float"), ShaderArgument("strength", (), "float"))
    def reference(context, images, values):
        return _reference(name, context, images, values)
    zero = _params_for(name) if name == "turbulence_field" else ()
    return ShaderDifferentiation(
        name,
        f"gpu/nodes/primitives/vector_field/{name}.glsl",
        inputs,
        params,
        (ShaderArgument("destTex", (4,), "rgba32float"),),
        zero_gradient_parameters=zero,
        behavior="constant" if zero else "smooth_for_fixed_sampling",
        sampling="pointwise",
        reference=reference,
        defaults={
            "constant_field": {"vector": (0.0, 0.0)},
            "radial_field": {"strength": 1.0},
            "vortex_field": {"strength": 1.0},
            "turbulence_field": {"frequency": 0.012, "amplitude": 24.0, "octaves": 4, "lacunarity": 2.0, "gain": 0.5, "seed": 0},
            "field_add": {},
            "field_scale": {"amount": 1.0},
        }[name],
    )
