"""Differentiable linear light scene nodes and display conversion."""

from functools import partial

from sdf_ui.gpu.nodes.base import (
    ImageBinding,
    Plugin,
    PluginFamily,
    ShaderFile,
    ShaderInterface,
    TextureKind,
    Uniform,
)
from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation

from .grain import WGSL as GRAIN_WGSL
from .grain import reference_noise
from .kernel import PointwiseKernel

_COLOR = TextureKind.COLOR
_SDF = TextureKind.SDF
_LINEAR_RGBA = "linear_rgba"
_SOURCE = "source"
_FLOAT_TYPE = "float"
_SRGB_BREAKPOINT = 0.04045
_SRGB_LINEAR_SCALE = 12.92
_SRGB_OFFSET = 0.055
_SRGB_SCALE = 1.055
_SRGB_GAMMA = 2.4
_LAB_MIN_LUMA = 1e-6
_SRGB_LINEAR_BREAKPOINT = 0.0031308


def _rgba(value):
    from sdf_ui.learning.render import _tensor

    value = _tensor(value)
    if value.shape != (4,):
        raise ValueError("Linear color requires an RGBA vector of four components")
    return value


def _uniforms(values, names):
    result = {}
    scalars = {}
    for name in names:
        if _is_color_uniform(name):
            result[name] = _validated_color_uniform(values[name])
            continue
        result[name], scalars[name] = _validated_scalar_uniform(name, values[name])
    if "inner" in scalars and scalars["outer"] <= scalars["inner"]:
        raise ValueError("Linear fill outer distance must exceed inner distance")
    return result


def _is_color_uniform(name):
    return name in {_LINEAR_RGBA, "linear_background"}


def _validated_color_uniform(value):
    import numpy as np
    tensor = _rgba(value)
    data = tensor.numpy()
    if not np.isfinite(data).all() or np.any(data[:3] < 0) or not 0 <= data[3] <= 1:
        raise ValueError("Linear RGBA requires finite nonnegative RGB and alpha in [0, 1]")
    return tensor


def _validated_scalar_uniform(name, value):
    import numpy as np
    scalar = float(value.item() if hasattr(value, "item") else value)
    if name == "grain_amount" and not 0 <= scalar <= 1:
        raise ValueError("Film grain amount must be in [0, 1]")
    if name == "grain_size" and not 1 <= scalar < float("inf"):
        raise ValueError("Film grain size must be finite and at least 1")
    nonnegative = {"threshold", "amount", "tone", "dither"}
    if not np.isfinite(scalar) or (name in nonnegative and scalar < 0):
        raise ValueError(f"{name} must be finite and, for finishing controls, nonnegative")
    return value, scalar


def _clip01(value):
    from sdf_ui.learning.autodiff import where
    return where(value.data <= 0, 0, where(value.data >= 1, 1, value))


def _reference(context, inputs, p, *, op):
    import numpy as np

    from sdf_ui.learning.autodiff import exp, where
    from sdf_ui.learning.render import _rgba as pack
    from sdf_ui.learning.render import _tensor

    base_result = _reference_base(context, inputs, p, op, _tensor, pack)
    if base_result is not None:
        return base_result
    source = _tensor(inputs[_SOURCE])
    rgb, alpha = source[..., :3], source[..., 3]
    operations = {
        "decode": lambda: (_decode(rgb, where), alpha),
        "over": lambda: _over(rgb, alpha, inputs, _tensor, where),
        "expose": lambda: (rgb * exp(_tensor(p["stops"]) * np.log(2)), alpha),
        "grain": lambda: _grain(rgb, alpha, context, p, _tensor, where),
        "extract": lambda: _extract(rgb, alpha, p, _tensor, where),
        "glow": lambda: _glow(rgb, alpha, inputs, p, context, _tensor, where),
        "finish": lambda: _finish(rgb, alpha, p, context, np, _tensor, where),
    }
    operation = operations.get(op)
    if operation is not None:
        rgb, alpha = operation()
    return pack(rgb, alpha)


def _reference_base(context, inputs, params, op, tensor, pack):
    if op == "clear":
        return _rgba(params[_LINEAR_RGBA]) + (context.x * 0)[..., None]
    if op == "ink":
        distance = tensor(inputs["sdf"])
        rgba = _rgba(params[_LINEAR_RGBA])
        return pack(rgba[:3] + (distance * 0)[..., None], rgba[3] * _clip01(0.5 - distance))
    if op == "fill":
        distance = tensor(inputs["sdf"])
        inner, outer = tensor(params["inner"]), tensor(params["outer"])
        weight = _clip01((distance - inner) / (outer - inner))
        weight = (weight * weight * (3 - 2 * weight))[..., None]
        foreground = _rgba(params[_LINEAR_RGBA])
        background = _rgba(params["linear_background"])
        return foreground + (background - foreground) * weight
    return None


def _decode(rgb, where):
    return where(
        rgb.data <= _SRGB_BREAKPOINT,
        rgb / _SRGB_LINEAR_SCALE,
        ((rgb + _SRGB_OFFSET) / _SRGB_SCALE).maximum(0) ** _SRGB_GAMMA,
    )


def _over(rgb, alpha, inputs, tensor, where):
    top = tensor(inputs["top"])
    top_alpha = top[..., 3]
    remaining = alpha * (1 - top_alpha)
    output_alpha = top_alpha + remaining
    denominator = where(output_alpha.data > 0, output_alpha, 1)
    output_rgb = (top[..., :3] * top_alpha[..., None] + rgb * remaining[..., None]) / denominator[..., None]
    output_rgb = where((output_alpha.data > 0)[..., None], output_rgb, 0)
    return output_rgb, output_alpha


def _grain(rgb, alpha, context, params, tensor, where):
    amount = tensor(params["grain_amount"])
    size = tensor(params["grain_size"])
    amount = where(amount.data < 0, 0, where(amount.data > 1, 1, amount))
    size = where(size.data < 1, 1, size)
    luma = (rgb * [0.2126, 0.7152, 0.0722]).sum(axis=-1)
    factor = 1 + amount * reference_noise(context, size) / (1 + 3 * luma)
    return rgb * factor[..., None], alpha


def _extract(rgb, alpha, params, tensor, where):
    luma = (rgb * [0.2126, 0.7152, 0.0722]).sum(axis=-1)
    excess = luma - tensor(params["threshold"])
    amount = where(excess.data > 0, excess, 0) / where(luma.data > _LAB_MIN_LUMA, luma, _LAB_MIN_LUMA)
    return rgb * (alpha * amount)[..., None], alpha * 0


def _glow(rgb, alpha, inputs, params, tensor, where):
    light = tensor(inputs["light"])[..., :3] * tensor(params["amount"])
    glow_alpha = _clip01((light * [0.2126, 0.7152, 0.0722]).sum(axis=-1))
    output_alpha = alpha + (1 - alpha) * glow_alpha
    denominator = where(output_alpha.data > 0, output_alpha, 1)
    output_rgb = where(
        (output_alpha.data > 0)[..., None],
        (rgb * alpha[..., None] + light) / denominator[..., None],
        0,
    )
    return output_rgb, output_alpha


def _finish(rgb, alpha, params, context, np, tensor, where):
    rgb = where(rgb.data > 0, rgb, 0)
    luma = (rgb * [0.2126, 0.7152, 0.0722]).sum(axis=-1)
    rgb = rgb / (1 + luma[..., None] * tensor(params["tone"]))
    rgb = where(
        rgb.data <= _SRGB_LINEAR_BREAKPOINT,
        rgb * _SRGB_LINEAR_SCALE,
        _SRGB_SCALE * rgb.maximum(_SRGB_LINEAR_BREAKPOINT) ** (1 / _SRGB_GAMMA) - _SRGB_OFFSET,
    )
    pattern = np.array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5])
    index = (context.y.data.astype(int) % 4) * 4 + context.x.data.astype(int) % 4
    noise = ((pattern[index] + 0.5) / 16 - 0.5) / 255
    return _clip01(rgb + tensor(params["dither"]) * noise[..., None]), alpha


_OPS = {
    "clear": (
        (),
        ((_LINEAR_RGBA, "vec4"),),
        "return array<D,4>(u[0],u[1],u[2],u[3]);",
    ),
    "ink": (
        (("sdf", _SDF),),
        ((_LINEAR_RGBA, "vec4"),),
        """
        let coverage=clip(sub(k(0.5),ins[0][0]));
        return array<D,4>(u[0],u[1],u[2],mul(u[3],coverage));
    """,
    ),
    "fill": (
        (("sdf", _SDF),),
        (
            (_LINEAR_RGBA, "vec4"),
            ("linear_background", "vec4"),
            ("inner", _FLOAT_TYPE),
            ("outer", _FLOAT_TYPE),
        ),
        """
        let t=clip(div(sub(ins[0][0],u[8]),sub(u[9],u[8])));
        let weight=mul(mul(t,t),sub(k(3.0),mul(k(2.0),t)));
        var out: Pixel;
        for (var i=0; i<4; i++) { out[i]=mix_d(u[i],u[i+4],weight); }
        return out;
    """,
    ),
    "decode": (
        ((_SOURCE, _COLOR),),
        (),
        """
        var out=ins[0];
        for (var i=0; i<3; i++) { out[i]=decode(out[i]); }
        return out;
    """,
    ),
    "over": (
        ((_SOURCE, _COLOR), ("top", _COLOR)),
        (),
        """
        let a=ins[0][3];
        let b=ins[1][3];
        let remaining=mul(a,sub(k(1.0),b));
        let alpha=add(b,remaining);
        var out: Pixel;
        if alpha.v>0.0 {
            for (var i=0; i<3; i++) {
                out[i]=div(add(mul(ins[0][i],remaining),mul(ins[1][i],b)),alpha);
            }
        }
        out[3]=alpha;
        return out;
    """,
    ),
    "expose": (
        ((_SOURCE, _COLOR),),
        (("stops", _FLOAT_TYPE),),
        """
        var out=ins[0];
        let multiplier=scale2(u[0]);
        for (var i=0; i<3; i++) { out[i]=mul(out[i],multiplier); }
        return out;
    """,
    ),
    "grain": (
        ((_SOURCE, _COLOR),),
        (("grain_amount", _FLOAT_TYPE), ("grain_size", _FLOAT_TYPE)),
        """
        var out=ins[0];
        var amount=u[0];
        if amount.v<0.0 || amount.v>1.0 { amount=k(clamp(amount.v,0.0,1.0)); }
        var size=u[1];
        if size.v<1.0 { size=k(1.0); }
        let gentle=div(amount,add(k(1.0),mul(k(3.0),luminance(out))));
        let factor=add(k(1.0),mul(gentle,grain_noise(pos,size)));
        for (var i=0; i<3; i++) { out[i]=mul(out[i],factor); }
        return out;
    """,
    ),
    "extract": (
        ((_SOURCE, _COLOR),),
        (("threshold", _FLOAT_TYPE),),
        """
        var out=ins[0];
        let luma=luminance(out);
        let safe=select(0.000001,luma.v,luma.v>0.000001);
        var denominator=k(safe);
        if luma.v>0.000001 { denominator=luma; }
        let weight=mul(out[3],div(positive(sub(luma,u[0])),denominator));
        for (var i=0; i<3; i++) { out[i]=mul(out[i],weight); }
        out[3]=k(0.0);
        return out;
    """,
    ),
    "glow": (
        ((_SOURCE, _COLOR), ("light", _COLOR)),
        (("amount", _FLOAT_TYPE),),
        """
        var out=ins[0];
        var light=ins[1];
        for (var i=0; i<3; i++) { light[i]=mul(light[i],u[0]); }
        let a=out[3];
        let alpha=add(a,mul(sub(k(1.0),a),clip(luminance(light))));
        for (var i=0; i<3; i++) {
            let energy=add(mul(out[i],a),light[i]);
            out[i]=k(0.0);
            if alpha.v>0.0 { out[i]=div(energy,alpha); }
        }
        out[3]=alpha;
        return out;
    """,
    ),
    "finish": (
        ((_SOURCE, _COLOR),),
        (("tone", _FLOAT_TYPE), ("dither", _FLOAT_TYPE)),
        """
        var out=ins[0];
        for (var i=0; i<3; i++) { out[i]=positive(out[i]); }
        let denominator=add(k(1.0),mul(u[0],luminance(out)));
        let noise=mul(u[1],k(bayer(pos)));
        for (var i=0; i<3; i++) { out[i]=clip(add(encode(div(out[i],denominator)),noise)); }
        return out;
    """,
    ),
}


def register_plugins(registry):
    for op, (inputs, parameters, body) in _OPS.items():
        _register_operation(registry, op, inputs, parameters, body)


def _register_operation(registry, op, inputs, parameters, body):
    name = f"hdr_{op}"
    path = f"gpu/shaders/{name}.wgsl"
    contract = _shader_contract(name, path, op, inputs, parameters)
    descriptor = _shader_descriptor(name, path, op, inputs, parameters, body, contract)
    registry.register(Plugin(
        name,
        PluginFamily.POSTPROCESSING,
        _COLOR,
        tuple(kind for _, kind in inputs),
        params=tuple(key for key, _ in parameters),
        shader=descriptor,
        input_uniforms=tuple(port for port, _ in inputs),
        make_uniforms=partial(_uniforms, names=tuple(key for key, _ in parameters)),
        mode="RGB" if op == "finish" else "LINEAR",
    ))


def _shader_contract(name, path, op, inputs, parameters):
    return ShaderDifferentiation(
        name,
        path,
        tuple(ShaderArgument(port, () if kind == _SDF else (4,), "r32f" if kind == _SDF else "rgba8") for port, kind in inputs),
        tuple(ShaderArgument(key, (4,) if kind == "vec4" else (), kind) for key, kind in parameters),
        (ShaderArgument("destTex", (4,), "rgba8"),),
        notes="Differentiate unquantized linear-light math with fixed branches and a fixed dither pattern.",
        reference=partial(_reference, op=op),
    )


def _shader_descriptor(name, path, op, inputs, parameters, body, contract):
    return ShaderFile(
        name,
        path,
        derivative_contract=contract,
        interface=ShaderInterface(
            ImageBinding("destTex", 0, write=True),
            tuple(ImageBinding(port, i, read=True) for i, (port, _) in enumerate(inputs, 1)),
            tuple(Uniform(key, kind) for key, kind in parameters),
        ),
        wgsl=PointwiseKernel(contract, body, mixed=op, helpers=GRAIN_WGSL if op == "grain" else ""),
    )
