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


def _rgba(value):
    from sdf_ui.learning.render import _tensor

    value = _tensor(value)
    if value.shape != (4,):
        raise ValueError("Linear color requires an RGBA vector of four components")
    return value


def _uniforms(values, names):
    import numpy as np

    result = {}
    scalars = {}
    for name in names:
        value = values[name]
        if name in {"linear_rgba", "linear_background"}:
            tensor = _rgba(value)
            data = tensor.numpy()
            if (
                not np.isfinite(data).all()
                or np.any(data[:3] < 0)
                or not 0 <= data[3] <= 1
            ):
                raise ValueError(
                    "Linear RGBA requires finite nonnegative RGB and alpha in [0, 1]"
                )
            result[name] = tensor
        else:
            scalar = float(value.item() if hasattr(value, "item") else value)
            if name == "grain_amount" and not 0 <= scalar <= 1:
                raise ValueError("Film grain amount must be in [0, 1]")
            if name == "grain_size" and not 1 <= scalar < float("inf"):
                raise ValueError("Film grain size must be finite and at least 1")
            if not np.isfinite(scalar) or (
                name in {"threshold", "amount", "tone", "dither"} and scalar < 0
            ):
                raise ValueError(
                    f"{name} must be finite and, for finishing controls, nonnegative"
                )
            result[name] = value
            scalars[name] = scalar
    if "inner" in scalars and scalars["outer"] <= scalars["inner"]:
        raise ValueError("Linear fill outer distance must exceed inner distance")
    return result


def _reference(context, inputs, p, *, op):
    import numpy as np

    from sdf_ui.learning.autodiff import exp, where
    from sdf_ui.learning.render import _rgba as pack
    from sdf_ui.learning.render import _tensor

    def clip01(value):
        return where(value.data <= 0, 0, where(value.data >= 1, 1, value))

    if op == "clear":
        return _rgba(p["linear_rgba"]) + (context.x * 0)[..., None]
    if op == "ink":
        distance = _tensor(inputs["sdf"])
        opacity = clip01(0.5 - distance)
        rgba = _rgba(p["linear_rgba"])
        return pack(rgba[:3] + (distance * 0)[..., None], rgba[3] * opacity)
    if op == "fill":
        distance = _tensor(inputs["sdf"])
        inner, outer = _tensor(p["inner"]), _tensor(p["outer"])
        t = clip01((distance - inner) / (outer - inner))
        t = (t * t * (3 - 2 * t))[..., None]
        fg, bg = _rgba(p["linear_rgba"]), _rgba(p["linear_background"])
        return fg + (bg - fg) * t

    source = _tensor(inputs["source"])
    rgb, a = source[..., :3], source[..., 3]
    if op == "decode":
        rgb = where(
            rgb.data <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).maximum(0) ** 2.4
        )
    elif op == "over":
        top = _tensor(inputs["top"])
        top_a = top[..., 3]
        remaining = a * (1 - top_a)
        alpha = top_a + remaining
        denominator = where(alpha.data > 0, alpha, 1)
        rgb = (
            top[..., :3] * top_a[..., None] + rgb * remaining[..., None]
        ) / denominator[..., None]
        rgb = where((alpha.data > 0)[..., None], rgb, 0)
        a = alpha
    elif op == "expose":
        rgb = rgb * exp(_tensor(p["stops"]) * np.log(2))
    elif op == "grain":
        amount = _tensor(p["grain_amount"])
        size = _tensor(p["grain_size"])
        amount = where(amount.data < 0, 0, where(amount.data > 1, 1, amount))
        size = where(size.data < 1, 1, size)
        luma = (rgb * [0.2126, 0.7152, 0.0722]).sum(axis=-1)
        factor = 1 + amount * reference_noise(context, size) / (1 + 3 * luma)
        rgb = rgb * factor[..., None]
    elif op == "extract":
        luma = (rgb * [0.2126, 0.7152, 0.0722]).sum(axis=-1)
        excess = luma - _tensor(p["threshold"])
        amount = where(excess.data > 0, excess, 0) / where(luma.data > 1e-6, luma, 1e-6)
        rgb = rgb * (a * amount)[..., None]
        a = a * 0
    elif op == "glow":
        light = _tensor(inputs["light"])[..., :3] * _tensor(p["amount"])
        # Give light in transparent areas coverage, while retaining HDR
        # energy in straight RGB. An opaque source stays opaque.
        glow_alpha = clip01((light * [0.2126, 0.7152, 0.0722]).sum(axis=-1))
        alpha = a + (1 - a) * glow_alpha
        denominator = where(alpha.data > 0, alpha, 1)
        rgb = where(
            (alpha.data > 0)[..., None],
            (rgb * a[..., None] + light) / denominator[..., None],
            0,
        )
        a = alpha
    elif op == "finish":
        rgb = where(rgb.data > 0, rgb, 0)
        luma = (rgb * [0.2126, 0.7152, 0.0722]).sum(axis=-1)
        rgb = rgb / (1 + luma[..., None] * _tensor(p["tone"]))
        # Avoid an infinite derivative in the unused power branch at zero.
        rgb = where(
            rgb.data <= 0.0031308,
            rgb * 12.92,
            1.055 * rgb.maximum(0.0031308) ** (1 / 2.4) - 0.055,
        )
        pattern = np.array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5])
        index = (context.y.data.astype(int) % 4) * 4 + context.x.data.astype(int) % 4
        noise = ((pattern[index] + 0.5) / 16 - 0.5) / 255
        rgb = clip01(rgb + _tensor(p["dither"]) * noise[..., None])
    return pack(rgb, a)


_OPS = {
    "clear": (
        (),
        (("linear_rgba", "vec4"),),
        "return array<D,4>(u[0],u[1],u[2],u[3]);",
    ),
    "ink": (
        (("sdf", _SDF),),
        (("linear_rgba", "vec4"),),
        """
        let coverage=clip(sub(k(0.5),ins[0][0]));
        return array<D,4>(u[0],u[1],u[2],mul(u[3],coverage));
    """,
    ),
    "fill": (
        (("sdf", _SDF),),
        (
            ("linear_rgba", "vec4"),
            ("linear_background", "vec4"),
            ("inner", "float"),
            ("outer", "float"),
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
        (("source", _COLOR),),
        (),
        """
        var out=ins[0];
        for (var i=0; i<3; i++) { out[i]=decode(out[i]); }
        return out;
    """,
    ),
    "over": (
        (("source", _COLOR), ("top", _COLOR)),
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
        (("source", _COLOR),),
        (("stops", "float"),),
        """
        var out=ins[0];
        let multiplier=scale2(u[0]);
        for (var i=0; i<3; i++) { out[i]=mul(out[i],multiplier); }
        return out;
    """,
    ),
    "grain": (
        (("source", _COLOR),),
        (("grain_amount", "float"), ("grain_size", "float")),
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
        (("source", _COLOR),),
        (("threshold", "float"),),
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
        (("source", _COLOR), ("light", _COLOR)),
        (("amount", "float"),),
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
        (("source", _COLOR),),
        (("tone", "float"), ("dither", "float")),
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
        name = f"hdr_{op}"
        path = f"gpu/shaders/{name}.wgsl"
        contract = ShaderDifferentiation(
            name,
            path,
            tuple(
                ShaderArgument(
                    port,
                    () if kind == _SDF else (4,),
                    "r32f" if kind == _SDF else "rgba8",
                )
                for port, kind in inputs
            ),
            tuple(
                ShaderArgument(key, (4,) if kind == "vec4" else (), kind)
                for key, kind in parameters
            ),
            (ShaderArgument("destTex", (4,), "rgba8"),),
            notes="Differentiate unquantized linear-light math with fixed branches and a fixed dither pattern.",
            reference=partial(_reference, op=op),
        )
        descriptor = ShaderFile(
            name,
            path,
            derivative_contract=contract,
            interface=ShaderInterface(
                ImageBinding("destTex", 0, write=True),
                tuple(
                    ImageBinding(port, i, read=True)
                    for i, (port, _) in enumerate(inputs, 1)
                ),
                tuple(Uniform(key, kind) for key, kind in parameters),
            ),
            wgsl=PointwiseKernel(
                contract, body, mixed=op, helpers=GRAIN_WGSL if op == "grain" else ""
            ),
        )
        registry.register(
            Plugin(
                name,
                PluginFamily.POSTPROCESSING,
                _COLOR,
                tuple(kind for _, kind in inputs),
                params=tuple(key for key, _ in parameters),
                shader=descriptor,
                input_uniforms=tuple(port for port, _ in inputs),
                make_uniforms=partial(
                    _uniforms, names=tuple(key for key, _ in parameters)
                ),
                mode="RGB" if op == "finish" else "LINEAR",
            )
        )
