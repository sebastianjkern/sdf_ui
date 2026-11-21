"""Fixed-coordinate Perlin shader layout and reference implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import Tensor

    x = context.x.data.astype(np.float32) / np.float32(256.0) * np.float32(1.5)
    y = context.y.data.astype(np.float32) / np.float32(256.0) * np.float32(1.5)
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy

    def mod289(v):
        return v - np.floor(v * np.float32(1.0 / 289.0)) * np.float32(289.0)

    def permute(v):
        return mod289((v * np.float32(34.0) + np.float32(10.0)) * v)

    def corner(dx, dy):
        index = permute(permute(mod289(ix + dx)) + mod289(iy + dy))
        gx = index * np.float32(1.0 / 41.0)
        gx = (gx - np.floor(gx)) * np.float32(2.0) - np.float32(1.0)
        gy = np.abs(gx) - np.float32(0.5)
        gx = gx - np.floor(gx + np.float32(0.5))
        norm = np.float32(1.79284291400159) - np.float32(0.85373472095314) * (
            gx * gx + gy * gy
        )
        return gx * norm * (fx - dx) + gy * norm * (fy - dy)

    def fade(t):
        return t * t * t * (t * (t * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0))

    u, v = fade(fx), fade(fy)
    lower = corner(0, 0) * (1 - u) + corner(1, 0) * u
    upper = corner(0, 1) * (1 - u) + corner(1, 1) * u
    value = np.float32(2.3) * (lower * (1 - v) + upper * v)
    return Tensor(np.stack((value, value, value, np.ones_like(value)), axis=-1))


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True))
CONTRACT = ShaderDifferentiation(
    "perlin_noise", "gpu/nodes/shading/perlin_noise/perlin_noise.wgsl", (), (),
    (ShaderArgument("destTex", (4,), "rgba8"),), behavior="constant",
    notes="Fixed pixel coordinates/noise constants; no trainable shader inputs.", reference=reference,
)
