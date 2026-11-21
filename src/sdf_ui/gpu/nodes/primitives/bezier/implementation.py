"""Quadratic Bézier shader's layout and active-root differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def _curve_parameters(params):
    import numpy as np

    from sdf_ui.learning.render import _tensor

    a, b, c = (_tensor(params[name]) for name in ("a", "b", "c"))
    b = b + np.where((2.0 * b - a - c).data == 0.0, 1e-4, 0.0)
    av, bv, cv = a.data, b.data, c.data
    chord, control = cv - av, bv - av
    if chord[0] * control[1] - chord[1] * control[0] == 0:
        raise ValueError("bezier requires non-collinear adjusted control points")
    linear, quadratic = 2.0 * (bv - av), av - 2.0 * bv + cv
    return a, b, c, linear, quadratic


def _root_candidates(context, a, linear, quadratic):
    import numpy as np

    av = a.data
    delta = np.stack((context.x.data - av[0], context.y.data - av[1]), axis=-1)
    leading = 2.0 * np.dot(quadratic, quadratic)
    second = 3.0 * np.dot(linear, quadratic)
    third = np.dot(linear, linear) - 2.0 * (delta @ quadratic)
    fourth = -(delta @ linear)
    candidates = np.empty(context.x.shape + (3,))
    for index in np.ndindex(context.x.shape):
        roots = np.roots((leading, second, third[index], fourth[index]))
        real = np.sort(roots.real[np.abs(roots.imag) <= 1 / 1_000_000_000])
        candidates[index] = np.clip(np.resize(real, 3), 0.0, 1.0)
    candidates.sort(axis=-1)
    return candidates


def _candidate_distances(context, candidates, a, b, c):

    from sdf_ui.learning.autodiff import stack
    from sdf_ui.learning.render import _length

    distances = []
    for index in range(3):
        t = candidates[..., index]
        point = a * (1.0 - t[..., None]) ** 2 + 2.0 * b * (t * (1.0 - t))[..., None]
        point = point + c * (t * t)[..., None]
        distances.append(_length(context.x - point[..., 0], context.y - point[..., 1]))
    return stack(distances, axis=-1)


def _active_root_mean(values, candidates):
    import numpy as np

    from sdf_ui.learning.autodiff import where

    unique = np.concatenate(
        (np.ones(candidates.shape[:-1] + (1,), dtype=bool),
         np.diff(candidates) != 0.0), axis=-1,
    )
    data = values.data
    active = unique & (data == data.min(axis=-1, keepdims=True))
    return where(active, values, 0.0).sum(axis=-1) / active.sum(axis=-1)


def reference(context, inputs, params):
    a, b, c, linear, quadratic = _curve_parameters(params)
    roots = _root_candidates(context, a, linear, quadratic)
    distances = _candidate_distances(context, roots, a, b, c)
    return _active_root_mean(distances, roots)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=tuple(Uniform(name, "vec2") for name in ("a", "b", "c")),
)
CONTRACT = ShaderDifferentiation(
    "bezier", "gpu/nodes/primitives/bezier/bezier.wgsl", (),
    tuple(ShaderArgument(name, (2,), "vec2") for name in ("a", "b", "c")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Closest-point envelope derivative with active root/endpoint fixed; final abs removes sign.",
    reference=reference,
)
