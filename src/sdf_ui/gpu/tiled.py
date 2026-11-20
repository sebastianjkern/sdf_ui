"""Out-of-core GPU evaluation of still images with finite sampling support."""

import math
from contextlib import nullcontext
from pathlib import Path

from PIL import Image

from ..core.texture import Renderer, _scale_spatial_params
from .context import _integer_pair

# These host nodes are compositions of region-safe shaders, or explicitly
# evaluate their geometry in global coordinates within the requested crop.
COMPOSITES = {
    "blur",
    "blur_9",
    "blur_13",
    "linear_gradient",
    "radial_gradient",
    "polygon",
}
NEIGHBOR_SHADERS = {
    "blur_hor_9",
    "blur_ver_9",
    "blur_hor_13",
    "blur_ver_13",
    "warp_sdf",
    "warp_color",
}
_WARPS = {"warp_sdf", "warp_color"}
# Fill uses a one-pixel finite difference for antialiasing. The gradients
# lower to a fill internally.
_STENCIL_RADIUS = {"fill": 1, "radial_gradient": 1, "linear_gradient": 1}


def _field_bound(node, resolve, memo):
    """Upper bound on the magnitude of the bilinearly interpolated field."""
    identity = id(node)
    if identity not in memo:
        params = resolve(node)
        if node.op == "constant_field":
            bound = math.hypot(*params["vector"])
        elif node.op in {"radial_field", "vortex_field"}:
            # r * exp(-r²/2) is at most exp(-1/2); 1 is conservative.
            bound = abs(params["strength"])
        elif node.op == "turbulence_field":
            bound = math.sqrt(2) * abs(params["amplitude"])
            if params["gain"] < 0:
                # Alternating weights may cancel in the shader's normalizer.
                gain = abs(params["gain"])
                octaves = params["octaves"]
                if octaves not in range(1, 9):
                    raise ValueError(
                        "turbulence octaves must be an integer from 1 to 8"
                    )
                bound *= sum(gain**i for i in range(int(octaves))) / 0.0001
        elif node.op == "field_add":
            bound = sum(_field_bound(child, resolve, memo) for child in node.inputs)
        elif node.op == "field_scale":
            bound = abs(params["amount"]) * _field_bound(node.inputs[0], resolve, memo)
        else:
            raise NotImplementedError(
                f"Cannot bound displacement for field '{node.op}'"
            )
        if not math.isfinite(bound):
            raise ValueError("Warp displacement bound must be finite")
        memo[identity] = bound
    return memo[identity]


def required_halo(renderer, scene):
    """Validate the graph and add sampling reach along its longest branch."""
    from ..core.spatial import is_spatial_node
    from .nodes.registry import registry

    def resolve(node):
        params = {name: renderer._resolve(value) for name, value in node.params}
        return _scale_spatial_params(node.op, params, renderer.ctx.device_scale)

    halos, bounds = {}, {}

    def visit(node):
        if id(node) in halos:
            return halos[id(node)]
        if getattr(node, "tex", None) is not None:
            raise NotImplementedError(
                "Tiled export requires unmaterialized scene inputs"
            )
        if is_spatial_node(node):
            from .spatial import tile_dependencies

            halo = max(
                (
                    visit(child) + reach
                    for child, reach in tile_dependencies(renderer, node)
                ),
                default=0,
            )
            halos[id(node)] = halo
            return halo
        children = renderer._region_children(node, registry, tiled=True)
        halo = max((visit(child) for child in children), default=0)
        halo += _STENCIL_RADIUS.get(node.op, 0)
        if node.op in {"blur", "blur_9", "blur_13"}:
            params = resolve(node)
            radius = 6 if node.op == "blur_13" or params.get("base") == 13 else 4
            halo += radius * (max(0, params["n"]) + 1)
        elif node.op in _WARPS:
            strength = abs(resolve(node)["strength"])
            # Interpolation can read the pixel just beyond the displacement.
            halo += (
                math.ceil(strength * _field_bound(node.inputs[1], resolve, bounds)) + 2
            )
        halos[id(node)] = halo
        return halo

    return visit(scene)


def save_tiled(canvas, scene, path, *, tile_size, params, progress):
    """Read back one expanded tile at a time; assemble on the CPU."""
    from ..core.texture import _texture_bytes

    if getattr(scene, "kind", None) != "color":
        raise TypeError("save_tiled requires a color scene")
    tile_w, tile_h = _integer_pair(tile_size, "tile_size", positive=True)
    width, height = canvas.size
    path = Path(path)
    with Renderer(canvas, params=params, progress=progress) as probe:
        halo = required_halo(probe, scene)
    limit = canvas.device.limits["max-texture-dimension-2d"]
    if min(width, tile_w + 2 * halo) > limit or min(height, tile_h + 2 * halo) > limit:
        raise ValueError("Tile size plus sampling halo exceeds the GPU texture limit")

    previous = canvas._tiled_export
    canvas._tiled_export = True
    try:
        with Image.new("RGBA", canvas.size) as assembled:
            for y in range(0, height, tile_h):
                for x in range(0, width, tile_w):
                    right, top = min(x + tile_w, width), min(y + tile_h, height)
                    left, bottom = max(0, x - halo), max(0, y - halo)
                    padded_right = min(width, right + halo)
                    padded_top = min(height, top + halo)
                    rect = (left, bottom, padded_right - left, padded_top - bottom)
                    with (
                        Renderer(
                            canvas,
                            params=params,
                            progress=progress,
                            evaluation_rect=rect,
                        ) as renderer,
                        renderer.render(scene) as output,
                    ):
                        display = (
                            nullcontext(output)
                            if output.mode == "RGB"
                            else canvas.render(output.to_rgb())
                        )
                        with display as rgb:
                            with Image.frombytes(
                                "RGBA", rgb.size, _texture_bytes(rgb.tex)
                            ) as tile:
                                with tile.crop(
                                    (x - left, y - bottom, right - left, top - bottom)
                                ) as core:
                                    with core.transpose(
                                        Image.Transpose.FLIP_TOP_BOTTOM
                                    ) as upright:
                                        assembled.paste(upright, (x, height - top))
            assembled.save(path)
    finally:
        canvas._tiled_export = previous
    return path
