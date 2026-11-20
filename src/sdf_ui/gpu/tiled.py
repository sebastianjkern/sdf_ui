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


def _negative_gain_turbulence_bound(params, bound):
    gain = abs(params["gain"])
    octaves = params["octaves"]
    if octaves not in range(1, 9):
        raise ValueError("turbulence octaves must be an integer from 1 to 8")
    return bound * sum(gain**i for i in range(int(octaves))) / 0.0001


def _base_field_bound(node, params, resolve, memo):
    if node.op == "constant_field":
        return math.hypot(*params["vector"])
    if node.op in {"radial_field", "vortex_field"}:
        return abs(params["strength"])
    if node.op == "turbulence_field":
        bound = math.sqrt(2) * abs(params["amplitude"])
        return _negative_gain_turbulence_bound(params, bound) if params["gain"] < 0 else bound
    if node.op == "field_add":
        return sum(_field_bound(child, resolve, memo) for child in node.inputs)
    if node.op == "field_scale":
        return abs(params["amount"]) * _field_bound(node.inputs[0], resolve, memo)
    raise NotImplementedError(f"Cannot bound displacement for field '{node.op}'")


def _field_bound(node, resolve, memo):
    """Upper bound on the magnitude of the bilinearly interpolated field."""
    identity = id(node)
    if identity not in memo:
        bound = _base_field_bound(node, resolve(node), resolve, memo)
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


def _validate_tiled_request(canvas, scene, tile_size, params, progress):
    if getattr(scene, "kind", None) != "color":
        raise TypeError("save_tiled requires a color scene")
    tile_width, tile_height = _integer_pair(tile_size, "tile_size", positive=True)
    with Renderer(canvas, params=params, progress=progress) as probe:
        halo = required_halo(probe, scene)
    limit = canvas.device.limits["max-texture-dimension-2d"]
    if min(canvas.size[0], tile_width + 2 * halo) > limit or min(
        canvas.size[1], tile_height + 2 * halo
    ) > limit:
        raise ValueError("Tile size plus sampling halo exceeds the GPU texture limit")
    return tile_width, tile_height, halo


def _expanded_tile(x, y, right, top, width, height, halo):
    left_edge, bottom_edge = max(0, x - halo), max(0, y - halo)
    right_edge, top_edge = min(width, right + halo), min(height, top + halo)
    rect = (
        left_edge, bottom_edge,
        right_edge - left_edge, top_edge - bottom_edge,
    )
    crop = (x - left_edge, y - bottom_edge, right - left_edge, top - bottom_edge)
    return rect, crop


def _paste_tile(assembled, canvas, scene, params, progress, rect, crop, position):
    from ..core.texture import _texture_bytes

    with (
        Renderer(canvas, params=params, progress=progress, evaluation_rect=rect) as renderer,
        renderer.render(scene) as output,
    ):
        display = nullcontext(output) if output.mode == "RGB" else canvas.render(output.to_rgb())
        with display as rgb:
            with Image.frombytes("RGBA", rgb.size, _texture_bytes(rgb.tex)) as tile:
                with tile.crop(crop) as core:
                    with core.transpose(Image.Transpose.FLIP_TOP_BOTTOM) as upright:
                        assembled.paste(upright, position)


def _render_tile_grid(assembled, canvas, scene, params, progress, tile_width, tile_height, halo):
    width, height = canvas.size
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            right, top = min(x + tile_width, width), min(y + tile_height, height)
            rect, crop = _expanded_tile(x, y, right, top, width, height, halo)
            _paste_tile(
                assembled, canvas, scene, params, progress, rect, crop,
                (x, height - top),
            )


def save_tiled(canvas, scene, path, *, tile_size, params, progress):
    """Read back one expanded tile at a time; assemble on the CPU."""
    tile_width, tile_height, halo = _validate_tiled_request(
        canvas, scene, tile_size, params, progress
    )
    path = Path(path)
    previous = canvas._tiled_export
    canvas._tiled_export = True
    try:
        with Image.new("RGBA", canvas.size) as assembled:
            _render_tile_grid(
                assembled, canvas, scene, params, progress,
                tile_width, tile_height, halo,
            )
            assembled.save(path)
    finally:
        canvas._tiled_export = previous
    return path
