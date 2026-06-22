import argparse
import logging
from contextlib import contextmanager

from examples.animation_example import animation_example
from examples.arc_example import arc_example
from examples.capsule_example import capsule_example
from examples.complex_shapes_example import complex_shapes_example
from examples.convex_polygon_example import convex_polygon_example
from examples.dawn_example import dawn_example
from examples.diamond_example import diamond_example
from examples.ellipse_example import ellipse_example
from examples.font_example import font_example
from examples.github_banner_example import github_banner_example
from examples.half_plane_example import half_plane_example
from examples.impossible_city import impossible_city_example
from examples.isolines_example import isolines_example
from examples.light_example import light_example
from examples.masked_union_example import masked_union_example
from examples.neuromorphism_light_example import neuromorphism_light_example
from examples.ngon_example import ngon_example
from examples.parallelogram_example import parallelogram_example
from examples.partial_derivative import partial_derivative_example
from examples.polygon_example import polygon_example
from examples.render_api_example import render_api_example
from examples.ring_example import ring_example
from examples.rotation_example import rotation_example
from examples.sector_example import sector_example
from examples.striped_circles_example import striped_circles_example
from examples.text_rendering_example import text_rendering_example
from examples.transparency_example import transparency_example
from examples.voronoi_example import voronoi_example
from sdf_ui import logger
from sdf_ui.core.texture import TextureNode

logger().setLevel(logging.CRITICAL)


examples = {
    "masked_union": masked_union_example,
    "rotation": rotation_example,
    "voronoi": voronoi_example,
    "text": text_rendering_example,
    "transparency": transparency_example,
    "font": font_example,
    "complex": complex_shapes_example,
    "anim": animation_example,
    "arc": arc_example,
    "pd": partial_derivative_example,
    "ic": impossible_city_example,
    "render_api": render_api_example,
    "striped_circles": striped_circles_example,
    "github": github_banner_example,
    "art": dawn_example,
    "isolines": isolines_example,
    "light": light_example,
    "neuromorphism": neuromorphism_light_example,
    "half_plane": half_plane_example,
    "capsule": capsule_example,
    "convex_polygon": convex_polygon_example,
    "diamond": diamond_example,
    "ellipse": ellipse_example,
    "ngon": ngon_example,
    "parallelogram": parallelogram_example,
    "polygon": polygon_example,
    "ring": ring_example,
    "sector": sector_example,
}

_manual_stats_examples = {"text", "github", "art"}


def _format_stats(stats):
    return (
        f"Render stats: {stats.shader_dispatches} dispatches, "
        f"{stats.texture_allocations} textures, "
        f"{stats.cache_hits} cache hits, "
        f"{stats.cache_misses} cache misses, "
        f"{stats.elapsed_seconds:.3f}s"
    )


def _print_stats(ctx):
    stats = getattr(ctx, "last_render_stats", None)
    if stats is not None:
        print(_format_stats(stats))


@contextmanager
def _stats_hooks(enabled):
    if not enabled:
        yield
        return

    original_show = TextureNode.show
    original_save = TextureNode.save

    def show_with_stats(self, ctx=None, params=None, cache=None, conversion=True, size=None):
        result = original_show(
            self, ctx=ctx, params=params, cache=cache, conversion=conversion, size=size
        )
        _print_stats(ctx or self.context)
        return result

    def save_with_stats(self, name="./image.png", ctx=None, params=None, cache=None, conversion=True, size=None):
        result = original_save(
            self,
            name=name,
            ctx=ctx,
            params=params,
            cache=cache,
            conversion=conversion,
            size=size,
        )
        _print_stats(ctx or self.context)
        return result

    TextureNode.show = show_with_stats
    TextureNode.save = save_with_stats
    try:
        yield
    finally:
        TextureNode.show = original_show
        TextureNode.save = original_save


def _run_example(name, example, *, stats=True):
    print(f"== {name} ==")
    with _stats_hooks(stats and name not in _manual_stats_examples):
        example()
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="SDF_UI Examples",
        description="Show the results of some sdf_ui examples",
    )

    parser.add_argument(
        "example",
        nargs="?",
        default="all",
        help=f"Options: {list(examples.keys()) + ['all']}",
    )
    parser.add_argument(
        "--no-stats",
        dest="stats",
        action="store_false",
        help="Disable automatic render-stat printing around show/save calls",
    )
    parser.set_defaults(stats=True)

    args = parser.parse_args(argv)

    if args.example == "all":
        for name, example in examples.items():
            _run_example(name, example, stats=args.stats)
        return

    if args.example in examples:
        _run_example(args.example, examples[args.example], stats=args.stats)
        return

    parser.error(f"Unknown example '{args.example}'")


if __name__ == "__main__":
    main()
