from __future__ import annotations

import math
from pathlib import Path

import pytest

import sdf_ui
from sdf_ui import Canvas, DispatchConfig, color, core, sdf
from sdf_ui.core.api import ColorNamespace, SDFNamespace
from sdf_ui.core.context import Context
from sdf_ui.core.expressions import (
    cos,
    evaluate_expr,
    param,
    percent,
    percent_of_min,
    percent_x,
    percent_y,
    sin,
)
from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.registry import PluginRegistry, registry
from sdf_ui.core.texture import (
    MultiOutputResult,
    PostNamespace,
    freeze,
    texture_params,
)


class FakeContext:
    def __init__(self, size=(200, 100)):
        self.size = size

    def percent(self, value):
        return value / 100 * self.size[0]

    def percent_x(self, value):
        return value / 100 * self.size[0]

    def percent_y(self, value):
        return value / 100 * self.size[1]

    def percent_of_min(self, value):
        return value / 100 * min(self.size)


def test_root_exports_the_new_public_api():
    assert Canvas is Context
    assert DispatchConfig is core.DispatchConfig
    assert isinstance(sdf, SDFNamespace)
    assert isinstance(color, ColorNamespace)
    assert hasattr(sdf_ui, "core")
    assert hasattr(sdf_ui, "logger")
    assert not hasattr(sdf_ui, "disc")
    assert "Canvas" in sdf_ui.__all__
    assert "DispatchConfig" in sdf_ui.__all__
    assert "sdf" in sdf_ui.__all__
    assert "color" in sdf_ui.__all__


def test_dispatch_config_calculates_group_counts_and_validates_values():
    default = DispatchConfig()
    custom = DispatchConfig(shader_local_size=(8, 4))
    explicit = DispatchConfig(group_count=(3, 2))

    assert default.groups_for_size((33, 17)) == (3, 2, 1)
    assert custom.shader_local_size == (8, 4, 1)
    assert custom.groups_for_size((33, 17)) == (5, 5, 1)
    assert explicit.groups_for_size((100, 100)) == (3, 2, 1)
    assert DispatchConfig.from_value({"shader_local_size": (4, 4)}) == DispatchConfig(
        shader_local_size=(4, 4, 1)
    )

    with pytest.raises(ValueError, match="greater than 0"):
        DispatchConfig(shader_local_size=(0, 16))
    with pytest.raises(TypeError, match="DispatchConfig"):
        DispatchConfig.from_value((16, 16))


def test_namespace_dir_exposes_the_public_factories():
    sdf_names = set(dir(sdf))
    color_names = set(dir(color))

    assert {"circle", "disc", "rect", "rounded_rect", "line", "bezier", "grid", "triangle"}.issubset(
        sdf_names
    )
    assert {
        "clear",
        "clear_color",
        "linear_gradient",
        "radial_gradient",
        "noise",
        "perlin_noise",
        "film_grain",
    }.issubset(color_names)


@pytest.mark.parametrize(
    ("namespace_name", "factory_name", "args", "expected_kind", "expected_op"),
    [
        ("sdf", "circle", ((16, 16), 5), "sdf", "circle"),
        ("sdf", "disc", ((16, 16), 5), "sdf", "disc"),
        ("sdf", "rect", ((16, 16), (10, 10), (2, 2, 2, 2)), "sdf", "rect"),
        (
            "sdf",
            "rounded_rect",
            ((16, 16), (10, 10), (2, 2, 2, 2)),
            "sdf",
            "rounded_rect",
        ),
        ("sdf", "line", ((0, 0), (15, 15)), "sdf", "line"),
        ("sdf", "bezier", ((0, 0), (8, 12), (15, 0)), "sdf", "bezier"),
        ("sdf", "triangle", ((0, 0), (15, 0), (8, 15)), "sdf", "triangle"),
        ("sdf", "grid", ((8, 8), (4, 4)), "sdf", "grid"),
        ("color", "clear", ("#ffffff",), "color", "clear"),
        ("color", "clear_color", ("#ffffff",), "color", "clear_color"),
        (
            "color",
            "linear_gradient",
            ((0, 8), (15, 8), (255, 0, 0, 255), (0, 0, 255, 255)),
            "color",
            "linear_gradient",
        ),
        (
            "color",
            "radial_gradient",
            ((8, 8), (255, 0, 0, 255), (0, 0, 255, 255)),
            "color",
            "radial_gradient",
        ),
        ("color", "noise", (), "color", "noise"),
        ("color", "perlin_noise", (), "color", "perlin_noise"),
        ("color", "film_grain", (), "color", "film_grain"),
    ],
)
def test_public_factories_build_expected_texture_kinds(
    namespace_name, factory_name, args, expected_kind, expected_op
):
    namespace = sdf if namespace_name == "sdf" else color
    node = getattr(namespace, factory_name)(*args)

    assert node.kind == expected_kind
    assert node.op == expected_op


def test_expression_constructors_and_evaluation_cover_percent_syntax():
    ctx = FakeContext()

    assert percent(25).op == "percent"
    assert percent_x(25).op == "percent_x"
    assert percent_y(25).op == "percent_y"
    assert percent_of_min(25).op == "percent_of_min"
    assert sin(1).op == "sin"
    assert cos(1).op == "cos"

    assert evaluate_expr("25%", ctx, {}) == 50
    assert evaluate_expr("25%x", ctx, {}) == 50
    assert evaluate_expr("25%y", ctx, {}) == 25
    assert evaluate_expr("25%min", ctx, {}) == 25


def test_expression_arithmetic_and_param_lookup():
    ctx = FakeContext()
    expr = percent_x(10) + sin(math.pi / 2) * 2 - 1

    assert evaluate_expr(expr, ctx, {}) == pytest.approx(21.0)
    assert evaluate_expr(param("radius", 12), ctx, {}) == 12
    assert evaluate_expr(param("radius"), ctx, {"radius": 33}) == 33

    with pytest.raises(KeyError, match="Missing render parameter 'radius'"):
        evaluate_expr(param("radius"), ctx, {})


def test_texture_params_and_freeze_preserve_texture_nodes():
    node = sdf.circle((1, 2), 3)

    frozen = freeze({"b": [1, {"x": 2}], "a": node})
    params = texture_params(b=[1, {"x": 2}], a=node)

    assert frozen == (("a", node), ("b", (1, (("x", 2),))))
    assert params == (("a", node), ("b", (1, (("x", 2),))))


def test_plugin_bind_and_registry_validation_cover_common_failure_modes():
    plugin = Plugin(
        "demo",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        (TextureKind.SDF,),
        params=("radius", "falloff"),
        defaults={"falloff": 2},
    )
    shape = sdf.circle((1, 2), 3)

    inputs, params = plugin.bind((shape, 7), {})
    assert inputs == (shape,)
    assert params == {"radius": 7, "falloff": 2}

    with pytest.raises(TypeError, match="expects at least 1 input argument"):
        plugin.bind((), {})
    with pytest.raises(TypeError, match="too many positional arguments"):
        plugin.bind((shape, 7, 8, 9), {})
    with pytest.raises(TypeError, match="unexpected parameter 'unknown'"):
        plugin.bind((shape,), {"unknown": 1})
    with pytest.raises(TypeError, match="missing required parameter"):
        Plugin(
            "missing",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("radius",),
        ).bind((shape,), {})

    fresh_registry = PluginRegistry()
    fresh_registry.register(plugin)

    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register(plugin)
    with pytest.raises(KeyError, match="Unknown texture plugin 'missing'"):
        fresh_registry.get("missing")


def test_registry_metadata_points_to_real_shader_files():
    registry.ensure_loaded()
    package_root = Path(__file__).resolve().parents[1] / "src" / "sdf_ui"

    public_names = set(registry.public_names())
    assert {"circle", "rect", "line", "bezier", "triangle", "clear", "linear_gradient"}.issubset(
        public_names
    )

    shader_names = {descriptor.name for descriptor in registry.shader_files()}
    assert {"circle", "fill", "layer_mask", "blur_ver_9", "blur_hor_9"}.issubset(
        shader_names
    )

    for descriptor in registry.shader_files():
        assert (package_root / descriptor.path).exists(), descriptor


def test_registry_method_names_match_texture_kinds():
    registry.ensure_loaded()

    sdf_methods = set(registry.method_names_for("sdf"))
    color_methods = set(registry.method_names_for("color"))

    assert {
        "fill",
        "fill_from_texture",
        "generate_mask",
        "light",
        "outline",
        "shadow",
    }.issubset(sdf_methods)
    assert {"alpha_overlay", "mask", "multiply", "transparency", "blur", "invert"}.issubset(
        color_methods
    )
    assert "fill" not in color_methods
    assert "alpha_overlay" not in sdf_methods


def test_texture_node_dir_and_clone_helpers_are_kind_aware():
    sdf_node = sdf.circle((1, 2), 3)
    color_node = color.clear("#fff")

    assert {"fill", "outline", "shadow", "light", "generate_mask"}.issubset(
        dir(sdf_node)
    )
    assert {
        "alpha_overlay",
        "mask",
        "multiply",
        "transparency",
        "blur",
        "blur_9",
        "blur_13",
        "invert",
        "dithering",
        "dither_1bit",
    }.issubset(dir(color_node))
    assert "alpha_overlay" not in dir(sdf_node)
    assert "fill" not in dir(color_node)

    named = sdf_node.named("circle").cache().uncached()
    assert named.label == "circle"
    assert not named.should_cache
    assert type(named) is type(sdf_node)


def test_post_namespace_dir_exposes_the_postprocessing_methods():
    post = color.clear("#fff").post

    assert isinstance(post, PostNamespace)
    assert {
        "blur",
        "blur_9",
        "blur_13",
        "dithering",
        "dither_1bit",
        "invert",
        "to_lab",
        "to_rgb",
    }.issubset(dir(post))


def test_multi_output_results_unpack_to_typed_outputs():
    result = sdf.circle((16, 16), 10).masked_union(sdf.circle((20, 16), 10))
    assert isinstance(result, MultiOutputResult)

    sdf_result, color_result = result

    assert sdf_result.kind == "sdf"
    assert color_result.kind == "color"
    assert sdf_result.op == "output"
    assert color_result.op == "output"
    assert color_result.mode == "RGB"

    with pytest.raises(TypeError, match="not iterable"):
        list(MultiOutputResult(op="not_masked_union"))
