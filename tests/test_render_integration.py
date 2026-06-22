from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sdf_ui import Canvas, color, sdf
from tests.helpers import render, rgba_array


@pytest.mark.parametrize(
    ("factory_name", "args"),
    [
        ("circle", ((16, 16), 5)),
        ("disc", ((16, 16), 5)),
        ("rect", ((16, 16), (10, 10), (2, 2, 2, 2))),
        ("rounded_rect", ((16, 16), (10, 10), (2, 2, 2, 2))),
        ("line", ((0, 0), (15, 15))),
        ("bezier", ((0, 0), (8, 12), (15, 0))),
        ("triangle", ((0, 0), (15, 0), (8, 15))),
        ("grid", ((8, 8), (4, 4))),
        ("clear", ("#ffffff",)),
        ("clear_color", ("#ffffff",)),
        ("linear_gradient", ((0, 8), (15, 8), (255, 0, 0, 255), (0, 0, 255, 255))),
        ("radial_gradient", ((8, 8), (255, 0, 0, 255), (0, 0, 255, 255))),
        ("noise", ()),
        ("perlin_noise", ()),
        ("film_grain", ()),
    ],
)
def test_public_factories_render_successfully(factory_name, args):
    namespace = sdf if hasattr(sdf, factory_name) else color
    texture = render(getattr(namespace, factory_name)(*args), size=(16, 16))

    assert texture.tex.size == (16, 16)
    assert texture.kind in {"sdf", "color"}


def test_clear_color_renders_solid_white():
    pixels = rgba_array(render(color.clear("#ffffff").to_rgb(), size=(8, 8)))

    assert pixels.shape == (8, 8, 4)
    assert np.all(pixels == 255)


def test_circle_fill_has_distinct_inside_and_outside_pixels():
    pixels = rgba_array(
        render(
            sdf.circle((8, 8), 5).fill((255, 255, 255, 255), (0, 0, 0, 255)).to_rgb(),
            size=(16, 16),
        )
    )

    assert pixels[8, 8, 0] > 200
    assert pixels[0, 0, 0] < 20


def test_percentage_expressions_are_resolved_during_render():
    pixels = rgba_array(
        render(
            sdf.circle(("50%x", "50%y"), "25%min")
            .fill((255, 255, 255, 255), (0, 0, 0, 255))
            .to_rgb(),
            size=(40, 20),
        )
    )

    assert pixels[10, 20, 0] > 200
    assert pixels[0, 0, 0] < 20


def test_fill_from_texture_maps_a_layer_through_an_sdf():
    pixels = rgba_array(
        render(
            sdf.circle((8, 8), 5).fill_from_texture(
                color.clear((255, 0, 0, 255)), background=(0, 0, 0, 255)
            ),
            size=(16, 16),
        )
    )

    assert pixels[8, 8, 0] > 200
    assert pixels[0, 0, 0] < 20


def test_alpha_overlay_and_mask_layers_change_pixels():
    overlay_pixels = rgba_array(
        render(
            color.clear((0, 0, 0, 0))
            .alpha_overlay(sdf.circle((8, 8), 5).fill((255, 0, 0, 255), (0, 0, 0, 0)))
            .to_rgb(),
            size=(16, 16),
        )
    )
    mask_pixels = rgba_array(
        render(
            color.clear((0, 0, 0, 0))
            .mask(
                color.clear((255, 0, 0, 255)),
                sdf.circle((8, 8), 5).generate_mask(),
            )
            .to_rgb(),
            size=(16, 16),
        )
    )

    assert overlay_pixels[8, 8, 3] == 255
    assert overlay_pixels[0, 0, 3] == 0
    assert mask_pixels[8, 8, 3] < 10
    assert mask_pixels[0, 0, 3] > 200


def test_color_compositing_and_transparency_render_predictably():
    multiply_pixels = rgba_array(
        render(
            color.clear((255, 0, 0, 255))
            .multiply(color.clear((0, 255, 0, 255)))
            .to_rgb(),
            size=(8, 8),
        )
    )
    transparency_pixels = rgba_array(
        render(color.clear("#ffffff").transparency(0.5).to_rgb(), size=(8, 8))
    )

    assert multiply_pixels[0, 0, 0] < 5
    assert multiply_pixels[0, 0, 1] < 5
    assert multiply_pixels[0, 0, 2] < 5
    assert transparency_pixels[0, 0, 3] == 128


def test_rgb_layer_operations_normalize_tuple_colors():
    red = color.clear((255, 0, 0, 255))
    blue = color.clear((0, 0, 255, 255))

    clear_pixels = rgba_array(render(red.to_rgb(), size=(8, 8)))
    overlay_pixels = rgba_array(
        render(color.clear((0, 0, 0, 0)).alpha_overlay(red), size=(8, 8))
    )
    mask_pixels = rgba_array(
        render(
            color.clear((0, 0, 0, 255)).mask(
                blue, sdf.circle((4, 4), 3).generate_mask()
            ),
            size=(8, 8),
        )
    )

    assert int(clear_pixels[0, 0, 0]) == pytest.approx(255, abs=2)
    assert int(clear_pixels[0, 0, 1]) == pytest.approx(0, abs=2)
    assert int(clear_pixels[0, 0, 2]) == pytest.approx(0, abs=2)
    assert int(clear_pixels[0, 0, 3]) == 255
    assert int(overlay_pixels[0, 0, 0]) == pytest.approx(255, abs=2)
    assert int(overlay_pixels[0, 0, 1]) == pytest.approx(0, abs=2)
    assert int(overlay_pixels[0, 0, 2]) == pytest.approx(0, abs=2)
    assert int(overlay_pixels[0, 0, 3]) == 255
    assert mask_pixels[4, 4, 2] < 20
    assert mask_pixels[0, 0, 2] > 200


@pytest.mark.parametrize(
    "factory_name", ["blur", "blur_9", "blur_13", "dithering", "dither_1bit", "invert"]
)
def test_color_postprocessing_methods_render_successfully(factory_name):
    texture = render(getattr(color.clear("#ffffff"), factory_name)(), size=(8, 8))

    assert texture.tex.size == (8, 8)


def test_blur_preserves_a_solid_white_image():
    pixels = rgba_array(render(color.clear("#ffffff").blur().to_rgb(), size=(8, 8)))

    assert pixels.min() >= 245


def test_invert_turns_white_into_black():
    pixels = rgba_array(render(color.clear("#ffffff").invert().to_rgb(), size=(8, 8)))

    assert np.all(pixels[..., :3] < 5)
    assert np.all(pixels[..., 3] == 255)


@pytest.mark.parametrize(
    ("factory_name", "point_a", "point_b"),
    [
        ("radial", (8, 8), (0, 0)),
        ("linear", (0, 8), (8, 8)),
    ],
)
def test_gradient_factories_produce_distinct_pixels(factory_name, point_a, point_b):
    if factory_name == "radial":
        scene = color.radial_gradient(
            (8, 8), (255, 0, 0, 255), (0, 0, 255, 255), inner=0, outer=8
        )
    else:
        scene = color.linear_gradient(
            (0, 8), (15, 8), (255, 0, 0, 255), (0, 0, 255, 255)
        )

    pixels = rgba_array(render(scene.to_rgb(), size=(16, 16)))

    assert not np.array_equal(pixels[point_a], pixels[point_b])


def test_outline_shadow_light_and_partial_derivative_render_distinct_features():
    outline_pixels = rgba_array(
        render(
            sdf.circle((8, 8), 5)
            .outline((255, 255, 255, 255), (0, 0, 0, 255), inflate=1)
            .to_rgb(),
            size=(16, 16),
        )
    )
    shadow_pixels = rgba_array(
        render(
            sdf.circle((8, 8), 5)
            .shadow(distance=3, inflate=0, transparency=0.5)
            .to_rgb(),
            size=(16, 16),
        )
    )
    light_pixels = rgba_array(
        render(
            sdf.circle((16, 16), 10)
            .light(
                (80, 180, 220, 255),
                (0, 0, 0, 255),
                light_dir=(-0.7, -0.7, 0.8),
                ambient=0.2,
                diffuse=0.45,
                specular=0.0,
                normal_strength=4.0,
                bevel=8.0,
                shade_background=True,
                background_bevel=8.0,
            ),
            size=(32, 32),
        )
    )
    derivative_pixels = rgba_array(
        render(sdf.circle((8, 8), 5).partial_derivative(), size=(16, 16))
    )

    assert outline_pixels[8, 2, 0] > 200
    assert outline_pixels[8, 8, 0] < 20
    assert shadow_pixels[8, 8, 3] > shadow_pixels[0, 0, 3]
    assert shadow_pixels[8, 2, 3] > shadow_pixels[0, 0, 3]
    assert light_pixels[5, 16, 2] > light_pixels[27, 16, 2]
    assert light_pixels[16, 5, 2] > light_pixels[16, 27, 2]
    assert light_pixels[8, 8, 2] > light_pixels[24, 24, 2]
    assert light_pixels[16, 16, 2] > light_pixels[0, 0, 2]
    light_alpha_pixels = rgba_array(
        render(
            sdf.circle((16, 16), 10).light(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
                ambient=1.0,
                diffuse=0.0,
                specular=0.0,
            ),
            size=(32, 32),
        )
    )
    antialias_values = light_alpha_pixels[..., 0]
    assert np.count_nonzero((antialias_values > 0) & (antialias_values < 255)) > 0
    soft_profile_pixels = rgba_array(
        render(
            sdf.circle((16, 16), 10).light(
                (80, 180, 255, 255),
                (0, 0, 0, 255),
                light_dir=(-0.7, -0.7, 0.8),
                ambient=0.2,
                diffuse=0.9,
                specular=0.0,
                normal_strength=7.0,
                height_profile="soft",
                height_gamma=0.6,
            ),
            size=(32, 32),
        )
    )
    assert not np.array_equal(light_pixels, soft_profile_pixels)
    background_lit_pixels = rgba_array(
        render(
            sdf.circle((16, 16), 8).light(
                (255, 255, 255, 255),
                (128, 128, 128, 255),
                light_dir=(-0.6, -0.7, 0.45),
                ambient=0.5,
                diffuse=0.8,
                specular=0.0,
                normal_strength=6.0,
                height=0.6,
                shade_background=True,
                background_bevel=8.0,
            ),
            size=(32, 32),
        )
    )
    assert not np.array_equal(background_lit_pixels[16, 6], background_lit_pixels[0, 0])
    assert np.array_equal(background_lit_pixels[0, 0], [128, 128, 128, 255])
    groove_pixels = rgba_array(
        render(
            sdf.rounded_rect((16, 16), (13, 10), (3, 3, 3, 3))
            .subtract(sdf.circle((16, 16), 5))
            .light(
                (255, 255, 255, 255),
                (128, 128, 128, 255),
                light_dir=(-0.6, -0.7, 0.45),
                ambient=0.5,
                diffuse=0.8,
                specular=0.0,
                normal_strength=6.0,
                bevel=3.0,
                height=0.6,
                shade_background=True,
                background_bevel=3.0,
            ),
            size=(32, 32),
        )
    )
    assert np.array_equal(groove_pixels[16, 16], [128, 128, 128, 255])
    assert not np.array_equal(groove_pixels[16, 11], groove_pixels[16, 16])
    assert derivative_pixels[0, 0, 0] > derivative_pixels[8, 8, 0]


def test_sdf_boolean_and_transform_methods_render_successfully():
    base = sdf.circle((8, 8), 5)
    other = sdf.circle((10, 8), 5)

    scenes = [
        base.abs(),
        base.repeat(),
        base.union(other),
        base.intersection(other),
        base.subtract(other),
        base.smooth_union(other, k=2.0),
        base.interpolate(other, t=0.25),
    ]

    for scene in scenes:
        texture = render(scene, size=(16, 16))
        assert texture.tex.size == (16, 16)


def test_masked_union_renders_both_outputs():
    result = sdf.circle((8, 8), 5).masked_union(sdf.circle((10, 8), 5))
    sdf_texture, color_texture = result

    assert render(sdf_texture, size=(16, 16)).tex.size == (16, 16)
    assert render(color_texture, size=(16, 16)).tex.size == (16, 16)


def test_save_round_trips_color_and_sdf_outputs(tmp_path):
    color_path = tmp_path / "color.png"
    sdf_path = tmp_path / "sdf.png"

    color.clear("#ffffff").save(str(color_path), size=(8, 8))
    sdf.circle((8, 8), 5).save(str(sdf_path), size=(16, 16))

    with Image.open(color_path) as image:
        assert image.mode == "RGBA"
        assert image.size == (8, 8)

    with Image.open(sdf_path) as image:
        assert image.mode == "L"
        assert image.size == (16, 16)


def test_canvas_session_cache_reuses_rendered_texture_objects():
    scene = color.clear("#ffffff").cache("white")
    cache = {}

    with Canvas((8, 8)) as ctx:
        first = scene.render(ctx, cache=cache)
        second = scene.render(ctx, cache=cache)
        with ctx.session(cache=cache) as renderer:
            third = renderer.render(scene)

        uncached = scene.uncached().render(ctx, cache=cache)

    assert first is second is third
    assert uncached is not first


def test_canvas_session_exposes_render_stats_and_cache_info():
    scene = color.clear("#ffffff").to_rgb()
    cache = {}

    with Canvas((8, 8)) as ctx:
        with ctx.session(cache=cache) as renderer:
            first = renderer.render(scene)
            second = renderer.render(scene)
            cache_info = renderer.cache_info()
            stats = renderer.stats

        assert ctx.last_render_stats is stats

    assert first is second
    assert cache_info.entries == 2
    assert cache_info.hits >= 1
    assert cache_info.misses >= 2
    assert stats.shader_dispatches == 2
    assert stats.texture_allocations == 2
    assert stats.render_calls == 2
    assert stats.elapsed_seconds > 0
    assert stats.as_dict()["shader_dispatches_by_name"] == {
        "clear_color": 1,
        "to_rgb": 1,
    }
