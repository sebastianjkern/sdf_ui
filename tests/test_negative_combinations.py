from __future__ import annotations

import pytest

from sdf_ui import color, sdf
from sdf_ui.core.plugins.registry import registry


def _node(kind: str):
    if kind == "sdf":
        return sdf.circle((8, 8), 5)
    if kind == "color":
        return color.clear("#ffffff")
    raise ValueError(kind)


@pytest.mark.parametrize(
    ("name", "args", "kwargs", "match"),
    [
        ("union", ("sdf", "color"), {}, "input 1 should be sdf"),
        ("intersection", ("sdf", "color"), {}, "input 1 should be sdf"),
        ("subtract", ("sdf", "color"), {}, "input 1 should be sdf"),
        ("smooth_union", ("sdf", "color"), {}, "input 1 should be sdf"),
        ("masked_union", ("sdf", "color"), {}, "input 1 should be sdf"),
        ("interpolate", ("sdf", "color"), {}, "input 1 should be sdf"),
        ("fill", ("color",), {"fg_color": (255, 255, 255, 255)}, "input 0 should be sdf"),
        ("outline", ("color",), {"fg_color": (255, 255, 255, 255)}, "input 0 should be sdf"),
        ("fill_from_texture", ("sdf", "sdf"), {}, "input 1 should be color"),
        ("generate_mask", ("color",), {}, "input 0 should be sdf"),
        ("shadow", ("color",), {}, "input 0 should be sdf"),
        ("partial_derivative", ("color",), {}, "input 0 should be sdf"),
        ("abs", ("color",), {}, "input 0 should be sdf"),
        ("repeat", ("color",), {}, "input 0 should be sdf"),
        ("alpha_overlay", ("sdf", "color"), {}, "input 0 should be color"),
        ("mask", ("sdf", "color", "color"), {}, "input 0 should be color"),
        ("multiply", ("sdf", "color"), {}, "input 0 should be color"),
        ("transparency", ("sdf",), {"alpha": 0.5}, "input 0 should be color"),
        ("blur", ("sdf",), {}, "input 0 should be color"),
        ("blur_9", ("sdf",), {}, "input 0 should be color"),
        ("blur_13", ("sdf",), {}, "input 0 should be color"),
        ("invert", ("sdf",), {}, "input 0 should be color"),
        ("dithering", ("sdf",), {}, "input 0 should be color"),
        ("dither_1bit", ("sdf",), {}, "input 0 should be color"),
    ],
)
def test_registry_rejects_wrong_texture_kinds(name, args, kwargs, match):
    args = tuple(_node(kind) if kind in {"sdf", "color"} else kind for kind in args)
    with pytest.raises(TypeError, match=match):
        registry.build(name, *args, **kwargs)


@pytest.mark.parametrize(
    ("texture_kind", "attribute"),
    [
        ("sdf", "alpha_overlay"),
        ("sdf", "blur"),
        ("sdf", "invert"),
        ("color", "fill"),
        ("color", "outline"),
        ("color", "shadow"),
        ("color", "generate_mask"),
        ("color.post", "fill"),
        ("color.post", "shadow"),
        ("sdf", "post"),
    ],
)
def test_incompatible_methods_are_hidden(texture_kind, attribute):
    texture = _node("sdf" if texture_kind == "sdf" else "color")
    if texture_kind == "color.post":
        texture = texture.post

    with pytest.raises(AttributeError):
        getattr(texture, attribute)
