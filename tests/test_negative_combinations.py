from __future__ import annotations

import pytest

from sdf_ui import color, sdf
from sdf_ui.gpu.nodes.registry import registry

_COLOR_TEXT = 'color'

_INPUT_0_SHOULD_BE_COLOR_TEXT = 'input 0 should be color'

_INPUT_0_SHOULD_BE_SDF_TEXT = 'input 0 should be sdf'

_INPUT_1_SHOULD_BE_SDF_TEXT = 'input 1 should be sdf'

def _node(kind: str):
    if kind == "sdf":
        return sdf.circle((8, 8), 5)
    if kind == _COLOR_TEXT:
        return color.clear("#ffffff")
    raise ValueError(kind)


@pytest.mark.parametrize(
    ("name", "args", "kwargs", "match"),
    [
        ("union", ("sdf", _COLOR_TEXT), {}, _INPUT_1_SHOULD_BE_SDF_TEXT),
        ("intersection", ("sdf", _COLOR_TEXT), {}, _INPUT_1_SHOULD_BE_SDF_TEXT),
        ("subtract", ("sdf", _COLOR_TEXT), {}, _INPUT_1_SHOULD_BE_SDF_TEXT),
        ("smooth_union", ("sdf", _COLOR_TEXT), {}, _INPUT_1_SHOULD_BE_SDF_TEXT),
        ("masked_union", ("sdf", _COLOR_TEXT), {}, _INPUT_1_SHOULD_BE_SDF_TEXT),
        ("interpolate", ("sdf", _COLOR_TEXT), {}, _INPUT_1_SHOULD_BE_SDF_TEXT),
        (
            "fill",
            (_COLOR_TEXT,),
            {"fg_color": (255, 255, 255, 255)},
            _INPUT_0_SHOULD_BE_SDF_TEXT,
        ),
        (
            "outline",
            (_COLOR_TEXT,),
            {"fg_color": (255, 255, 255, 255)},
            _INPUT_0_SHOULD_BE_SDF_TEXT,
        ),
        ("fill_from_texture", ("sdf", "sdf"), {}, "input 1 should be color"),
        ("generate_mask", (_COLOR_TEXT,), {}, _INPUT_0_SHOULD_BE_SDF_TEXT),
        ("shadow", (_COLOR_TEXT,), {}, _INPUT_0_SHOULD_BE_SDF_TEXT),
        ("partial_derivative", (_COLOR_TEXT,), {}, _INPUT_0_SHOULD_BE_SDF_TEXT),
        ("abs", (_COLOR_TEXT,), {}, _INPUT_0_SHOULD_BE_SDF_TEXT),
        ("repeat", (_COLOR_TEXT,), {}, _INPUT_0_SHOULD_BE_SDF_TEXT),
        ("alpha_overlay", ("sdf", _COLOR_TEXT), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("mask", ("sdf", _COLOR_TEXT, _COLOR_TEXT), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("multiply", ("sdf", _COLOR_TEXT), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("transparency", ("sdf",), {"alpha": 0.5}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("blur", ("sdf",), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("blur_9", ("sdf",), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("blur_13", ("sdf",), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("invert", ("sdf",), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("dithering", ("sdf",), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
        ("dither_1bit", ("sdf",), {}, _INPUT_0_SHOULD_BE_COLOR_TEXT),
    ],
)
def test_registry_rejects_wrong_texture_kinds(name, args, kwargs, match):
    args = tuple(_node(kind) if kind in {"sdf", _COLOR_TEXT} else kind for kind in args)
    with pytest.raises(TypeError, match=match):
        registry.build(name, *args, **kwargs)


@pytest.mark.parametrize(
    ("texture_kind", "attribute"),
    [
        ("sdf", "alpha_overlay"),
        ("sdf", "blur"),
        ("sdf", "invert"),
        (_COLOR_TEXT, "fill"),
        (_COLOR_TEXT, "outline"),
        (_COLOR_TEXT, "shadow"),
        (_COLOR_TEXT, "generate_mask"),
        ("color.post", "fill"),
        ("color.post", "shadow"),
        ("sdf", "post"),
    ],
)
def test_incompatible_methods_are_hidden(texture_kind, attribute):
    texture = _node("sdf" if texture_kind == "sdf" else _COLOR_TEXT)
    if texture_kind == "color.post":
        texture = texture.post

    with pytest.raises(AttributeError):
        getattr(texture, attribute)
