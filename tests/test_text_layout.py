import numpy as np
import pytest

from sdf_ui import fonts, sdf
from sdf_ui.gpu.nodes.primitives.svg.plugin import _collect_shapes, _parse_svg
from sdf_ui.gpu.nodes.primitives.text.plugin import (
    _composite_glyph,
    _justify_placements,
)
from tests.text_helpers import use_mock_text_shaper


def _unit_width(text):
    return len(fonts._graphemes(text))


def test_wrapping_uses_hyphen_line_break_opportunities(monkeypatch):
    use_mock_text_shaper(monkeypatch)

    lines = fonts.layout_text("hello-world", ("font",), 1, box_width=6)

    assert tuple(
        line for _glyphs, _width, line, _last, _truncated, _first in lines
    ) == (
        "hello-",
        "world",
    )
    assert tuple(
        is_last for _glyphs, _width, _line, is_last, _truncated, _first in lines
    ) == (
        False,
        True,
    )


def test_wrapping_keeps_cjk_punctuation_attached_to_its_preceding_character(
    monkeypatch,
):
    use_mock_text_shaper(monkeypatch)

    lines = fonts.layout_text("あいう、えお", ("font",), 1, box_width=2)

    assert tuple(
        line for _glyphs, _width, line, _last, _truncated, _first in lines
    ) == (
        "あい",
        "う、",
        "えお",
    )


def test_emergency_wrap_does_not_split_combining_graphemes(monkeypatch):
    use_mock_text_shaper(monkeypatch)

    lines = fonts.layout_text("a\u0301bc", ("font",), 1, box_width=1)

    assert tuple(
        line for _glyphs, _width, line, _last, _truncated, _first in lines
    ) == (
        "a\u0301",
        "b",
        "c",
    )


def test_wrap_preserves_nonbreaking_space(monkeypatch):
    use_mock_text_shaper(monkeypatch)

    lines = fonts.layout_text("a\u00a0b", ("font",), 1, box_width=1)

    assert tuple(
        line for _glyphs, _width, line, _last, _truncated, _first in lines
    ) == ("a\u00a0b",)

    measured = fonts.measure_text("a\u00a0b", size=16, box_width=1)
    assert measured["overflow"] is True


def test_wrapping_splits_mandatory_unicode_line_breaks():
    lines = fonts.layout_text("first\u2028second", (fonts.default_font_path(),), 16)

    assert tuple(
        line for _glyphs, _width, line, _last, _truncated, _first in lines
    ) == (
        "first",
        "second",
    )


def test_measure_text_reports_vertical_overflow():
    result = fonts.measure_text(
        "A line that wraps", size=24, box_width=80, box_height=20
    )

    assert len(result["lines"]) > 1
    assert result["overflow"] is True


def test_math_font_size_preserves_glyph_scale_across_formula_heights():
    matplotlib = pytest.importorskip("matplotlib")

    def glyph_size(expression, height):
        node = sdf.math(expression, height=height, font_size=32)
        params = dict(node.params)
        root, view_box = _parse_svg(params["source"])
        shapes = _collect_shapes(root, view_box, (0, 0), params["height"], 32)
        contours, _fill_rule = shapes[0]
        points = np.asarray([point for contour in contours for point in contour])
        return np.ptp(points, axis=0)

    with matplotlib.rc_context({"svg.fonttype": "none"}):
        plain = glyph_size("x", None)
        superscript = glyph_size("x^2", None)
        fixed = glyph_size("x^2", 64)

    np.testing.assert_allclose(plain, superscript)
    assert fixed[1] > superscript[1]


def test_glyph_compositing_can_be_clipped_to_a_text_box():
    field = np.full((5, 5), 10.0, dtype=np.float32)
    patch = {
        "field": np.full((5, 5), -1.0, dtype=np.float32),
        "origin": (0.0, 0.0),
        "units_per_pixel": 1.0,
    }

    _composite_glyph(field, patch, 0, 0, 1, 1, clip_rect=(2, 1, 4, 4))

    assert np.all(field[1:4, 2:4] < 0)
    assert np.all(field[:1, :] > 0)
    assert np.all(field[:, :2] > 0)
    assert np.all(field[:, 4:] > 0)


def test_mixed_direction_runs_and_isolates_are_resolved():
    text = "abc אבג 123"
    isolated = "abc \u2067אבג\u2069 123"

    expected = [("abc ", "ltr"), ("123", "ltr"), ("אבג ", "rtl")]
    assert fonts._bidi_runs(text, None) == expected
    isolated_runs = fonts._bidi_runs(isolated, None)
    assert ("אבג", "rtl") in isolated_runs
    assert all(direction in {"ltr", "rtl"} for _run, direction in isolated_runs)


def test_default_ligatures_can_be_disabled():
    path = fonts.default_font_path()
    default = fonts.layout_text("office", (path,), 32)[0][0]
    no_ligatures = fonts.layout_text("office", (path,), 32, features={"liga": False})[
        0
    ][0]

    assert len(default) < len(no_ligatures)


def test_combining_marks_shape_into_positioned_glyphs():
    shaped = fonts.layout_text("a\u0301", (fonts.default_font_path(),), 32)[0][0]

    assert len(shaped) == 1


def test_justification_distributes_space_on_nonfinal_lines_only():
    path = fonts.default_font_path()
    space = fonts.glyph_id_for_char(path, " ")
    placements = [
        (fonts.glyph_id_for_char(path, "a"), path, 0.0, 0.0),
        (space, path, 10.0, 0.0),
        (fonts.glyph_id_for_char(path, "b"), path, 15.0, 0.0),
    ]

    justified, width = _justify_placements(
        placements, "a b", 20.0, 50.0, paragraph_last=False
    )
    final, final_width = _justify_placements(
        placements, "a b", 20.0, 50.0, paragraph_last=True
    )

    assert justified[2][2] == 45.0
    assert width == 50.0
    assert final == placements
    assert final_width == 20.0


def test_trailing_spaces_can_be_preserved_or_trimmed():
    path = fonts.default_font_path()
    trimmed = fonts.layout_text(
        "abc   ", (path,), 16, box_width=100, trim_trailing_spaces=True
    )[0]
    preserved = fonts.layout_text(
        "abc   ", (path,), 16, box_width=100, trim_trailing_spaces=False
    )[0]

    assert trimmed[2] == "abc"
    assert preserved[2] == "abc   "
    assert preserved[1] > trimmed[1]


def test_overflow_error_mode_rejects_a_too_short_box():
    from types import SimpleNamespace

    import pytest

    from sdf_ui.gpu.nodes.primitives.text.plugin import render_text
    from sdf_ui.gpu.nodes.registry import registry

    params = dict(registry.get("text").defaults)
    params.update(
        value="A line",
        box=(0, 0, 100, 8),
        overflow="error",
    )
    renderer = SimpleNamespace(ctx=SimpleNamespace(size=(100, 40)))

    with pytest.raises(ValueError, match="exceeds the configured box"):
        render_text(renderer, (), params)


def test_clipping_mode_requires_a_text_box():
    from types import SimpleNamespace

    import pytest

    from sdf_ui.gpu.nodes.primitives.text.plugin import render_text
    from sdf_ui.gpu.nodes.registry import registry

    params = dict(registry.get("text").defaults)
    params.update(value="text", overflow="clip")
    renderer = SimpleNamespace(ctx=SimpleNamespace(size=(100, 40)))

    with pytest.raises(ValueError, match="overflow modes require a box"):
        render_text(renderer, (), params)


def test_measure_glyph_exposes_advance_and_ink_bounds():
    from sdf_ui.text import measure_glyph

    measured = measure_glyph("A", size=24)

    assert measured["advance"] > 0
    left, bottom, right, top = measured["ink_bounds"]
    assert right > left
    assert top > bottom


def test_measure_text_exposes_per_line_metrics_and_ink_bounds():
    measured = fonts.measure_text("A short line", size=20, box_width=75)

    assert measured["line_count"] == len(measured["line_metrics"])
    assert all(isinstance(line["baseline"], float) for line in measured["line_metrics"])
    left, bottom, right, top = measured["ink_bounds"]
    assert right > left
    assert top > bottom


def test_inline_spans_build_inside_a_text_box():
    from sdf_ui import sdf
    from sdf_ui.core.sdf import SDFTexture
    from sdf_ui.text import span

    scene = sdf.text(
        [span("A styled ", size=18), span("label", size=26, decoration="underline")],
        box=(10, 10, 120, 50),
        align="center",
        valign="middle",
    )

    assert isinstance(scene, SDFTexture)
