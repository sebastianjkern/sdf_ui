from sdf_ui import badge, glyph, icon


def test_icon_and_glyph_helpers_create_glyph_sdfs():
    assert glyph("A", 0.05, 12, 24).op == "glyph"
    assert icon("A", size=32).op == "text"


def test_badge_composes_background_and_centered_text():
    result = badge("NEW", center=(100, 50))

    assert result.op == "alpha_overlay"
    background, label = result.inputs
    assert background.op == "fill"
    assert background.inputs[0].op == "rounded_rect"
    assert label.op == "fill"
    assert label.inputs[0].op == "text"
