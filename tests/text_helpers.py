from sdf_ui import fonts


def use_mock_text_shaper(monkeypatch):
    monkeypatch.setattr(fonts, "_shape_with_options", _mock_text_shaper)


def _mock_text_shaper(text, paths, size, missing_glyph, options):
    return [], float(len(fonts._graphemes(text))), text
