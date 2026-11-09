"""Paths to fonts distributed with sdf_ui."""

from pathlib import Path


def default_font_path() -> str:
    """Return the packaged Roboto Regular font for text and glyph rendering."""
    return str(Path(__file__).with_name("assets") / "Roboto-Regular.ttf")
