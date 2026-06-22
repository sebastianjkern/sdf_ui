__docformat__ = "google"

from .core.plugins.common import build


def glyph(char, scale, ox, oy, path="fonts/SFUIDisplay-Bold.ttf", samples=16):
    """
    Generate the signed distance field (SDF) for a glyph.

    Parameters:
    - char (str): The character for which to generate the SDF.
    - scale (float): Scaling factor for the glyph.
    - ox, oy (float): Offset of the glyph.
    - path (str): Path to the font file.
    - samples (int): Number of line segments used per quadratic Bezier.

    Returns:
        SDFTexture: A render node producing the filled glyph SDF.
    """
    if len(char) != 1:
        raise ValueError("glyph expects exactly one character")
    return build(
        "glyph",
        char=char,
        scale=scale,
        ox=ox,
        oy=oy,
        path=path,
        samples=samples,
    )


def text(
    value,
    size=64,
    ox=0,
    oy=0,
    path="fonts/georgia_regular.ttf",
    samples=16,
    cache_size=128,
    line_height=1.2,
    oversample=2.0,
    min_render_size=64,
):
    """
    Generate a signed distance field (SDF) for a text run.

    Glyph SDF patches are cached by font, character, curve sampling, and
    requested pixel size. ``cache_size`` sets the minimum cached pixels-per-em
    value, ``oversample`` raises the cache resolution for larger text, and
    ``min_render_size`` keeps small text from losing too much SDF distance when
    resized down.
    """
    return build(
        "text",
        value=value,
        size=size,
        ox=ox,
        oy=oy,
        path=path,
        samples=samples,
        cache_size=cache_size,
        line_height=line_height,
        oversample=oversample,
        min_render_size=min_render_size,
    )
