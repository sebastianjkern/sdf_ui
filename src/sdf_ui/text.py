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
):
    """
    Generate a signed distance field (SDF) for a text run.

    Glyph SDF patches are cached by font, character, curve sampling, and cache
    resolution, then resized into the destination texture during layout.
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
    )
