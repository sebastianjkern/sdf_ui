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
