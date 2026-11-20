__docformat__ = "google"

from functools import reduce

import numpy
import PIL
import PIL.Image

from . import sdf


def image_to_sdf(path: str, resize: bool = True, preview: bool = False):
    """
    Convert a image to a signed distance field (SDF).

    Args:
        path (str): The path to the input image file.
        resize (bool, optional): Whether to resize the image to 128x128 pixels. Default is True.
        preview (bool, optional): Whether to show the threshold source image. Default is False.

    Returns:
        SDF: A signed distance field representation of the input image.

    Example:
        sdf = image_to_sdf("path/to/image.png")

    Notes:
        - The function reads the input image and optionally resizes it to 128x128 pixels.
        - It converts the image to a black-and-white format by averaging the color channels.
        - Pixels darker than a specified threshold are identified.
        - For each dark pixel, a disc is created and combined into a single SDF using smooth union operations.

    """

    image: PIL.Image.Image = PIL.Image.open(path)
    if resize:
        image = image.resize((128, 128))
    pix = numpy.array(image)

    pix = (pix @ [1, 1, 1, 0]) / 3

    if preview:
        PIL.Image.fromarray(pix).show()

    i, j = numpy.where(pix < 150)
    center_points = list(zip(i, j))
    if not center_points:
        raise ValueError(
            f"No dark pixels found in '{path}', so no SDF texture can be generated"
        )

    discs = [sdf.circle((x[0], x[1]), 1.75) for x in center_points]
    return reduce(lambda x, y: x.smooth_union(y, k=1.5), discs)
