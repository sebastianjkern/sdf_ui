__docformat__ = "google"

import PIL.Image
from sdf_ui import *
import PIL
import numpy
from functools import reduce

def image_to_sdf(ctx: Context, path: str, resize: bool = True):
    """
    Convert a image to a signed distance field (SDF).
    
    Args:
        ctx (Context): The context used for creating SDFs.
        path (str): The path to the input image file.
        resize (bool, optional): Whether to resize the image to 128x128 pixels. Default is True.
        
    Returns:
        SDF: A signed distance field representation of the input image.
        
    Example:
        sdf = image_to_sdf(ctx, "path/to/image.png")
        
    Notes:
        - The function reads the input image and optionally resizes it to 128x128 pixels.
        - It converts the image to a black-and-white format by averaging the color channels.
        - Pixels darker than a specified threshold are identified.
        - For each dark pixel, a disc is created and combined into a single SDF using smooth union operations.
        
    """

    # Read image
    image = PIL.Image.open(path)
    if resize:
        image = image.resize((128, 128))
    pix = numpy.array(image)

    # Convert color image to bw image
    # TODO: Improve file format compatibility
    pix = (pix @ [1, 1, 1, 0]) / 3
    # pix = (pix @ [1, 1, 1]) / 3

    PIL.Image.fromarray(pix).show()

    # find pixels that are darker than the threshold
    i, j = numpy.where(pix < 150)
    center_points = list(zip(i, j))

    # combine pixel sdfs to full sdf
    discs = [disc(ctx, (x[0], x[1]), 1.75) for x in center_points]
    return reduce(lambda x, y: x.smooth_union(y, k=1.5), discs)
