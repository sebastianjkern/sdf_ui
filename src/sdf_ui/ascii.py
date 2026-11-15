__docformat__ = "google"

import numpy as np
import PIL
from console import bg, fg, fx
from PIL import Image


def _validate_ascii_dimensions(cols, scale, original_width, original_height, rows):
    if cols <= 0:
        raise ValueError("cols must be greater than 0")
    if scale <= 0:
        raise ValueError("scale must be greater than 0")
    if cols > original_width or rows > original_height:
        raise ValueError(
            "the requested ASCII output is larger than the source image; "
            "choose fewer columns or a smaller scale"
        )


def get_average_l(image):
    """
    Get the average luminance value of an image.

    Parameters:
    - image (Image): The input image.

    Returns:
    - float: The average luminance value.

    Example:
    >>> input_image = Image.open("example.jpg")
    >>> average_luminance = get_average_l(input_image)
    """
    im = np.array(image)
    w, h = im.shape
    return np.average(im.reshape(w * h))


def get_average_c(image):
    """
    Get the average color value of an image.

    Parameters:
    - image (Image): The input image.

    Returns:
    - tuple: The average color value as a tuple (R, G, B).

    Example:
    >>> input_image = Image.open("example.jpg")
    >>> average_color = get_average_c(input_image)
    """
    im = np.array(image)
    return tuple(np.array(np.mean(im, axis=(0, 1)).astype(int)))


def rgb_to_hex(rgb):
    """
    Convert an RGB tuple to a hexadecimal color code.

    Parameters:
    - rgb (tuple): RGB values as a tuple (R, G, B).

    Returns:
    - str: Hexadecimal color code.

    Example:
    >>> color_tuple = (255, 0, 128)
    >>> hex_color = rgb_to_hex(color_tuple)
    """
    return "%02x%02x%02x" % rgb[:3]


def _prepare_ascii_images(image, invert, enhance):
    """Prepare the color and luminance images shared by both render modes."""
    luminance = image.convert("L")
    if enhance:
        image = PIL.ImageEnhance.Contrast(image).enhance(3)
        image = PIL.ImageEnhance.Brightness(image).enhance(0.85)
        luminance = PIL.ImageEnhance.Contrast(luminance).enhance(3)
        luminance = PIL.ImageEnhance.Brightness(luminance).enhance(0.85)
    if invert:
        image = PIL.ImageOps.invert(image)
        luminance = PIL.ImageOps.invert(luminance)
    return image, luminance


def _ascii_rows(width, height, cols, scale):
    """Yield row index and pixel bounds for the shared ASCII sampling grid."""
    if cols <= 0:
        raise ValueError("cols must be greater than 0")
    if scale <= 0:
        raise ValueError("scale must be greater than 0")
    cell_width = width / cols
    cell_height = cell_width / scale
    rows = int(height / cell_height)
    _validate_ascii_dimensions(cols, scale, width, height, rows)
    for row in range(rows):
        y1 = int(row * cell_height)
        y2 = height if row == rows - 1 else int((row + 1) * cell_height)
        cells = []
        for column in range(cols):
            x1 = int(column * cell_width)
            x2 = width if column == cols - 1 else int((column + 1) * cell_width)
            cells.append((x1, x2, y1, y2))
        yield row, cells


def _ascii_character(luminance, more_levels):
    table = (
        "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
        if more_levels
        else "@%#*+=-:. "
    )
    levels = len(table) - 1
    return table[min(int(luminance * levels / 255), levels)]


def convert_image_to_ascii_colored(
    image, cols, scale, more_levels, invert, enhance, as_background
):
    """
    Convert a colored image to colored ASCII art and print it.

    Parameters:
    - image (Image): The input image.
    - cols (int): Number of columns in the ASCII output.
    - scale (float): Scaling factor for the ASCII art.
    - more_levels (bool): If True, use a higher level of ASCII characters for better detail.
    - invert (bool): If True, invert the colors of the image.
    - enhance (bool): If True, enhance the contrast and brightness of the image.
    - as_background (bool): If True, use the color as the background.

    Example:
    >>> input_image = Image.open("example.jpg")
    >>> convert_image_to_ascii_colored(input_image, 80, 0.43, True, False, True, True)
    """
    image, bw_image = _prepare_ascii_images(image, invert, enhance)
    original_width, original_height = image.size
    for _row, cells in _ascii_rows(original_width, original_height, cols, scale):
        print(
            "".join(
                _colored_ascii_cell(
                    image, bw_image, x1, x2, y1, y2, more_levels, as_background
                )
                for x1, x2, y1, y2 in cells
            )
        )


def _colored_ascii_cell(image, bw_image, x1, x2, y1, y2, more_levels, as_background):
    img = image.crop((x1, y1, x2, y2))
    bg_img = bw_image.crop((x1, y1, x2, y2))
    avg = int(get_average_l(bg_img))
    color = rgb_to_hex(get_average_c(img))
    symbol = _ascii_character(avg, more_levels)
    if as_background:
        return getattr(bg, "t_" + color) + " " + fx.end
    return getattr(fg, "t_" + color) + symbol + fx.end


def convert_image_to_ascii(image, cols, scale, more_levels, invert, enhance):
    """
    Convert an image to ASCII art and print it.

    Parameters:
    - image (Image): The input image.
    - cols (int): Number of columns in the ASCII output.
    - scale (float): Scaling factor for the ASCII art.
    - more_levels (bool): If True, use a higher level of ASCII characters for better detail.
    - invert (bool): If True, invert the colors of the image.
    - enhance (bool): If True, enhance the contrast and brightness of the image.

    Example:
    >>> input_image = Image.open("example.jpg")
    >>> convert_image_to_ascii(input_image, 80, 0.43, True, False, True)
    """

    image, _ = _prepare_ascii_images(image.convert("L"), invert, enhance)
    original_width, original_height = image.size
    ascii_image = []
    for _row, cells in _ascii_rows(original_width, original_height, cols, scale):
        row = []
        for x1, x2, y1, y2 in cells:
            img = image.crop((x1, y1, x2, y2))
            avg = int(get_average_l(img))
            row.append(_ascii_character(avg, more_levels))
        ascii_image.append("".join(row))

    for row in ascii_image:
        print(row)


def print_texture(texture, colored=True):
    """
    Print the content of a texture using ASCII art.

    Parameters:
    - texture (ColorTexture): The ColorTexture to print.
    - colored (bool): If True, print in colored ASCII art. If False, print in grayscale.

    Example:
    >>> color_texture = ColorTexture(...)
    >>> print_texture(color_texture)
    """
    image = Image.frombytes("RGBA", texture.tex.size, texture.tex.read(), "raw")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if not colored:
        convert_image_to_ascii(image, 150, 0.43, True, True, True)
    else:
        convert_image_to_ascii_colored(
            image,
            150,
            0.43,
            more_levels=False,
            invert=False,
            enhance=True,
            as_background=True,
        )
