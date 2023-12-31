__docformat__ = "google"

import PIL
import numpy as np

from PIL import Image

from console import fg, fx, bg


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
    return '%02x%02x%02x' % rgb[:3]


def convert_image_to_ascii_colored(image, cols, scale, more_levels, invert, enhance, as_background):
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
    hr_ascii_table = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    lr_ascii_table = '@%#*+=-:. '

    bw_image = image.convert('L')

    if enhance:
        image = PIL.ImageEnhance.Contrast(image).enhance(3)
        image = PIL.ImageEnhance.Brightness(image).enhance(0.85)

        bw_image = PIL.ImageEnhance.Contrast(bw_image).enhance(3)
        bw_image = PIL.ImageEnhance.Brightness(bw_image).enhance(0.85)

    if invert:
        image = PIL.ImageOps.invert(image)
        bw_image = PIL.ImageOps.invert(bw_image)

    original_width, original_height = image.size[0], image.size[1]

    new_width = original_width / cols
    new_height = new_width / scale

    rows = int(original_height / new_height)

    if cols > original_width or rows > original_height:
        # TODO: Upscale in case of low input resolution
        exit(0)

    ascii_image = []
    for j in range(rows):
        y1 = int(j * new_height)
        y2 = int((j + 1) * new_height)

        if j == rows - 1:
            y2 = original_height

        ascii_image.append("")

        for i in range(cols):
            x1 = int(i * new_width)
            x2 = int((i + 1) * new_width)
            if i == cols - 1:
                x2 = original_width

            img = image.crop((x1, y1, x2, y2))
            bg_img = bw_image.crop((x1, y1, x2, y2))

            avg = int(get_average_l(bg_img))
            clr = get_average_c(img)

            if more_levels:
                symbol = hr_ascii_table[int((avg * 69) / 255)]
            else:
                symbol = lr_ascii_table[int((avg * 9) / 255)]

            if as_background:
                symbol = getattr(bg, "t_" + rgb_to_hex(clr)) + ' ' + fx.end
            else:
                symbol = getattr(fg, "t_" + rgb_to_hex(clr)) + symbol + fx.end

            ascii_image[j] += symbol

    for row in ascii_image:
        print(row)


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

    hr_ascii_table = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    lr_ascii_table = '@%#*+=-:. '

    image = image.convert('L')

    if enhance:
        image = PIL.ImageEnhance.Contrast(image).enhance(3)
        image = PIL.ImageEnhance.Brightness(image).enhance(0.85)

    if invert:
        image = PIL.ImageOps.invert(image)

    original_width, original_height = image.size[0], image.size[1]

    new_width = original_width / cols
    new_height = new_width / scale

    rows = int(original_height / new_height)

    if cols > original_width or rows > original_height:
        # TODO: Upscale in case of low input resolution
        exit(0)

    ascii_image = []
    for j in range(rows):
        y1 = int(j * new_height)
        y2 = int((j + 1) * new_height)
        if j == rows - 1:
            y2 = original_height
        ascii_image.append("")

        for i in range(cols):
            x1 = int(i * new_width)
            x2 = int((i + 1) * new_width)
            if i == cols - 1:
                x2 = original_width

            img = image.crop((x1, y1, x2, y2))
            avg = int(get_average_l(img))

            if more_levels:
                greyscale_value = hr_ascii_table[int((avg * 69) / 255)]
            else:
                greyscale_value = lr_ascii_table[int((avg * 9) / 255)]

            ascii_image[j] += greyscale_value

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
        convert_image_to_ascii_colored(image, 150, 0.43, more_levels=False, invert=False, enhance=True,
                                       as_background=True)
