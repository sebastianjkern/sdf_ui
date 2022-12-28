import PIL
import numpy as np

from console import fg, fx, bg


def get_average_l(image):
    im = np.array(image)
    w, h = im.shape
    return np.average(im.reshape(w * h))


def get_average_c(image):
    im = np.array(image)
    return tuple(np.array(np.mean(im, axis=(0, 1)).astype(int)))


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb[:3]


def convert_image_to_ascii_colored(image, cols, scale, more_levels, invert, enhance, as_background):
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
