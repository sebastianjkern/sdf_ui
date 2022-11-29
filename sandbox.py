import random

from framework.context import Context, hex_col

size = (1080, 1080)


def rand_color():
    red = random.randint(0, 255) / 255
    green = random.randint(0, 255) / 255
    blue = random.randint(0, 255) / 255
    return red, green, blue


with Context(size) as ctx:
    sdf = ctx.disc((size[0] / 2, size[1] / 2), 0)
    bg = hex_col("#2C2D35")
    layer = ctx.fill(sdf, (*rand_color(), 1.0), (*rand_color(), 1.0), inflate=0, inner=0, outer=750)

    grid_col = rand_color()

    sdf = ctx.grid((10, 10), (50, 50))
    grid = ctx.fill(sdf, (*grid_col, 1.0), (*grid_col, 0.0), inflate=.5)
    layer = ctx.overlay(grid, layer)

    sdf = ctx.grid((10, 10), (150, 150))
    grid = ctx.fill(sdf, (*grid_col, 1.0), (*grid_col, 0.0), inflate=1.5)
    layer = ctx.overlay(grid, layer)

    for _ in range(25):
        x, y = random.randint(50, size[0] - 50), random.randint(50, size[1] - 50)
        r = random.randint(10, 100)
        sdf = ctx.disc((x, y), r)
        col0 = rand_color()
        col1 = (*col0, 1.0)
        col2 = (*col0, 0.0)
        f = ctx.fill(sdf, col1, col2, 0)
        layer = ctx.overlay(f, layer)

    sdf = ctx.line((100, 100), (900, 100))
    gradient = ctx.fill(sdf, (*rand_color(), 1.0), (*rand_color(), 1.0), inflate=0, inner=0, outer=500)

    blur = ctx.blur(layer, n=25, base=13)

    rgb = ctx.to_rgb(blur)
    rgb.save("blur.png")

    # Mask and glass overlay
    mask_sdf = ctx.rounded_rect((int(size[1] / 2), int(size[1] / 2)), (int(size[1] / 3), int(size[1] / 3)),
                                (150, 150, 150, 150))

    mask_layer = ctx.fill(mask_sdf, (.0, .0, .0, 1.0), (1.0, 1.0, 1.0, 1.0), 0)

    overlay_outline = ctx.outline(mask_sdf, (1.0, 1.0, 1.0, .25), (0.75, 0.75, 0.75, 0.0), inflate=-1.5)

    # glass_col = (44 / 200, 45 / 200, 53 / 200, 0.45)
    glass_col = (0.85, 0.75, 0.85, 0.45)
    glass = ctx.fill(mask_sdf, glass_col, (0.0, 0.0, 0.0, 0.0), 0)

    TRANSPARENT = ctx.clear_color((0.0, 0.0, 0.0, 0.0))

    gradient = ctx.transparency(gradient, 0.45)

    gradient_masked = ctx.mask(gradient, TRANSPARENT, mask_layer)
    masked = ctx.mask(blur, layer, mask_layer)
    overlay = ctx.overlay(glass, masked)
    overlay = ctx.overlay(overlay_outline, overlay)

    overlay = ctx.to_rgb(overlay)
    # overlay.save("image1_wod.png")
    overlay = ctx.dithering(overlay)
    overlay.save("image1.png")

exit()

with Context(size) as ctx:
    scale = 0.65
    offset_x = 50
    offset_y = 50

    glyph_sdf = ctx.glyph("e", scale, offset_x, offset_y, path="fonts/georgia_regular.ttf")
    glyph_sdf.show()

    layer = ctx.fill(glyph_sdf, hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), 7.5)
    bg = ctx.fill(glyph_sdf, hex_col("#2C2D35"), hex_col("#2C2D35"), 7.5)

    mask = ctx.fill(glyph_sdf, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), 7.5)
    shadow = ctx.blur(mask, n=10, base=13)
    with_shadow = ctx.overlay(layer, shadow)

    overlay = ctx.overlay(with_shadow, bg)
    overlay = ctx.to_rgb(overlay)
    overlay = ctx.dithering(overlay)

    overlay.show()
    overlay.save("image2.png")
