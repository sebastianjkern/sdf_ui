import random
import time

from framework.context import Context, hex_col, get_tex_registry

size = (int(1080), int(1080))


def rand_color():
    red = random.randint(0, 255) / 255
    green = random.randint(0, 255) / 255
    blue = random.randint(0, 255) / 255
    return red, green, blue


def rand_point():
    x = random.randint(0, size[0])
    y = random.randint(0, size[1])

    return x, y


# logger().setLevel(level=logging.CRITICAL)

COLORS = [
    "#62bb47",
    "#fcb827",
    "#f6821f",
    "#e03a3c",
    "#963d97",
    "#009ddc"
]

# Test with freeform Gradients
with Context(size) as ctx:
    base_color = ctx.clear_color(hex_col(random.choice(COLORS)))

    col0 = random.choice(COLORS)
    col1 = random.choice(COLORS)
    col2 = random.choice(COLORS)
    col3 = random.choice(COLORS)

    radial1 = ctx.radial_gradient((100, 100), hex_col(col0, alpha=150), hex_col(col0, alpha=0.0), inner=50, outer=750)
    radial2 = ctx.radial_gradient((750, 500), hex_col(col1, alpha=255), hex_col(col1, alpha=0.0), inner=50, outer=750)
    radial3 = ctx.radial_gradient((100, 750), hex_col(col2, alpha=180), hex_col(col2, alpha=0.0), inner=50, outer=750)

    overlay0 = ctx.overlay(radial1, base_color)
    overlay1 = ctx.overlay(radial2, overlay0)
    overlay2 = ctx.overlay(radial3, overlay1)

    bezier = ctx.bezier(rand_point(), rand_point(), rand_point())
    filled = ctx.fill(bezier, hex_col(col3, alpha=150), hex_col(col3, alpha=0), inflate=0, inner=0, outer=250)

    overlay1.delete()
    overlay1 = ctx.overlay(filled, overlay2)

    overlay0.delete()
    overlay0 = ctx.to_rgb(overlay1)
    # overlay0.show()

    bezier.delete()
    filled.delete()

    base_color.delete()
    radial1.delete()
    radial2.delete()
    radial3.delete()

    overlay1.delete()
    overlay2.delete()

    grain = ctx.film_grain()

    temp = ctx.transparency(grain, 10 / 255)
    grain.delete()

    out = ctx.overlay(temp, overlay0)

    out.show()
    out.save("image3.png")

    temp.delete()
    overlay0.delete()
    out.delete()

    get_tex_registry()

with Context(size) as ctx:
    for _ in range(1):
        start = time.time_ns()
        radial_gradient_backdrop = ctx.radial_gradient((size[0] / 2, size[1] / 2), hex_col("#004C81", alpha=255),
                                                       hex_col("#062A4A", alpha=255), inner=0,
                                                       outer=ctx.percent_of_min(75))

        sdf = ctx.grid((10, 10), (50, 50))
        grid = ctx.fill(sdf, hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=.5)
        thin_grid = ctx.overlay(grid, radial_gradient_backdrop)

        sdf.delete()
        grid.delete()
        radial_gradient_backdrop.delete()

        sdf = ctx.grid((10, 10), (150, 150))
        grid = ctx.fill(sdf, hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=1.5)
        layer = ctx.overlay(grid, thin_grid)

        sdf.delete()
        grid.delete()
        thin_grid.delete()

        for _ in range(5):
            x, y = random.randint(50, size[0] - 50), random.randint(50, size[1] - 50)
            r = random.randint(10, 100)
            sdf = ctx.disc((x, y), r)
            col0 = hex_col(random.choice(COLORS))[:3]
            col1 = (*col0, 1.0)
            col2 = (*col0, 0.0)
            f = ctx.fill(sdf, col1, col2, 0)
            temp = ctx.overlay(f, layer)
            layer.delete()
            layer = temp
            f.delete()
            sdf.delete()

        gradient = ctx.linear_gradient((100, 100), (900, 100), (*rand_color(), 1.0), (*rand_color(), 1.0))
        blur = ctx.blur(layer, n=15, base=13)

        # Mask and glass overlay
        mask_sdf = ctx.rounded_rect(
            (int(size[0] / 2), int(size[1] / 2)),
            (int(size[1] / 3), int(size[1] / 3)),
            (size[0] / 10, size[0] / 10, size[0] / 10, size[0] / 10))

        mask_layer = ctx.generate_mask(mask_sdf)
        overlay_outline = ctx.outline(mask_sdf, (1.0, 1.0, 1.0, .25), (1.0, 1.0, 1.0, 0.0), inflate=-1.5)

        # glass_col = (44 / 200, 45 / 200, 53 / 200, 0.45)
        glass_col = (0.75, 0.75, 0.75, 0.75)
        glass = ctx.fill(mask_sdf, glass_col, (0.0, 0.0, 0.0, 0.0), 0)

        TRANSPARENT = ctx.clear_color((0.0, 0.0, 0.0, 0.0))

        transparent_gradient = ctx.transparency(gradient, 0.45)

        gradient.delete()

        gradient_masked = ctx.fill_from_texture(mask_sdf, transparent_gradient)
        gradient_masked.show()
        masked = ctx.mask(blur, layer, mask_layer)
        overlay = ctx.overlay(glass, masked)
        with_outline = ctx.overlay(overlay_outline, overlay)
        overlay.delete()

        overlay = ctx.to_rgb(with_outline)
        # overlay = ctx.dithering(overlay)
        overlay.save("image1.png")

        mask_sdf.delete()
        transparent_gradient.delete()
        TRANSPARENT.delete()
        with_outline.delete()
        overlay.delete()
        masked.delete()
        gradient_masked.delete()
        glass.delete()
        overlay_outline.delete()
        mask_layer.delete()
        blur.delete()
        layer.delete()

        get_tex_registry()

        print((time.time_ns() - start) / 1e6)

exit()

with Context(size) as ctx:
    scale = 0.65
    offset_x = 50
    offset_y = 50

    glyph_sdf = ctx.glyph("e", scale, offset_x, offset_y, path="fonts/georgia_regular.ttf")
    glyph_sdf.show()

    radial_gradient_backdrop = ctx.fill(glyph_sdf, hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), 7.5)
    bg = ctx.fill(glyph_sdf, hex_col("#2C2D35"), hex_col("#2C2D35"), 7.5)

    mask = ctx.fill(glyph_sdf, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), 7.5)
    shadow = ctx.blur(mask, n=10, base=13)
    with_shadow = ctx.overlay(radial_gradient_backdrop, shadow)

    overlay = ctx.overlay(with_shadow, bg)
    overlay = ctx.to_rgb(overlay)
    overlay = ctx.dithering(overlay)

    overlay.show()
    overlay.save("image2.png")
