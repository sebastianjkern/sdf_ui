# SDF UI

___

<img src="./image1.png" width="500" > <img src="./image2.png" width="500" >

## Example

```python
import random

from framework.context import Context, hex_col

size = (1920, 1080)


def rand_color():
    red = random.randint(0, 255) / 255
    green = random.randint(0, 255) / 255
    blue = random.randint(0, 255) / 255
    return red, green, blue


with Context(size) as ctx:
    sdf = ctx.disc((250, 250), 200)
    bg = hex_col("#2C2D35")
    layer = ctx.fill(sdf, (*rand_color(), 1.0), bg, inflate=0)

    for _ in range(20):
        x, y = random.randint(0, size[0]), random.randint(0, size[1])
        r = random.randint(10, 250)
        sdf = ctx.disc((x, y), r)
        col0 = rand_color()
        col1 = (*col0, 1.0)
        col2 = (*col0, 0.0)
        f = ctx.fill(sdf, col1, col2, inflate=0.5)
        layer = ctx.overlay(f, layer)

    blur = ctx.blur(layer, 100)

    # Mask and glass overlay
    mask_sdf = ctx.rounded_rect((int(size[0] / 2), int(size[1] / 2)), (int(size[0] / 5), int(size[1] / 3)),
                                (150, 150, 150, 150))

    mask_layer = ctx.fill(mask_sdf, (.0, .0, .0, 1.0), (1.0, 1.0, 1.0, 1.0), 0)

    overlay_outline = ctx.outline(mask_sdf, (1.0, 1.0, 1.0, .25), (0.75, 0.75, 0.75, 0.0), inflate=-1.5)

    glass = ctx.fill(mask_sdf, (44 / 255, 45 / 255, 53 / 255, 0.15), (0.0, 0.0, 0.0, 0.0), 0)

    masked = ctx.mask(blur, layer, mask_layer)
    overlay = ctx.overlay(glass, masked)
    overlay = ctx.overlay(overlay_outline, overlay)

    overlay = ctx.to_rgb(overlay)
    overlay.show()
    overlay.save("image1.png")

with Context(size) as ctx:
    scale = 0.65
    offset_x = 50
    offset_y = 50

    glyph_sdf = ctx.glyph("M", scale, offset_x, offset_y)
    glyph_sdf.show()

    layer = ctx.fill(glyph_sdf, hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), 7.5)
    bg = ctx.fill(glyph_sdf, hex_col("#2C2D35"), hex_col("#2C2D35"), 7.5)

    mask = ctx.fill(glyph_sdf, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), 7.5)
    shadow = ctx.blur(mask, 10)
    with_shadow = ctx.overlay(layer, shadow)

    overlay = ctx.overlay(with_shadow, bg)
    overlay = ctx.to_rgb(overlay)

    overlay.show()
    overlay.save("image2.png")

```
