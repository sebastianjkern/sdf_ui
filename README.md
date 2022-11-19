# SDF UI

___

<img src="./image1.png" width="500" > <img src="./image2.png" width="500" >

## Example

```python
from framework.context import Context, hex_col

size = (1920, 1080)

with Context(size) as ctx:
    rect = ctx.rounded_rect((450, 450), (200, 200), (10, 50, 150, 150))
    disc = ctx.disc((250, 250), 200)

    bezier = ctx.bezier((900, 800), (700, 400), (350, 350))

    glyph_sdf = ctx.smooth_min(rect, disc, k=0.025)
    glyph_sdf = ctx.smooth_min(bezier, glyph_sdf, k=0.025)

    layer = ctx.fill(glyph_sdf, hex_col("#e9c46a"), hex_col("#2C2D35"), 0)
    blur = ctx.blur(layer, 10)

    mask_sdf = ctx.rounded_rect((int(size[0] / 2), int(size[1] / 2)), (int(size[0] / 5), int(size[1] / 3)),
                                (150, 150, 150, 150))

    mask_layer = ctx.fill(mask_sdf, (.0, .0, .0, 1.0), (1.0, 1.0, 1.0, 1.0), 0)

    overlay_outline = ctx.outline(mask_sdf, (1.0, 1.0, 1.0, .25), (0.75, 0.75, 0.75, 0.0), inflate=-1.5)

    glass = ctx.fill(mask_sdf, (44 / 255 + 0.15, 45 / 255 + 0.15, 53 / 255 + 0.15, 0.4), (0.0, 0.0, 0.0, 0.0), 0)

    masked = ctx.mask(blur, layer, mask_layer)
    overlay = ctx.overlay(glass, masked)
    overlay = ctx.overlay(overlay_outline, overlay)

    overlay.save("image1.png")
    overlay.show()

with Context((1920, 1080)) as ctx:
    scale = 0.65
    offset_x = 50
    offset_y = 50

    glyph_sdf = ctx.glyph("M", scale, offset_x, offset_y)

    layer = ctx.fill(glyph_sdf, hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), 7.5)
    bg = ctx.fill(glyph_sdf, hex_col("#2C2D35"), hex_col("#2C2D35"), 7.5)

    mask = ctx.fill(glyph_sdf, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), 7.5)
    shadow = ctx.blur(mask, 5)
    with_shadow = ctx.overlay(layer, shadow)

    overlay = ctx.overlay(with_shadow, bg)

    overlay.show()
    overlay.save("image2.png")

```
