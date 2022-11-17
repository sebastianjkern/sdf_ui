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

    union = ctx.smooth_min(rect, disc, k=0.025)
    union = ctx.smooth_min(bezier, union, k=0.025)

    layer = ctx.fill(union, hex_col("#e9c46a"), hex_col("#2C2D35"), 0)
    blur = ctx.blur(layer, 10)

    mask_sdf = ctx.rounded_rect((int(size[0] / 2), int(size[1] / 2)), (int(size[0] / 5), int(size[1] / 3)),
                                (150, 150, 150, 150))

    mask_layer = ctx.fill(mask_sdf, (.0, .0, .0, 1.0), (1.0, 1.0, 1.0, 1.0), 0)

    overlay_outline = ctx.outline(mask_sdf, (1.0, 1.0, 1.0, .25), (0.75, 0.75, 0.75, 0.0))

    masked = ctx.mask(blur, layer, mask_layer)
    overlayed = ctx.overlay(overlay_outline, masked)

    overlayed.save("image1.png")


def collinear(x1, y1, x2, y2, x3, y3):
    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    return area <= 0.000000001


with Context(size) as ctx:
    control_points = ctx.get_glyph("fonts/SFUIDisplay-Bold.ttf", "A")

    scale = 0.45
    offx = 250
    offy = 150

    union = None

    for x, shape in enumerate(control_points):
        for y, stroke in enumerate(shape):
            ax, ay = stroke[0][0] * scale + offx, stroke[0][1] * scale + offy
            bx, by = stroke[1][0] * scale + offx, stroke[1][1] * scale + offy
            cx, cy = stroke[2][0] * scale + offx, stroke[2][1] * scale + offy

            bezier = None

            a = (ax, ay)
            b = (bx, by)
            c = (cx, cy)

            if not collinear(ax, ay, bx, by, cx, cy):
                bezier = ctx.bezier(a, b, c)
            else:
                bezier = ctx.line(a, c)

            if y == 0 and x == 0:
                union = bezier
            else:
                union = ctx.union(union, bezier)

    layer = ctx.fill(union, hex_col("#e9c46a"), hex_col("#2C2D35"), 1.25)
    layer.save("image2.png")

```
