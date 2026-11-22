"""Render independent procedural surfaces with GPU light and cross-hatching."""

from math import pi
from pathlib import Path

from PIL import Image, ImageDraw

from sdf_ui import Canvas, paint, param, sdf, surface

_SIZE = (240, 165)
_PAPER = "#f5eee0"
_INK = "#27465a"
_LEFT = (-9 / 10, 0.35, 0.85)
_RIGHT = (0.9, 0.35, 0.85)
_OUTPUT = Path("out/examples/surface_lighting")


def _panel():
    return sdf.rounded_rect((120, 83), (103, 65), 22).subtract(
        sdf.circle((178, 64), 17)
    )


def _height(amplitude):
    return surface.ramp((120, 83), (0.06, 0.03)).add(
        surface.gaussian((param("ridge_x"), 83), (21, 55), amplitude)
    )


def _cross_hatch(shape, height, direction):
    hatch = paint.HatchStyle(
        (
            paint.HatchPass(pi / 4, 8, 1.0, min_shadow=0.18),
            paint.HatchPass(-pi / 4, 8, 1.0, min_shadow=0.4),
            paint.HatchPass(pi / 2, 7, 0.9, min_shadow=0.63),
        ),
        light=paint.HatchLight(direction=direction, surface=height),
    )
    return paint.scene(
        paint.fill(shape, "#dfcfb5"),
        paint.fill(shape, _INK, hatch=hatch),
        paint.outline(shape, _INK, width=1.25),
        background=_PAPER,
    )


def _gpu_tile(canvas, shape, height, direction):
    x, y, z = direction
    gpu_direction = (x, -y, z)
    output = (
        shape.light(_INK, bg_color=_PAPER, light_dir=gpu_direction)
        if height is None
        else shape.light_surface(height, _INK, light_dir=gpu_direction)
    )
    with canvas.render(output, {"ridge_x": 105.0}) as rendered:
        tile = Image.fromarray(rendered.read()[::-1].copy())
    background = Image.new("RGBA", _SIZE, _PAPER)
    background.alpha_composite(tile)
    return background


def _hatch_tile(canvas, scene, scratch, ridge_x):
    scene.save_png(scratch, canvas=canvas, params={"ridge_x": ridge_x})
    with Image.open(scratch) as rendered:
        return rendered.copy()


def _scenarios():
    panel = _panel()
    raised = _height(22)
    recessed = _height(-22)
    flat = surface.ramp((0, 0), (0, 0))
    oval = sdf.ellipse((120, 83), (103, 65)).subtract(sdf.circle((178, 64), 17))
    return (
        ("SDF-derived relief", panel, None, _LEFT),
        ("Flat height field", panel, flat, _LEFT),
        ("Raised ridge", panel, raised, _LEFT),
        ("Raised ridge / right light", panel, raised, _RIGHT),
        ("Recessed ridge", panel, recessed, _LEFT),
        ("Same ridge / oval", oval, raised, _LEFT),
    )


def _comparison(canvas, output, scratch):
    cases = _scenarios()
    w, h = _SIZE
    sheet = Image.new("RGB", (len(cases) * w, 2 * h + 89), _PAPER)
    label = ImageDraw.Draw(sheet)
    label.text((16, 11), "Independent relief on the same cutout panel", fill=_INK)
    label.text((16, 31), "GPU lighting", fill=_INK)
    label.text((16, h + 67), "Cross-hatch lighting", fill=_INK)
    for index, (name, shape, height, direction) in enumerate(cases):
        x = index * w
        label.text((x + 16, 49), name, fill=_INK)
        sheet.paste(_gpu_tile(canvas, shape, height, direction), (x, 63))
        hatch = _cross_hatch(shape, height, direction)
        sheet.paste(_hatch_tile(canvas, hatch, scratch, 105.0), (x, h + 86))
    sheet.save(output / "comparison.png")


def _motion(canvas, output, scratch):
    scene = _cross_hatch(_panel(), _height(22), _LEFT)
    centers = list(range(67, 156, 7))
    frames = [
        _hatch_tile(canvas, scene, scratch, float(center)).convert("RGB")
        for center in [*centers, *centers[-2:0:-1]]
    ]
    frames[0].save(
        output / "moving_ridge.gif",
        save_all=True,
        append_images=frames[1:],
        duration=95,
        loop=0,
    )


def surface_lighting_example(output=_OUTPUT):
    from tempfile import TemporaryDirectory

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary, Canvas(_SIZE) as canvas:
        scratch = Path(temporary) / "hatch.png"
        _comparison(canvas, output, scratch)
        _motion(canvas, output, scratch)
    return output


if __name__ == "__main__":
    surface_lighting_example()
