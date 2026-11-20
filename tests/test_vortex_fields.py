import numpy as np

from sdf_ui import Canvas, color, sdf
from sdf_ui.core.color import ColorTexture
from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes.shading.vortex_streamlines.plugin import render_streamlines


class FakeTexture:
    def __init__(self, data=None):
        self.data = data

    def read(self):
        return self.data

    def write(self, data):
        self.data = data


class FakeContext:
    size = (41, 41)

    def r32f(self):
        return FakeTexture()

    def rgba8(self):
        return FakeTexture()


class FakeRenderer:
    def __init__(self):
        self.ctx = FakeContext()


def test_vortex_warp_changes_the_input_sdf_and_returns_an_sdf_texture():
    circle = sdf.circle((20, 20), 10)
    warped = circle.vortex_warp(((20, 20, 1),), radius=10, strength=0.5)
    with (
        Canvas((41, 41)) as canvas,
        canvas.render(circle) as original,
        canvas.render(warped) as result,
    ):
        source = original.read()
        field = result.read()
        assert isinstance(result, SDFTexture)

    assert field[20, 30] > source[20, 30]
    assert field[20, 20] == source[20, 20]


def test_vortex_streamlines_render_an_rgba_overlay():
    renderer = FakeRenderer()
    renderer.ctx.size = (48, 48)
    result = render_streamlines(
        renderer,
        (),
        {
            "vortices": ((24, 24, 1),),
            "radius": 20,
            "spacing": 16,
            "step_size": 2,
            "steps": 8,
            "color": (0.4, 1.0, 0.9, 0.8),
            "glow_color": (0.1, 0.8, 0.8, 0.4),
        },
    )

    assert isinstance(result, ColorTexture)
    pixels = np.frombuffer(result.tex.data, dtype=np.uint8).reshape(48, 48, 4)
    assert np.any(pixels[..., 3] > 0)


def test_vortex_operations_are_available_in_the_public_graph_api():
    field = sdf.circle((40, 40), 25).vortex_warp(((40, 40, 1),), radius=20)
    overlay = color.vortex_streamlines(((40, 40, 1),), radius=20)

    assert isinstance(field, SDFTexture)
    assert isinstance(overlay, ColorTexture)
