import numpy as np
import pytest

from sdf_ui import sdf
from sdf_ui.core.color import ColorTexture
from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes.primitives.voronoi.plugin import render_voronoi
from sdf_ui.gpu.nodes.shading.field_debug.plugin import render_field_debug


class FakeTexture:
    def __init__(self, shape=None):
        self.shape = shape
        self.data = None

    def write(self, data):
        self.data = data

    def read(self):
        return np.asarray(self.data)


class FakeContext:
    size = (5, 3)

    def r32f(self):
        return FakeTexture()

    def rgba8(self):
        return FakeTexture()


class FakeRenderer:
    ctx = FakeContext()


class FakeNode:
    def __init__(self, texture):
        self.tex = texture


def test_voronoi_cell_field_is_signed_distance_to_bisector():
    result = render_voronoi(
        FakeRenderer(), (), {"points": ((1, 1), (3, 1)), "cell": 0}
    )
    field = np.frombuffer(result.tex.data, dtype=np.float32).reshape(3, 5)

    assert field[1, 1] == pytest.approx(-1.0)
    assert field[1, 2] == pytest.approx(0.0)
    assert field[1, 3] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("points", "cell", "message"),
    [
        (((1, 1),), 0, "at least two"),
        (((1, 1), (1, 1)), 0, "distinct"),
        (((1, 1), (3, 1)), 2, "cell must"),
        (((1, 1), (np.inf, 1)), 0, "finite"),
    ],
)
def test_voronoi_rejects_invalid_sites(points, cell, message):
    with pytest.raises(ValueError, match=message):
        render_voronoi(FakeRenderer(), (), {"points": points, "cell": cell})


def test_field_debug_maps_signed_distance_and_clamps_range():
    source = FakeTexture()
    source.data = np.array([[-32.0, 0.0, 32.0]], dtype=np.float32)
    renderer = FakeRenderer()
    renderer.ctx.size = (3, 1)

    result = render_field_debug(renderer, (FakeNode(source),), {"range": 16})
    pixels = np.frombuffer(result.tex.data, dtype=np.uint8).reshape(1, 3, 4)

    assert tuple(pixels[0, 0, :3]) == (31, 64, 173)
    assert tuple(pixels[0, 1, :3]) == (240, 240, 209)
    assert tuple(pixels[0, 2, :3]) == (166, 46, 31)
    assert np.all(pixels[..., 3] == 255)


def test_field_debug_requires_a_positive_finite_range():
    source = FakeTexture()
    source.data = np.array([[0.0]], dtype=np.float32)
    renderer = FakeRenderer()
    renderer.ctx.size = (1, 1)

    with pytest.raises(ValueError, match="positive finite"):
        render_field_debug(renderer, (FakeNode(source),), {"range": 0})


def test_voronoi_and_field_debug_are_exposed_in_the_public_graph_api():
    cell = sdf.voronoi(((1, 1), (3, 1)), cell=0)

    assert isinstance(cell, SDFTexture)
    assert isinstance(cell.debug_distance(), ColorTexture)
    assert isinstance(cell.debug_seams(), ColorTexture)
    assert isinstance(cell.debug_boolean(sdf.circle((2, 1), 1)), ColorTexture)

    with pytest.raises(ValueError, match="Unknown boolean operation"):
        cell.debug_boolean(sdf.circle((2, 1), 1), operation="xor")
