import unittest

from sdf_ui import Canvas, sdf
from tests.helpers import assert_center_brighter, assert_public_plugin, rgba_array


def _render_convex_polygon(points):
    with Canvas((96, 96)) as ctx:
        image = sdf.convex_polygon(points).fill(
            (255, 255, 255, 255), (0, 0, 0, 255)
        ).render(ctx)
        return rgba_array(image).copy()


class ConvexPolygonPluginTests(unittest.TestCase):
    def test_convex_polygon_is_registered_and_public(self):
        assert_public_plugin(self, "convex_polygon", aliases=())

    def test_convex_polygon_renders_a_visible_shape(self):
        pixels = _render_convex_polygon(((48, 16), (72, 48), (48, 80), (24, 48)))
        assert_center_brighter(self, pixels, (48, 48), (8, 8))

    def test_convex_polygon_accepts_reversed_winding(self):
        pixels = _render_convex_polygon(((24, 48), (48, 80), (72, 48), (48, 16)))
        assert_center_brighter(self, pixels, (48, 48), (8, 8))


if __name__ == "__main__":
    unittest.main()
