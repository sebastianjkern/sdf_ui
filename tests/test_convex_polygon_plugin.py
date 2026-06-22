import unittest

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class ConvexPolygonPluginTests(unittest.TestCase):
    def test_convex_polygon_is_registered_and_public(self):
        registry.ensure_loaded()

        self.assertIn("convex_polygon", registry.public_names())
        self.assertIn("convex_polygon", registry.method_names_for("sdf"))
        self.assertEqual(registry.get("convex_polygon").family, "primitive")

    def test_convex_polygon_renders_a_visible_shape(self):
        points = ((48, 16), (72, 48), (48, 80), (24, 48))
        with Canvas((96, 96)) as ctx:
            image = sdf.convex_polygon(points).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        center = pixels[48, 48, 0]
        corner = pixels[8, 8, 0]
        self.assertGreater(center, corner + 50)

    def test_convex_polygon_accepts_reversed_winding(self):
        points = ((24, 48), (48, 80), (72, 48), (48, 16))
        with Canvas((96, 96)) as ctx:
            image = sdf.convex_polygon(points).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        center = pixels[48, 48, 0]
        corner = pixels[8, 8, 0]
        self.assertGreater(center, corner + 50)


if __name__ == "__main__":
    unittest.main()

