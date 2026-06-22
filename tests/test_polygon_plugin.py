import unittest

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class PolygonPluginTests(unittest.TestCase):
    def test_polygon_is_registered_and_public(self):
        registry.ensure_loaded()

        self.assertIn("polygon", registry.public_names())
        self.assertIn("polygon", registry.method_names_for("sdf"))
        self.assertEqual(registry.get("polygon").family, "primitive")

    def test_polygon_renders_a_visible_concave_shape(self):
        points = ((20, 20), (76, 20), (76, 36), (36, 36), (36, 60), (76, 60), (76, 76), (20, 76))
        with Canvas((96, 96)) as ctx:
            image = sdf.polygon(points).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        inside = pixels[48, 28, 0]
        notch = pixels[48, 56, 0]
        outside = pixels[8, 8, 0]

        self.assertGreater(inside, outside + 50)
        self.assertLess(notch, inside - 50)

    def test_polygon_accepts_reversed_winding(self):
        points = ((20, 76), (76, 76), (76, 60), (36, 60), (36, 36), (76, 36), (76, 20), (20, 20))
        with Canvas((96, 96)) as ctx:
            image = sdf.polygon(points).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        inside = pixels[48, 28, 0]
        notch = pixels[48, 56, 0]

        self.assertGreater(inside, notch + 50)


if __name__ == "__main__":
    unittest.main()

