import unittest

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class DiamondPluginTests(unittest.TestCase):
    def test_diamond_and_rhombus_are_registered(self):
        registry.ensure_loaded()

        self.assertIn("diamond", registry.public_names())
        self.assertIn("rhombus", registry.public_names())
        self.assertEqual(registry.get("diamond").family, "primitive")

    def test_diamond_renders_a_visible_shape(self):
        with Canvas((96, 96)) as ctx:
            image = sdf.diamond((48, 48), (30, 20)).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        center = pixels[48, 48, 0]
        outside = pixels[8, 8, 0]
        self.assertGreater(center, outside + 50)

    def test_rhombus_matches_diamond(self):
        with Canvas((96, 96)) as ctx:
            diamond = sdf.diamond((48, 48), (30, 20)).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            rhombus = sdf.rhombus((48, 48), (30, 20)).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            diamond_pixels = rgba_array(diamond).copy()
            rhombus_pixels = rgba_array(rhombus).copy()

        self.assertTrue((diamond_pixels == rhombus_pixels).all())


if __name__ == "__main__":
    unittest.main()

