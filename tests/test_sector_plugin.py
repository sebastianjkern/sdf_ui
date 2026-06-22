import unittest

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class SectorPluginTests(unittest.TestCase):
    def test_sector_wedge_and_pie_are_registered(self):
        registry.ensure_loaded()

        self.assertIn("sector", registry.public_names())
        self.assertIn("wedge", registry.public_names())
        self.assertIn("pie", registry.public_names())
        self.assertEqual(registry.get("sector").family, "primitive")

    def test_sector_renders_a_visible_slice(self):
        with Canvas((96, 96)) as ctx:
            image = sdf.sector((48, 48), 28, 0.0, 1.57079632679).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        inside = pixels[32, 64, 0]
        outside = pixels[64, 32, 0]

        self.assertGreater(inside, outside + 50)

    def test_pie_matches_sector(self):
        with Canvas((96, 96)) as ctx:
            sector = sdf.sector((48, 48), 28, 0.0, 1.57079632679).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pie = sdf.pie((48, 48), 28, 0.0, 1.57079632679).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            sector_pixels = rgba_array(sector).copy()
            pie_pixels = rgba_array(pie).copy()

        self.assertTrue((sector_pixels == pie_pixels).all())


if __name__ == "__main__":
    unittest.main()

