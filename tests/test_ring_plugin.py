import unittest

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class RingPluginTests(unittest.TestCase):
    def test_ring_and_annulus_are_registered(self):
        registry.ensure_loaded()

        self.assertIn("ring", registry.public_names())
        self.assertIn("annulus", registry.public_names())
        self.assertEqual(registry.get("ring").family, "primitive")

    def test_ring_renders_a_visible_band(self):
        with Canvas((96, 96)) as ctx:
            image = sdf.ring((48, 48), 28, 10).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        center = pixels[48, 48, 0]
        band = pixels[48, 76, 0]
        outside = pixels[8, 8, 0]

        self.assertLess(center, band)
        self.assertGreater(band, outside + 50)

    def test_annulus_matches_ring(self):
        with Canvas((96, 96)) as ctx:
            ring = sdf.ring((48, 48), 28, 10).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            annulus = sdf.annulus((48, 48), 28, 10).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            ring_pixels = rgba_array(ring).copy()
            annulus_pixels = rgba_array(annulus).copy()

        self.assertTrue((ring_pixels == annulus_pixels).all())


if __name__ == "__main__":
    unittest.main()

