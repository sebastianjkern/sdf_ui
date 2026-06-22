import unittest

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class CapsulePluginTests(unittest.TestCase):
    def test_capsule_and_segment_are_registered(self):
        registry.ensure_loaded()

        self.assertIn("capsule", registry.public_names())
        self.assertIn("segment", registry.public_names())
        self.assertEqual(registry.get("capsule").family, "primitive")
        self.assertEqual(registry.get("segment").family, "primitive")

    def test_capsule_renders_a_thick_segment(self):
        with Canvas((96, 64)) as ctx:
            image = sdf.capsule((18, 32), (78, 32), 10).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        center_row = pixels[32, :, 0]
        edge_row = pixels[8, :, 0]

        self.assertGreater(center_row.mean(), edge_row.mean() + 50)
        self.assertGreater(center_row[48], 200)
        self.assertLess(edge_row[48], 40)

    def test_segment_alias_matches_capsule(self):
        with Canvas((96, 64)) as ctx:
            capsule = sdf.capsule((18, 32), (78, 32), 10).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            segment = sdf.segment((18, 32), (78, 32), 10).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            capsule_pixels = rgba_array(capsule).copy()
            segment_pixels = rgba_array(segment).copy()

        self.assertTrue((capsule_pixels == segment_pixels).all())


if __name__ == "__main__":
    unittest.main()

