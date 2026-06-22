import unittest

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class NgonPluginTests(unittest.TestCase):
    def test_ngon_is_registered_and_public(self):
        registry.ensure_loaded()

        self.assertIn("ngon", registry.public_names())
        self.assertIn("ngon", registry.method_names_for("sdf"))
        self.assertEqual(registry.get("ngon").family, "primitive")

    def test_ngon_renders_a_visible_regular_polygon(self):
        with Canvas((96, 96)) as ctx:
            image = sdf.ngon((48, 48), 28, 6).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        center = pixels[48, 48, 0]
        outside = pixels[8, 8, 0]

        self.assertGreater(center, outside + 50)

    def test_ngon_rotation_changes_orientation(self):
        with Canvas((96, 96)) as ctx:
            flat = sdf.ngon((48, 48), 28, 4, rotation=0.0).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            rotated = sdf.ngon((48, 48), 28, 4, rotation=0.78539816339).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            flat_pixels = rgba_array(flat).copy()
            rotated_pixels = rgba_array(rotated).copy()

        self.assertNotEqual(
            int(flat_pixels[20, 48, 0]),
            int(rotated_pixels[20, 48, 0]),
        )


if __name__ == "__main__":
    unittest.main()

