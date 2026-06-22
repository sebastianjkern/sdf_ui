import unittest

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class ArcPluginTests(unittest.TestCase):
    def test_arc_is_registered_and_public(self):
        registry.ensure_loaded()

        self.assertIn("arc", registry.public_names())
        self.assertIn("arc", registry.method_names_for("sdf"))
        self.assertEqual(registry.get("arc").family, "primitive")

    def test_arc_renders_a_visible_curve(self):
        with Canvas((96, 96)) as ctx:
            image = sdf.arc((48, 48), 28, 0.0, 3.1415926535).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
                2,
            ).render(ctx)
            pixels = rgba_array(image).copy()

        center = pixels[48, 48, 0]
        curve = pixels[48, 20, 0]
        off_curve = pixels[68, 68, 0]

        self.assertLess(center, curve)
        self.assertGreater(curve, off_curve)

    def test_arc_handles_wrapped_angle_ranges(self):
        with Canvas((96, 96)) as ctx:
            image = sdf.arc((48, 48), 28, 5.5, 0.8).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
                2,
            ).render(ctx)
            pixels = rgba_array(image).copy()

        right_edge = pixels[48, 76, 0]
        left_edge = pixels[48, 20, 0]

        self.assertGreater(right_edge, left_edge)


if __name__ == "__main__":
    unittest.main()
