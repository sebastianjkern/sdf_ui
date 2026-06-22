import unittest

import numpy as np

from sdf_ui import Canvas, sdf
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class ParallelogramPluginTests(unittest.TestCase):
    def test_parallelogram_is_registered_and_public(self):
        registry.ensure_loaded()

        self.assertIn("parallelogram", registry.public_names())
        self.assertIn("parallelogram", registry.method_names_for("sdf"))
        self.assertEqual(registry.get("parallelogram").family, "primitive")

    def test_parallelogram_renders_a_visible_shape(self):
        with Canvas((96, 96)) as ctx:
            image = sdf.parallelogram((48, 48), (48, 28), 12).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            pixels = rgba_array(image).copy()

        center = pixels[48, 48, 0]
        outside = pixels[8, 8, 0]

        self.assertGreater(center, outside + 50)

    def test_skew_changes_orientation(self):
        with Canvas((96, 96)) as ctx:
            flat = sdf.parallelogram((48, 48), (48, 28), 0).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            skewed = sdf.parallelogram((48, 48), (48, 28), 12).fill(
                (255, 255, 255, 255),
                (0, 0, 0, 255),
            ).render(ctx)
            flat_pixels = rgba_array(flat).copy()
            skewed_pixels = rgba_array(skewed).copy()

        self.assertGreater(np.count_nonzero(flat_pixels != skewed_pixels), 0)


if __name__ == "__main__":
    unittest.main()
