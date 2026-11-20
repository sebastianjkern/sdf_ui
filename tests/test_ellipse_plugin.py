import unittest

from sdf_ui import Canvas, sdf
from tests.helpers import assert_public_plugin, rgba_array


class EllipsePluginTests(unittest.TestCase):
    def test_ellipse_is_registered_and_public(self):
        assert_public_plugin(self, "ellipse", aliases=())

    def test_ellipse_renders_a_visible_shape(self):
        with Canvas((96, 96)) as ctx:
            image = (
                sdf.ellipse((48, 48), (28, 16))
                .fill(
                    (255, 255, 255, 255),
                    (0, 0, 0, 255),
                )
                .render(ctx)
            )
            pixels = rgba_array(image).copy()

        center = pixels[48, 48, 0]
        outside = pixels[12, 12, 0]
        edge = pixels[48, 72, 0]

        self.assertGreater(center, outside + 50)
        self.assertGreater(edge, outside + 50)

    def test_ellipse_is_wider_than_it_is_tall(self):
        with Canvas((96, 96)) as ctx:
            image = (
                sdf.ellipse((48, 48), (32, 12))
                .fill(
                    (255, 255, 255, 255),
                    (0, 0, 0, 255),
                )
                .render(ctx)
            )
            pixels = rgba_array(image).copy()

        horizontal_span = pixels[48, :, 0] > 100
        vertical_span = pixels[:, 48, 0] > 100

        self.assertGreater(horizontal_span.sum(), vertical_span.sum())


if __name__ == "__main__":
    unittest.main()
