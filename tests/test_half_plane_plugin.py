import unittest

from sdf_ui import Canvas, sdf
from tests.helpers import assert_public_plugin, rgba_array


class HalfPlanePluginTests(unittest.TestCase):
    def test_half_plane_is_registered_and_public(self):
        assert_public_plugin(self, "half_plane", aliases=())

    def test_half_plane_renders_two_distinct_halves(self):
        with Canvas((64, 64)) as ctx:
            image = (
                sdf.half_plane((32, 32), (1, 0))
                .fill(
                    (255, 255, 255, 255),
                    (0, 0, 0, 255),
                )
                .render(ctx)
            )
            pixels = rgba_array(image).copy()

        left_mean = pixels[:, :24, 0].mean()
        right_mean = pixels[:, 40:, 0].mean()

        self.assertGreater(left_mean, right_mean + 50)

    def test_half_plane_accepts_arbitrary_normal_direction(self):
        with Canvas((32, 32)) as ctx:
            image = (
                sdf.half_plane((16, 16), (0, -1))
                .fill(
                    (255, 255, 255, 255),
                    (0, 0, 0, 255),
                )
                .render(ctx)
            )
            pixels = rgba_array(image).copy()

        top_mean = pixels[:12, :, 0].mean()
        bottom_mean = pixels[20:, :, 0].mean()

        self.assertGreater(top_mean, bottom_mean + 50)


if __name__ == "__main__":
    unittest.main()
