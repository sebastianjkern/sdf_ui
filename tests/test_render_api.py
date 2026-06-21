import unittest

from sdf_ui import Canvas, ColorTexture, SDFTexture, color, sdf


class RenderApiTests(unittest.TestCase):
    def test_root_exposes_texture_namespaces(self):
        import sdf_ui

        self.assertTrue(hasattr(sdf_ui, "sdf"))
        self.assertTrue(hasattr(sdf_ui, "color"))
        self.assertFalse(hasattr(sdf_ui, "disc"))

    def test_factory_builds_render_nodes(self):
        scene = sdf.circle((1, 2), 3)

        self.assertIsInstance(scene, SDFTexture)
        self.assertEqual(scene.op, "circle")

    def test_factories_and_texture_methods_are_dynamic(self):
        self.assertTrue(callable(getattr(sdf, "circle")))
        self.assertTrue(callable(getattr(color, "clear")))

        sdf_node = sdf.circle((1, 2), 3)
        color_node = color.clear("#fff")

        self.assertTrue(callable(getattr(sdf_node, "fill")))
        self.assertTrue(callable(getattr(color_node, "blur")))
        self.assertTrue(callable(getattr(color_node.post, "invert")))

    def test_color_scene_renders(self):
        with Canvas((33, 33)) as ctx:
            image = color.clear("#fff").render(ctx)

        self.assertIsInstance(image, ColorTexture)
        self.assertEqual(image.tex.size, (33, 33))

    def test_masked_union_result_renders_both_outputs(self):
        result = sdf.circle((16, 16), 10).masked_union(sdf.circle((20, 16), 10))
        sdf_result, mask_result = result

        with Canvas((32, 32)) as ctx:
            rendered_sdf = sdf_result.render(ctx)
            rendered_mask = mask_result.render(ctx)

        self.assertEqual(rendered_sdf.tex.size, (32, 32))
        self.assertEqual(rendered_mask.tex.size, (32, 32))


if __name__ == "__main__":
    unittest.main()
