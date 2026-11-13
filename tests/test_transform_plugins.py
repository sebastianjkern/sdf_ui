import unittest
from types import SimpleNamespace

from sdf_ui import sdf
from sdf_ui.gpu.nodes.registry import registry


class TransformPluginTests(unittest.TestCase):
    def _render(self, name, params):
        captured = {}
        output = SimpleNamespace(close=lambda: None)

        class FakeContext:
            def r32f(self):
                return output

            def run_node_shader(self, descriptor, dest, inputs, uniforms, **kwargs):
                captured.update(
                    descriptor=descriptor,
                    output=dest,
                    inputs=inputs,
                    uniforms=uniforms,
                )

        source = SimpleNamespace(tex=SimpleNamespace())
        renderer = SimpleNamespace(ctx=FakeContext())
        result = registry.get(name).render(renderer, [source], params)
        self.assertIs(result.tex, output)
        self.assertIs(captured["output"], output)
        self.assertEqual(captured["descriptor"].name, "transform")
        self.assertEqual(captured["inputs"], (source.tex,))
        return captured["uniforms"]

    def test_transform_plugins_are_registered_and_public(self):
        registry.ensure_loaded()
        for name in ("translate", "scale", "rotate", "skew"):
            with self.subTest(name=name):
                self.assertIn(name, registry.method_names_for("sdf"))
                self.assertEqual(registry.get(name).family, "primitive")

    def test_sdf_nodes_expose_transform_methods(self):
        node = sdf.circle((32, 32), 10)
        for name in ("translate", "scale", "rotate", "skew"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(node, name)))

    def test_translate_binds_expected_uniforms(self):
        uniforms = self._render("translate", {"offset": (12, 0)})
        self.assertEqual(uniforms["transform_kind"], 0.0)
        self.assertEqual(uniforms["offset"], (12, 0))

    def test_scale_binds_expected_uniforms(self):
        uniforms = self._render("scale", {"factor": 1.6, "center": (32, 32)})
        self.assertEqual(uniforms["transform_kind"], 1.0)
        self.assertEqual(uniforms["center"], (32, 32))
        self.assertEqual(uniforms["factor"], 1.6)

    def test_rotate_binds_expected_uniforms(self):
        uniforms = self._render("rotate", {"angle": 0.75, "center": (32, 32)})
        self.assertEqual(uniforms["transform_kind"], 2.0)
        self.assertEqual(uniforms["center"], (32, 32))
        self.assertEqual(uniforms["angle"], 0.75)

    def test_skew_binds_expected_uniforms(self):
        uniforms = self._render("skew", {"skew": (0.5, 0.0), "center": (32, 32)})
        self.assertEqual(uniforms["transform_kind"], 3.0)
        self.assertEqual(uniforms["center"], (32, 32))
        self.assertEqual(uniforms["skew"], (0.5, 0.0))

    def test_scale_rejects_a_zero_factor(self):
        with self.assertRaises(ValueError):
            registry.get("scale").render(
                SimpleNamespace(ctx=SimpleNamespace()),
                [SimpleNamespace(tex=SimpleNamespace())],
                {"factor": 0, "center": (32, 32)},
            )


if __name__ == "__main__":
    unittest.main()
