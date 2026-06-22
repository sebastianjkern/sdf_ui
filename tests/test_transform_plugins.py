import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sdf_ui import sdf
from sdf_ui.core.plugins.registry import registry


class TransformPluginTests(unittest.TestCase):
    def _fake_renderer(self):
        tex = SimpleNamespace(bind_to_image=lambda *args, **kwargs: None)
        ctx = SimpleNamespace(r32f=lambda: tex)
        renderer = SimpleNamespace(ctx=ctx)
        return renderer, tex

    def _fake_source(self):
        return SimpleNamespace(tex=SimpleNamespace(bind_to_image=lambda *args, **kwargs: None))

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
        renderer, _out_tex = self._fake_renderer()
        source = self._fake_source()
        captured = {}

        def fake_run_shader(ctx, shader_name, *, uniforms=None, image_bindings=None):
            captured["ctx"] = ctx
            captured["shader_name"] = shader_name
            captured["uniforms"] = dict(uniforms or {})
            captured["image_bindings"] = image_bindings

        with patch("sdf_ui.core.operations.run_shader", fake_run_shader):
            result = registry.get("translate").render(renderer, [source], {"offset": (12, 0)})

        self.assertIs(result.tex, renderer.ctx.r32f())
        self.assertEqual(captured["shader_name"], "transform")
        self.assertEqual(captured["uniforms"]["transform_kind"], 0.0)
        self.assertEqual(captured["uniforms"]["offset"], (12, 0))

    def test_scale_binds_expected_uniforms(self):
        renderer, _out_tex = self._fake_renderer()
        source = self._fake_source()
        captured = {}

        def fake_run_shader(ctx, shader_name, *, uniforms=None, image_bindings=None):
            captured["shader_name"] = shader_name
            captured["uniforms"] = dict(uniforms or {})

        with patch("sdf_ui.core.operations.run_shader", fake_run_shader):
            result = registry.get("scale").render(
                renderer,
                [source],
                {"factor": 1.6, "center": (32, 32)},
            )

        self.assertIs(result.context, renderer.ctx)
        self.assertEqual(captured["shader_name"], "transform")
        self.assertEqual(captured["uniforms"]["transform_kind"], 1.0)
        self.assertEqual(captured["uniforms"]["center"], (32, 32))
        self.assertEqual(captured["uniforms"]["factor"], 1.6)

    def test_rotate_binds_expected_uniforms(self):
        renderer, _out_tex = self._fake_renderer()
        source = self._fake_source()
        captured = {}

        def fake_run_shader(ctx, shader_name, *, uniforms=None, image_bindings=None):
            captured["shader_name"] = shader_name
            captured["uniforms"] = dict(uniforms or {})

        with patch("sdf_ui.core.operations.run_shader", fake_run_shader):
            registry.get("rotate").render(
                renderer,
                [source],
                {"angle": 0.75, "center": (32, 32)},
            )

        self.assertEqual(captured["shader_name"], "transform")
        self.assertEqual(captured["uniforms"]["transform_kind"], 2.0)
        self.assertEqual(captured["uniforms"]["center"], (32, 32))
        self.assertEqual(captured["uniforms"]["angle"], 0.75)

    def test_skew_binds_expected_uniforms(self):
        renderer, _out_tex = self._fake_renderer()
        source = self._fake_source()
        captured = {}

        def fake_run_shader(ctx, shader_name, *, uniforms=None, image_bindings=None):
            captured["shader_name"] = shader_name
            captured["uniforms"] = dict(uniforms or {})

        with patch("sdf_ui.core.operations.run_shader", fake_run_shader):
            registry.get("skew").render(
                renderer,
                [source],
                {"skew": (0.5, 0.0), "center": (32, 32)},
            )

        self.assertEqual(captured["shader_name"], "transform")
        self.assertEqual(captured["uniforms"]["transform_kind"], 3.0)
        self.assertEqual(captured["uniforms"]["center"], (32, 32))
        self.assertEqual(captured["uniforms"]["skew"], (0.5, 0.0))

    def test_scale_rejects_a_zero_factor(self):
        renderer, _out_tex = self._fake_renderer()
        source = self._fake_source()

        with self.assertRaises(ValueError):
            registry.get("scale").render(
                renderer,
                [source],
                {"factor": 0, "center": (32, 32)},
            )


if __name__ == "__main__":
    unittest.main()
