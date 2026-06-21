import unittest

from sdf_ui.core.plugins.loader import PLUGIN_PACKAGES
from sdf_ui.core.plugins.registry import registry


class PluginLoaderTests(unittest.TestCase):
    def test_builtin_plugins_are_loaded_from_family_packages(self):
        registry.ensure_loaded()

        self.assertIn("sdf_ui.core.plugins.primitives", PLUGIN_PACKAGES)
        self.assertIn("sdf_ui.core.plugins.shading", PLUGIN_PACKAGES)
        self.assertIn("sdf_ui.core.plugins.layer", PLUGIN_PACKAGES)
        self.assertIn("sdf_ui.core.plugins.postprocessing", PLUGIN_PACKAGES)
        self.assertEqual(registry.get("circle").family, "primitive")
        self.assertEqual(registry.get("fill").family, "shading")
        self.assertEqual(registry.get("alpha_overlay").family, "layer")
        self.assertEqual(registry.get("blur").family, "postprocessing")


if __name__ == "__main__":
    unittest.main()
