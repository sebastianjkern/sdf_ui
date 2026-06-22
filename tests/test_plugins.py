import unittest

import numpy as np

from sdf_ui import Canvas, sdf
from sdf_ui.core.sdf import SDFTexture
from sdf_ui.core.plugins.loader import PLUGIN_PACKAGES
from sdf_ui.core.plugins.registry import registry
from tests.helpers import rgba_array


class PluginLoaderTests(unittest.TestCase):
    def test_builtin_plugins_are_loaded_from_family_packages(self):
        registry.ensure_loaded()

        self.assertIn("sdf_ui.core.plugins.primitives", PLUGIN_PACKAGES)
        self.assertIn("sdf_ui.core.plugins.shading", PLUGIN_PACKAGES)
        self.assertIn("sdf_ui.core.plugins.layer", PLUGIN_PACKAGES)
        self.assertIn("sdf_ui.core.plugins.postprocessing", PLUGIN_PACKAGES)
        self.assertEqual(registry.get("circle").family, "primitive")
        self.assertEqual(registry.get("fill").family, "shading")
        self.assertEqual(registry.get("isolines").family, "shading")
        self.assertEqual(registry.get("alpha_overlay").family, "layer")
        self.assertEqual(registry.get("blur").family, "postprocessing")

    def test_isolines_is_available_as_an_sdf_method(self):
        registry.ensure_loaded()

        self.assertIn("isolines", registry.method_names_for("sdf"))
        with Canvas((48, 48)) as ctx:
            field = np.tile(np.arange(48, dtype=np.float32), (48, 1))
            tex = ctx.r32f()
            tex.write(field.tobytes())
            image = SDFTexture(tex=tex, context=ctx).isolines(
                (255, 255, 255, 255),
                (0, 0, 0, 0),
                spacing=4.0,
                line_width=1.0,
            ).render(ctx)

        alpha = rgba_array(image)[..., 3]
        self.assertGreater(np.count_nonzero(alpha), 0)
        self.assertGreater(int(alpha.max()), 0)


if __name__ == "__main__":
    unittest.main()
