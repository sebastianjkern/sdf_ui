from pathlib import Path

import numpy as np

from sdf_ui import Canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def render(texture, size=(32, 32), params=None, cache=None):
    ctx = Canvas(size)
    return texture.render(ctx, params=params, cache=cache)


def rgba_array(texture):
    width, height = texture.tex.size
    components = getattr(texture.tex, "components", 4)
    pixels = np.frombuffer(texture.tex.read(), dtype=np.uint8).reshape(
        (height, width, components)
    )
    return np.flipud(pixels)


def assert_public_plugin(test_case, name, aliases=()):
    from sdf_ui.gpu.nodes.registry import registry

    registry.ensure_loaded()
    public_names = registry.public_names()
    for public_name in (name, *aliases):
        test_case.assertIn(public_name, public_names)
    test_case.assertIn(name, registry.method_names_for("sdf"))
    test_case.assertEqual(registry.get(name).family, "primitive")


def shader_context(size):
    from types import SimpleNamespace

    from sdf_ui.learning.autodiff import Tensor

    y, x = np.indices((size[1], size[0]), dtype=float)
    return SimpleNamespace(size=size, x=Tensor(x), y=Tensor(y))


def assert_center_brighter(test_case, pixels, center, outside, margin=50):
    test_case.assertGreater(pixels[center[0], center[1], 0], pixels[outside[0], outside[1], 0] + margin)


def tracked_texture_close(texture_node_type):
    closed = []
    original_close = texture_node_type.close

    def close(texture):
        if texture.tex is not None:
            closed.append(texture)
        original_close(texture)

    return closed, close
