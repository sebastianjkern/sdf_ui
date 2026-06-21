__docformat__ = "google"

from pathlib import Path

from ..log import logger
from .plugins.registry import registry


class ShaderLibrary:
    """
    Loads, validates, compiles, and caches compute shaders for a ModernGL context.
    """

    def __init__(self, mgl_context, base_path=None):
        self._mgl_ctx = mgl_context
        self._base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self._cache = {}

    def get(self, shader_name: str):
        shader_files = {
            descriptor.name: descriptor for descriptor in registry.shader_files()
        }
        if shader_name not in shader_files:
            known = ", ".join(sorted(shader_files))
            raise KeyError(f"Unknown shader '{shader_name}'. Known shaders: {known}")

        if shader_name not in self._cache:
            descriptor = shader_files[shader_name]
            path = self._base_path / descriptor.path

            if not path.exists():
                raise FileNotFoundError(
                    f"Shader '{shader_name}' is registered at '{path}', but no file exists there."
                )

            code = path.read_text(encoding="utf-8")
            self._cache[shader_name] = self._mgl_ctx.compute_shader(code)
            logger().debug(f"Compiled and cached {shader_name} shader...")

        return self._cache[shader_name]
