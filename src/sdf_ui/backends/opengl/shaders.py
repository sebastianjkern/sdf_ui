__docformat__ = "google"

from pathlib import Path

from ...log import logger
from .nodes.registry import registry
from .shader_operation import ShaderOperation


class ShaderLibrary:
    """
    Loads, validates, compiles, and caches compute shaders for a ModernGL context.
    """

    def __init__(self, mgl_context, base_path=None, sdf_image_format="r32f"):
        self._mgl_ctx = mgl_context
        self._base_path = Path(base_path) if base_path else Path(__file__).resolve().parents[2]
        self._sdf_image_format = sdf_image_format
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

            operation = getattr(descriptor, "operation", None)
            if operation is None:
                operation = ShaderOperation(descriptor.name, descriptor.path)
            code = operation.forward_source(sdf_format=self._sdf_image_format, base_path=self._base_path)
            self._cache[shader_name] = self._mgl_ctx.compute_shader(code)
            logger().debug(f"Compiled and cached {shader_name} shader...")

        return self._cache[shader_name]

    def differentiation(self, shader_name: str):
        """Return the named pass's executable derivative contract without GL compilation."""
        descriptors = {descriptor.name: descriptor for descriptor in registry.shader_files()}
        try:
            return descriptors[shader_name].differentiation
        except KeyError as error:
            raise KeyError(f"Unknown shader '{shader_name}'") from error
