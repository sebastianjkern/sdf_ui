__docformat__ = "google"

from .base import Plugin, TextureKind


class PluginRegistry:
    def __init__(self):
        self._plugins = {}
        self._loaded = False

    def register(self, plugin: Plugin):
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' is already registered")
        self._plugins[plugin.name] = plugin
        return plugin

    def get(self, name: str) -> Plugin:
        self.ensure_loaded()
        try:
            return self._plugins[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._plugins))
            raise KeyError(
                f"Unknown texture plugin '{name}'. Known plugins: {known}"
            ) from exc

    def build(self, name: str, *args, **kwargs):
        plugin = self.get(name)
        inputs, params = plugin.bind(args, kwargs)
        self._check_inputs(plugin, inputs)
        return self._node(plugin, inputs, params)

    def method_names_for(self, kind: str):
        self.ensure_loaded()
        return tuple(
            name for name, plugin in self._plugins.items() if kind in plugin.method_of
        )

    def public_names(self):
        self.ensure_loaded()
        return tuple(name for name, plugin in self._plugins.items() if plugin.public)

    def shader_files(self):
        self.ensure_loaded()
        descriptors = []
        for plugin in self._plugins.values():
            if plugin.shader is not None:
                descriptors.append(plugin.shader)
            descriptors.extend(plugin.extra_shaders)
        return tuple(descriptors)

    def ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        from .loader import load_plugins

        load_plugins(self)

    def _check_inputs(self, plugin, inputs):
        for index, (expected, actual) in enumerate(zip(plugin.input_kinds, inputs)):
            if not hasattr(actual, "kind"):
                raise TypeError(
                    f"{plugin.name} input {index} should be a texture, got {type(actual)}"
                )
            if expected != TextureKind.MULTI_OUTPUT and actual.kind != expected:
                raise TypeError(
                    f"{plugin.name} input {index} should be {expected}, got {actual.kind}"
                )

    def _node(self, plugin, inputs, params):
        from sdf_ui.core.color import ColorTexture
        from sdf_ui.core.sdf import SDFTexture

        if plugin.result == TextureKind.SDF:
            return SDFTexture(op=plugin.name, inputs=inputs, params=params)
        if plugin.result == TextureKind.COLOR:
            return ColorTexture(
                op=plugin.name, inputs=inputs, params=params, mode=plugin.mode
            )
        if plugin.result == TextureKind.MULTI_OUTPUT:
            from sdf_ui.core.texture import MultiOutputResult
            from sdf_ui.core.texture import texture_params

            return MultiOutputResult(
                op=plugin.name, inputs=inputs, params=texture_params(**params)
            )
        raise ValueError(
            f"Plugin '{plugin.name}' has unknown result kind '{plugin.result}'"
        )


registry = PluginRegistry()
