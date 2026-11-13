"""Dynamic namespaces that expose plugin-backed factory functions."""

__docformat__ = "google"

from ..gpu.nodes.base import TextureKind
from ..gpu.nodes.common import build
from ._suggestions import unknown_name_message


class _OperationNamespace:
    kind = None

    def __getattr__(self, name):
        from ..gpu.nodes.registry import registry

        try:
            plugin = registry.get(name)
        except KeyError as exc:
            raise AttributeError(self._unknown_factory_message(name)) from exc
        if not plugin.public or plugin.result != self.kind or plugin.input_kinds:
            raise AttributeError(self._unknown_factory_message(name))

        def factory(*args, **kwargs):
            return build(name, *args, **kwargs)

        factory.__name__ = name
        factory.__qualname__ = f"{type(self).__name__}.{name}"
        factory.__doc__ = plugin.docstring()
        factory.__signature__ = plugin.signature()
        return factory

    def _unknown_factory_message(self, name):
        from ..gpu.nodes.registry import registry

        names = registry.factory_names_for(self.kind)
        return unknown_name_message(f"No {self.kind} factory named", name, names)

    def __dir__(self):
        from ..gpu.nodes.registry import registry

        names = set(super().__dir__())
        names.update(registry.factory_names_for(self.kind))
        return sorted(names)


class SDFNamespace(_OperationNamespace):
    kind = TextureKind.SDF


class ColorNamespace(_OperationNamespace):
    kind = TextureKind.COLOR
