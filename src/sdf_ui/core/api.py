__docformat__ = "google"

from .plugins.base import TextureKind
from .plugins.common import build


class _OperationNamespace:
    kind = None

    def __getattr__(self, name):
        from .plugins.registry import registry

        try:
            plugin = registry.get(name)
        except KeyError as exc:
            raise AttributeError(f"No {self.kind} factory named '{name}'") from exc
        if not plugin.public or plugin.result != self.kind or plugin.input_kinds:
            raise AttributeError(f"No {self.kind} factory named '{name}'")

        def factory(*args, **kwargs):
            return build(name, *args, **kwargs)

        return factory

    def __dir__(self):
        from .plugins.registry import registry

        names = set(super().__dir__())
        names.update(
            name
            for name in registry.public_names()
            if registry.get(name).result == self.kind
        )
        return sorted(names)


class SDFNamespace(_OperationNamespace):
    kind = TextureKind.SDF


class ColorNamespace(_OperationNamespace):
    kind = TextureKind.COLOR
