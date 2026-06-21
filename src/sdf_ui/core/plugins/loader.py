__docformat__ = "google"

from importlib import import_module
from importlib.metadata import entry_points
from pkgutil import walk_packages


PLUGIN_PACKAGES = (
    "sdf_ui.core.plugins.primitives",
    "sdf_ui.core.plugins.shading",
    "sdf_ui.core.plugins.layer",
    "sdf_ui.core.plugins.postprocessing",
)


def load_plugins(registry, packages=PLUGIN_PACKAGES):
    loaded_modules = set()
    for package_name in packages:
        package = import_module(package_name)
        modules = sorted(walk_packages(package.__path__, f"{package_name}."), key=lambda item: item.name)
        for module_info in modules:
            module_name = f"{module_info.name}.plugin" if module_info.ispkg else module_info.name
            if not module_name.endswith(".plugin"):
                continue
            if module_name in loaded_modules:
                continue
            loaded_modules.add(module_name)

            module = import_module(module_name)
            register = getattr(module, "register_plugins", None)
            if register is not None:
                register(registry)

    discovered = entry_points()
    plugin_entries = discovered.select(group="sdf_ui.core.plugins") if hasattr(discovered, "select") else discovered.get("sdf_ui.core.plugins", ())
    for entry_point in plugin_entries:
        plugin = entry_point.load()
        register = getattr(plugin, "register_plugins", plugin if callable(plugin) else None)
        if register is not None:
            register(registry)
