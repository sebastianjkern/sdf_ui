__docformat__ = "google"

from importlib import import_module
from pkgutil import walk_packages

PLUGIN_PACKAGES = (
    "sdf_ui.backends.opengl.nodes.primitives",
    "sdf_ui.backends.opengl.nodes.shading",
    "sdf_ui.backends.opengl.nodes.layer",
    "sdf_ui.backends.opengl.nodes.postprocessing",
)


def load_plugins(registry, packages=PLUGIN_PACKAGES):
    loaded_modules = set()
    for package_name in packages:
        package = import_module(package_name)
        modules = sorted(
            walk_packages(package.__path__, f"{package_name}."),
            key=lambda item: item.name,
        )
        for module_info in modules:
            module_name = (
                f"{module_info.name}.plugin" if module_info.ispkg else module_info.name
            )
            if not module_name.endswith(".plugin"):
                continue
            if module_name in loaded_modules:
                continue
            loaded_modules.add(module_name)

            module = import_module(module_name)
            register = getattr(module, "register_plugins", None)
            if register is not None:
                register(registry)
