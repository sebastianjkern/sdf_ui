from pathlib import Path
import sys

from setuptools import Command, setup as setuptools_setup

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def api_stub_manual_method_return_overrides():
    """Return return-type overrides for the small set of non-plugin aliases.

    Plugin-backed methods are generated from registry metadata by the stub
    builder, so new plugins do not need to touch this table.
    """
    return {
        "TextureNode": {
            "render": "Any",
            "show": "None",
            "save": "None",
            "named": "Any",
            "cache": "Any",
            "uncached": "Any",
        },
        "SDFTexture": {
            "__or__": "SDFTexture",
            "__and__": "SDFTexture",
            "__sub__": "SDFTexture",
            "intersect": "SDFTexture",
            "mask": "ColorTexture",
        },
        "ColorTexture": {
            "over": "ColorTexture",
            "to_lab": "ColorTexture",
            "to_rgb": "ColorTexture",
        },
        "PostNamespace": {
            "to_lab": "ColorTexture",
            "to_rgb": "ColorTexture",
        },
    }


class generate_api_stubs_command(Command):
    description = "generate public API type stubs from plugin metadata"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from build_tools.generate_api_stubs import generate_api_stubs

        generate_api_stubs()


if __name__ == "__main__":
    setuptools_setup(
        cmdclass={
            "generate_api_stubs": generate_api_stubs_command,
        }
    )
