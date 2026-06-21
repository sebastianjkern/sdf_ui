from __future__ import annotations

from pathlib import Path
import sys
from importlib.metadata import PackageNotFoundError, version


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

project = "sdf_ui"
author = "Sebastian Kern"
copyright = "2026, Sebastian Kern"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

try:
    release = version("sdf_ui")
except PackageNotFoundError:
    release = "0.1.0"

version = release

html_theme = "alabaster"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
}
