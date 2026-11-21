"""Kernel layout: every WGSL shader lives next to its owning node folder.

The packaged kernel store (``gpu/shaders``) is gone. Each compute kernel is
colocated with the node plugin that dispatches it:

- ``{name}.wgsl`` for the rgba8 forward kernel;
- ``{name}_rgba32f.wgsl`` for an rgba32f forward variant, when one exists;
- ``{name}_backward.wgsl`` for the native local-VJP kernel, when the
  operation is natively differentiated.

Backend infrastructure kernels without a node plugin live in
``gpu/nodes/core``. The ``shader_layout`` lint enforces this layout.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent / "nodes"
_INFRA_DIR = "core"

_INDEX: dict[str, Path] = {}


def _base_stem(stem: str) -> str:
    if stem in {"", "_backward", "_rgba32f"}:
        raise ValueError(f"Invalid WGSL kernel file name '{stem}'")
    if stem.endswith("_backward"):
        return stem[: -len("_backward")]
    if stem.endswith("_rgba32f"):
        return stem[: -len("_rgba32f")]
    return stem


def _reindex() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(_ROOT.rglob("*.wgsl")):
        if "__pycache__" in path.parts:
            continue
        base = _base_stem(path.stem)
        if base in index and index[base] != path.parent:
            raise RuntimeError(
                f"Duplicate WGSL kernel name '{base}' in {path.parent.as_posix()}"
            )
        index[base] = path.parent
    return index


def _kernel_index() -> dict[str, Path]:
    if not _INDEX:
        _INDEX.update(_reindex())
    return _INDEX


def node_folder(name: str) -> Path:
    """Return the folder holding the kernels of shader *name*."""
    try:
        return _kernel_index()[name]
    except KeyError:
        raise KeyError(f"Unknown WebGPU shader '{name}'") from None


def kernel_path(name: str, color_format: str = "rgba8", backward: bool = False) -> Path:
    """Resolve the kernel file path for shader *name*.

    Backward kernels are always ``{name}_backward.wgsl`` next to the forward
    kernel. Forward rgba32f dispatches prefer a ``{name}_rgba32f.wgsl``
    variant and fall back to the base forward kernel.
    """
    folder = node_folder(name)
    if backward:
        return folder / f"{name}_backward.wgsl"
    if color_format != "rgba8":
        variant = folder / f"{name}_rgba32f.wgsl"
        if variant.is_file():
            return variant
    return folder / f"{name}.wgsl"


def supports(name: str) -> bool:
    """Return True when a packaged kernel for shader *name* is present."""
    return name in _kernel_index()


def infra_folder() -> Path:
    """Return the folder for backend infrastructure kernels."""
    folder = _ROOT / _INFRA_DIR
    folder.mkdir(exist_ok=True)
    return folder
