"""Enforce the colocated WGSL node-kernel layout.

The WebGPU backend is WGSL-only. Every kernel lives next to the node whose
plugin registers it: a leaf node folder (one containing ``plugin.py``) must
hold a ``{op}.wgsl`` forward kernel and, when the operation ships a native
backward kernel, a ``{op}_backward.wgsl`` next to it. Kernel-only infra files
live in ``gpu/nodes/core``. The old monolithic ``gpu/shaders`` folder and any
GLSL file are forbidden.

Backward kernels are required for every packaged (non-generator) shader
descriptor except the operations listed in ``_TRAINING_UNSUPPORTED``: display
helpers (``field_*``, ``warp_*``, ``repeat_grid``) intentionally have no
native VJP today. Shrink that set as kernels land; never grow it silently.
"""

import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

_PLUGIN_MARKER = "plugin.py"
_BACKWARD_SUFFIX = "_backward"
_RGBA32F_SUFFIX = "_rgba32f"

# Packaged forward kernels without a native backward VJP today.
_TRAINING_UNSUPPORTED: frozenset[str] = frozenset(
    {
        "constant_field",
        "field_add",
        "field_contours",
        "field_scale",
        "field_visualize",
        "radial_field",
        "repeat_grid",
        "turbulence_field",
        "vortex_field",
        "warp_color",
        "warp_sdf",
    }
)

# Kernels referenced by name from plugin code (direct dispatch or with_name
# reads) without a registry descriptor of their own. Keep in sync with the
# dispatch sites; a kernel that loses its reader must be deleted, not listed.
_DIRECT_DISPATCH_KERNELS: frozenset[str] = frozenset(
    {
        "normalize_distance",
        "overlay_batch",
        "overlay_fused",
        "polyline",
        "redistance_finalize",
        "redistance_jump",
        "redistance_seed",
    }
)


@dataclass(frozen=True)
class Descriptor:
    """The subset of a registered shader descriptor the layout needs."""

    name: str
    path: str
    has_wgsl_generator: bool


def _nodes_root(src_root: Path) -> Path:
    """Return the node root ``<src_root>/gpu/nodes``."""
    return src_root / "gpu" / "nodes"


def _descriptor_folder(src_root: Path, item: Descriptor) -> Path:
    """Return the folder holding the packaged kernel of *item*."""
    declared = Path(item.path)
    if declared.is_absolute():
        return declared.parent
    return (src_root / declared).parent


def _packed_names(
    src_root: Path, descriptors: Iterable[Descriptor]
) -> dict[Path, set[str]]:
    """Map each folder to the packaged operation names declared in it."""
    folder_names: dict[Path, set[str]] = {}
    for item in descriptors:
        item_names = folder_names.setdefault(_descriptor_folder(src_root, item), set())
        item_names.add(item.name)
    return folder_names


def _kernel_file_names(names: Iterable[str]) -> set[str]:
    """Return every kernel basename the given operations may declare."""
    suffixes = (
        "",
        _RGBA32F_SUFFIX,
        _BACKWARD_SUFFIX,
        f"{_RGBA32F_SUFFIX}{_BACKWARD_SUFFIX}",
    )
    return {f"{name}{suffix}.wgsl" for name in names for suffix in suffixes}


def _kernel_base_stem(filename: str) -> str:
    """Strip the kernel suffix variants from a kernel filename."""
    if not filename.endswith(".wgsl"):
        return filename
    stem = filename[: -len(".wgsl")]
    for suffix in (
        f"{_RGBA32F_SUFFIX}{_BACKWARD_SUFFIX}",
        _RGBA32F_SUFFIX,
        _BACKWARD_SUFFIX,
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _glsl_and_layout_violations(src_root: Path) -> list[str]:
    """Flag GLSL files, stray shader folders and out-of-place WGSL files."""
    violations: list[str] = []
    stray = src_root / "gpu" / "shaders"
    if stray.is_dir():
        violations.append(
            f"shader-directory-forbidden: {stray} must not exist; "
            "kernels live in their node folders"
        )
    for glsl in sorted(src_root.rglob("*.glsl")):
        violations.append(f"glsl-shader-forbidden: {glsl}")
    nodes = _nodes_root(src_root)
    for wgsl in sorted(src_root.rglob("*.wgsl")):
        if not wgsl.is_relative_to(nodes):
            violations.append(f"wgsl-outside-node-folder: {wgsl}")
    return violations


def _leaf_folders(src_root: Path) -> set[Path]:
    """Return every leaf node folder (one holding a ``plugin.py``)."""
    return {path.parent for path in _nodes_root(src_root).rglob(_PLUGIN_MARKER)}


def _packaged_per_folder(
    src_root: Path, descriptors: Iterable[Descriptor]
) -> dict[Path, set[str]]:
    """Map each folder to packaged descriptor names without WGSL generators."""
    folder_names: dict[Path, set[str]] = {}
    generators = {item.name for item in descriptors if item.has_wgsl_generator}
    for item in descriptors:
        if item.name in generators:
            continue
        names = folder_names.setdefault(_descriptor_folder(src_root, item), set())
        names.add(item.name)
    return folder_names


def _kernel_placement_violations(
    src_root: Path, descriptors: Iterable[Descriptor]
) -> list[str]:
    """Flag packaged descriptors whose kernels are missing or not colocated."""
    violations: list[str] = []
    leaves = _leaf_folders(src_root)
    for folder, names in sorted(
        _packaged_per_folder(src_root, descriptors).items(),
        key=lambda item: str(item[0]),
    ):
        if folder not in leaves:
            violations.append(
                f"kernel-not-colocated: {sorted(names)} declared in {folder} "
                "must live in a leaf node folder (one holding a plugin.py)"
            )
            continue
        for name in sorted(names):
            forward = folder / f"{name}.wgsl"
            if not forward.is_file():
                violations.append(f"forward-kernel-missing: {forward}")
            if name in _TRAINING_UNSUPPORTED:
                continue
            backward = folder / f"{name}{_BACKWARD_SUFFIX}.wgsl"
            if not backward.is_file():
                violations.append(f"backward-kernel-missing: {backward}")
    return violations


def _orphan_kernel_violations(
    src_root: Path, descriptors: Iterable[Descriptor]
) -> list[str]:
    """Flag WGSL files in leaf folders that no descriptor declares."""
    violations: list[str] = []
    declared_by_folder = {
        folder: _kernel_file_names(names)
        for folder, names in _packed_names(src_root, descriptors).items()
    }
    for leaf in sorted(_leaf_folders(src_root)):
        base_stems = set(_DIRECT_DISPATCH_KERNELS)
        base_stems.update(
            _kernel_base_stem(name) for name in declared_by_folder.get(leaf, set())
        )
        for wgsl in sorted(leaf.rglob("*.wgsl")):
            if _kernel_base_stem(wgsl.name) not in base_stems:
                violations.append(f"kernel-unreferenced: {wgsl}")
    return violations


def lint_tree(src_root: Path, descriptors: Sequence[Descriptor]) -> list[str]:
    """Return every layout violation for the tree rooted at *src_root*."""
    return (
        _glsl_and_layout_violations(src_root)
        + _kernel_placement_violations(src_root, descriptors)
        + _orphan_kernel_violations(src_root, descriptors)
    )


def _registry_descriptors(src_root: Path) -> Iterator[Descriptor]:
    sys.path.insert(0, str(src_root.parent))
    from sdf_ui.gpu.nodes.registry import registry

    registry.ensure_loaded()
    for item in sorted(registry.shader_files(), key=lambda entry: entry.path):
        yield Descriptor(item.name, item.path, item.wgsl is not None)


def main() -> int:
    """Run the layout lint over the real tree and exit nonzero on violations."""
    src_root = _ROOT / "src" / "sdf_ui"
    violations = lint_tree(src_root, list(_registry_descriptors(src_root)))
    for violation in violations:
        print(f"shader_layout: {violation}")
    if violations:
        print(f"shader_layout: {len(violations)} violation(s)")
        return 1
    print("shader layout is colocated and WGSL-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
