"""Statically check GPU resource lifecycle conventions in ``src/sdf_ui/gpu``.

Three rules encode the conventions the shader-lifecycle guidance requires:

``pipeline-creation-confined``
    ``create_shader_module`` and ``create_compute_pipeline`` must only be
    called inside ``gpu/context.py``. All compute pipelines must be created
    through the context's ``compute_pipeline`` cache so the cache key covers
    the full source and entry point.

``deferred-destroy-guard``
    Outside the low-level owner files (``context.py``, ``backend.py``, and
    the synchronous readback staging in ``image.py``), a resource may only be
    destroyed after ``ctx._defer_resource_destroy(...)`` declined it. This
    keeps destruction ordered after queued, not-yet-submitted batch work.

``close-input-forbidden``
    A shader operation must not close a rendered input or its underlying
    texture: ``.close()`` on an ``inputs[...]`` element or on an ``.tex``
    attribute chain is a violation. Plugins close only resources they
    allocated themselves.

Scope-limited static checks only. Ownership aliasing, cache boundedness, and
exception-path coverage are exercised by the runtime tests, not this script.
"""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "sdf_ui"
_GPU_ROOT = _SRC_ROOT / "gpu"

_RULE = "gpu-lifecycle"

_PIPELINE_OWNERS = frozenset({"context.py"})
_PIPELINE_CREATORS = frozenset({"create_shader_module", "create_compute_pipeline"})

# Owners that submit their own encoders and may destroy immediately after the
# submit: the context (its deferred queue runs on submit), the wgpu backend
# wrapper, and the synchronous image readback staging buffers in image.py.
_DESTROY_OWNERS = frozenset({"context.py", "backend.py", "image.py"})

_DEFER_ATTR = "_defer_resource_destroy"
_INPUT_NAMES = frozenset({"inputs", "_inputs", "tops"})


def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(root):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _test_defers_resource(test: ast.expr) -> bool:
    for node in ast.walk(test):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == _DEFER_ATTR
        ):
            return True
    return False


def _destroy_is_defer_guarded(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    current: ast.AST = node
    while (parent := parents.get(current)) is not None:
        current = parent
        if isinstance(current, ast.If) and _test_defers_resource(current.test):
            return True
    return False


def _pipeline_violation(path: Path, node: ast.Call) -> str | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    receiver = func.value
    creates_pipeline = (
        isinstance(receiver, ast.Attribute) and receiver.attr == "device"
    ) or (isinstance(receiver, ast.Name) and receiver.id == "device")
    if creates_pipeline and path.name not in _PIPELINE_OWNERS:
        return (
            f"{path}:{node.lineno}: [{_RULE}] pipeline-creation-confined: "
            f"{func.attr} must be called in gpu/context.py only "
            "(so the pipeline cache key covers full source and entry point)"
        )
    return None


def _destroy_violation(
    path: Path, node: ast.Call, parents: dict[ast.AST, ast.AST]
) -> str | None:
    if path.name in _DESTROY_OWNERS or _destroy_is_defer_guarded(node, parents):
        return None
    return (
        f"{path}:{node.lineno}: [{_RULE}] deferred-destroy-guard: "
        "destroy() must follow a ctx._defer_resource_destroy() check "
        "so destruction is ordered after queued batch work"
    )


def _close_violation(path: Path, node: ast.Call) -> str | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    target = func.value
    closes_subscript_input = (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id in _INPUT_NAMES
    )
    closes_owned_texture = isinstance(target, ast.Attribute) and target.attr == "tex"
    if closes_subscript_input or closes_owned_texture:
        return (
            f"{path}:{node.lineno}: [{_RULE}] close-input-forbidden: "
            "shader operations may not close rendered inputs or a wrapped "
            "texture's underlying resource"
        )
    return None


def _violation_for_call(
    path: Path, node: ast.Call, parents: dict[ast.AST, ast.AST]
) -> str | None:
    func = node.func
    if isinstance(func, ast.Name) or not isinstance(func, ast.Attribute):
        return None
    if func.attr in _PIPELINE_CREATORS:
        return _pipeline_violation(path, node)
    if func.attr == "destroy":
        return _destroy_violation(path, node, parents)
    if func.attr == "close":
        return _close_violation(path, node)
    return None


def _module_violations(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        print(f"gpu lifecycle lint: parse error for {path}: {exc}", file=sys.stderr)
        return []
    parents = _parent_map(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            violation = _violation_for_call(path, node, parents)
            if violation is not None:
                violations.append(violation)
    return violations


def _python_modules(gpu_root: Path) -> Iterator[Path]:
    for path in sorted(gpu_root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def lint_tree(gpu_root: Path) -> list[str]:
    """Return one violation line per lifecycle convention breach."""
    violations: list[str] = []
    for path in _python_modules(gpu_root):
        violations.extend(_module_violations(path))
    return violations


def main() -> int:
    root = _GPU_ROOT
    if not root.is_dir():
        print(
            f"gpu lifecycle lint: gpu package root not found: {root}", file=sys.stderr
        )
        return 1
    violations = lint_tree(root)
    if not violations:
        print("gpu lifecycle lint: no lifecycle violations")
        return 0
    for violation in violations:
        print(f"  [!] {violation}")
    print(f"\ngpu lifecycle lint: {len(violations)} violation(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
