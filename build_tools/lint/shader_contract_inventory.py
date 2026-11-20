"""Require every shader descriptor builder to name its derivative contract."""

import ast
import sys
from pathlib import Path

_GPU_NODES_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "sdf_ui" / "gpu" / "nodes"
)
_RULE = "shader-contract-required"
_BUILDER = "shader"
_CONTRACT_KEYWORD = "derivative_contract"


def _builder_names(tree: ast.Module) -> frozenset[str]:
    names = {_BUILDER}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name == _BUILDER:
                names.add(alias.asname or alias.name)
    return frozenset(names)


def _is_shader_builder(node: ast.expr, names: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    return isinstance(node, ast.Attribute) and node.attr == _BUILDER


def _shader_call_violation(
    path: Path, call: ast.Call, names: frozenset[str]
) -> str | None:
    if not _is_shader_builder(call.func, names):
        return None
    contract = next(
        (
            keyword.value
            for keyword in call.keywords
            if keyword.arg == _CONTRACT_KEYWORD
        ),
        None,
    )
    if contract is not None and not (
        isinstance(contract, ast.Constant) and contract.value is None
    ):
        return None
    return (
        f"{path}:{call.lineno}: [{_RULE}] shader descriptor must declare a "
        f"{_CONTRACT_KEYWORD}"
    )


def lint_source(source: str, path: Path) -> list[str]:
    """Return source-level descriptor contract violations."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: parse error: {exc}"]
    names = _builder_names(tree)
    return [
        violation
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (violation := _shader_call_violation(path, node, names)) is not None
    ]


def lint_tree(root: Path) -> list[str]:
    violations: list[str] = []
    modules = sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )
    for path in modules:
        violations.extend(lint_source(path.read_text(encoding="utf-8"), path))
    return violations


def main() -> int:
    if not _GPU_NODES_ROOT.is_dir():
        print(
            f"shader contract lint: node package root not found: {_GPU_NODES_ROOT}",
            file=sys.stderr,
        )
        return 1
    violations = lint_tree(_GPU_NODES_ROOT)
    if violations:
        for violation in violations:
            print(violation)
        print(f"shader contract lint: {len(violations)} violation(s)")
        return 1
    print("shader contract lint: all descriptor builders declare derivative contracts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
