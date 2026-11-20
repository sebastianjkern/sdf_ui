"""Check that generic composite plugins have graph-lowering dispatch."""

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_ROOT = _ROOT / "src" / "sdf_ui" / "gpu" / "nodes"
_LOWERING_FILE = _ROOT / "src" / "sdf_ui" / "graph" / "lowering.py"
_RULE = "shader-lowering-coverage"
_COMPOSITE_HELPERS = frozenset(
    {
        "composite",
        "_composite_effect",
        "_composite_fill_operation",
        "_composite_transform",
        "_composite_layer",
    }
)


def _string_values(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return set().union(*(_string_values(item) for item in node.elts))
    return set()


def _uses_generic_lowering(node: ast.Call) -> bool:
    return any(
        keyword.arg == "lowering_func"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "lower_composite"
        for keyword in node.keywords
    )


class _PluginVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loop_names: dict[str, set[str]] = {}
        self.plugins: list[tuple[str, int]] = []

    def visit_For(self, node: ast.For) -> None:
        name = node.target.id if isinstance(node.target, ast.Name) else None
        previous = self.loop_names.get(name) if name is not None else None
        if name is not None:
            self.loop_names[name] = _string_values(node.iter)
        self.generic_visit(node)
        if name is not None:
            if previous is None:
                self.loop_names.pop(name, None)
            else:
                self.loop_names[name] = previous

    def visit_Call(self, node: ast.Call) -> None:
        if _is_plugin_constructor(node) and _uses_generic_lowering(node):
            self.plugins.extend(_plugin_names(node, self.loop_names))
        self.generic_visit(node)


def _is_plugin_constructor(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "Plugin"


def _plugin_names(
    node: ast.Call, loop_names: dict[str, set[str]]
) -> list[tuple[str, int]]:
    if not node.args:
        return []
    name_node = node.args[0]
    names = _string_values(name_node)
    if isinstance(name_node, ast.Name):
        names.update(loop_names.get(name_node.id, set()))
    return [(name, node.lineno) for name in sorted(names)]


def _generic_composites(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _PluginVisitor()
    visitor.visit(tree)
    return visitor.plugins


class _DispatchVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_name: str | None = None
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous = self.function_name
        self.function_name = node.name
        self.generic_visit(node)
        self.function_name = previous

    def visit_Compare(self, node: ast.Compare) -> None:
        if _is_name_dispatch(node) and self.function_name in _COMPOSITE_HELPERS:
            self.names.update(_dispatch_values(node))
        self.generic_visit(node)


def _is_name_dispatch(node: ast.Compare) -> bool:
    return isinstance(node.left, ast.Name) and node.left.id == "name"


def _dispatch_values(node: ast.Compare) -> set[str]:
    names: set[str] = set()
    for operator, comparator in zip(node.ops, node.comparators):
        if isinstance(operator, (ast.Eq, ast.In)):
            names.update(_string_values(comparator))
    return names


def _dispatch_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _DispatchVisitor()
    visitor.visit(tree)
    return visitor.names


def lint_tree(plugin_root: Path, lowering_file: Path) -> list[str]:
    """Return generic composite registrations missing graph dispatch."""
    handled = _dispatch_names(lowering_file)
    violations: list[str] = []
    for path in sorted(plugin_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for name, line in _generic_composites(path):
            if name not in handled:
                violations.append(
                    f"{path}:{line}: [{_RULE}] missing-dispatch: Plugin '{name}' "
                    f"uses lower_composite but {lowering_file} has no explicit "
                    "graph-lowering branch for it"
                )
    return violations


def main() -> int:
    if not _PLUGIN_ROOT.is_dir() or not _LOWERING_FILE.is_file():
        print(f"{_RULE}: source roots not found", file=sys.stderr)
        return 1
    violations = lint_tree(_PLUGIN_ROOT, _LOWERING_FILE)
    if violations:
        print("\n".join(violations))
        print(f"{_RULE}: {len(violations)} violation(s)")
        return 1
    print(f"{_RULE}: no missing graph-lowering branches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
