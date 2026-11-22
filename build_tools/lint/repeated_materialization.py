"""Catch repeated sequence materialization for indexed lookups."""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "sdf_ui"
_REPEATED = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
)


def _lookup_source(node: ast.AST) -> str | None:
    match node:
        case ast.Subscript(
            value=ast.Call(
                func=ast.Name(id="list" | "tuple"),
                args=[ast.Name(id=source)],
                keywords=[],
            ),
            slice=ast.Name(),
        ):
            return source
    return None


def _binds(scope: ast.AST, name: str) -> bool:
    return any(
        (
            isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Store)
            and item.id == name
        )
        or (isinstance(item, ast.arg) and item.arg == name)
        for item in ast.walk(scope)
    )


def _repeated_from(node: ast.AST, name: str, parents: dict[ast.AST, ast.AST]) -> bool:
    while node in parents:
        parent = parents[node]
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if isinstance(parent, (ast.For, ast.AsyncFor)) and node is parent.iter:
            node = parent
            continue
        if isinstance(parent, _REPEATED):
            return not _binds(parent, name)
        node = parent
    return False


def lint_file(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        source = _lookup_source(node)
        if (
            isinstance(node, ast.Subscript)
            and source
            and _repeated_from(node, source, parents)
        ):
            yield f"{path}:{node.lineno}: [repeated-materialization] Materialize {source!r} once before repeated indexing"


def main() -> int:
    violations = [
        error for path in sorted(_SRC_ROOT.rglob("*.py")) for error in lint_file(path)
    ]
    for error in violations:
        print(error, file=sys.stderr)
    if violations:
        return 1
    print("repeated materialization lint: no violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
