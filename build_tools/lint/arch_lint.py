"""Enforce the sdf_ui layer rules by parsing imports with ast.

Layers below `sdf_ui`:

- Engine layers (``core``, ``gpu``, ``graph``, ``learning``) may import each
  other and the support/text helpers. Load-order cycles between them are
  broken with deferred (function-scope) imports, so both module-scope and
  deferred imports are checked.
- Composition layers (``plot``, ``widgets``, ``chart``, ``anim``) build user
  scenes on top of the engine. The engine must never import them, and the
  scene layers must not reach into ``sdf_ui.gpu`` directly; they compose
  through the public ``sdf_ui.core`` API.
- Delivery modules (``ascii``, ``bw_to_sdf``) are user-facing drivers and may
  import anything.
- ``sdf_ui/__init__.py`` is the re-export seam and is exempt.

TYPE_CHECKING-only imports are exempt because they never execute at runtime.
"""

import ast
import sys
from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "sdf_ui"

ENGINE: frozenset[str] = frozenset({"core", "gpu", "graph", "learning"})
COMPOSITION: frozenset[str] = frozenset(
    {"plot", "widgets", "chart", "anim", "delivery"}
)
SUPPORT_MODULES = frozenset({"util", "log", "_json_identity"})
TEXT_MODULES = frozenset(
    {
        "text",
        "fonts",
        "_inline_text",
        "_font_bidi",
        "_font_graphemes",
        "_font_layout",
        "_font_resources",
    }
)

_HELPERS = frozenset(("support", "text"))
_LOW_LAYERS = frozenset(("core", "plot", "widgets"))
_CHART = frozenset(("chart",))

# Allowed imports per layer. A missing key allows nothing.
ALLOWED_IMPORTS: dict[str, AbstractSet[str]] = {
    "core": ENGINE | _HELPERS,
    "gpu": ENGINE | _HELPERS,
    "graph": ENGINE | _HELPERS,
    "learning": ENGINE | _HELPERS,
    "text": ENGINE | _HELPERS,
    "support": _HELPERS,
    "plot": _LOW_LAYERS | {"graph", "learning"} | _HELPERS,
    "widgets": _LOW_LAYERS | {"graph", "learning"} | _HELPERS,
    "chart": _LOW_LAYERS | _HELPERS | _CHART,
    "anim": ENGINE | _LOW_LAYERS | _HELPERS,
    "delivery": ENGINE | COMPOSITION | _HELPERS,
}


def source_layer(parts: tuple[str, ...]) -> str | None:
    """Map module path parts below sdf_ui to a layer name, None if outside."""
    if not parts or parts[-1] == "__init__":
        return None
    if parts[0] in ENGINE:
        return parts[0]
    if parts[0] in COMPOSITION:
        return parts[0]
    if parts[0] in ("ascii", "bw_to_sdf"):
        return "delivery"
    if parts[0] in SUPPORT_MODULES:
        return "support"
    if parts[0] in TEXT_MODULES:
        return "text"
    return None


def target_layer(module: str) -> str | None:
    """Map an absolute sdf_ui module name to a layer name, None if external."""
    tokens = module.split(".")
    if tokens[0] != "sdf_ui":
        return None
    if len(tokens) == 1:
        return None  # bare `import sdf_ui`
    return source_layer(tuple(tokens[1:]))


def resolve_import_node(
    node: ast.Import | ast.ImportFrom, parts: tuple[str, ...]
) -> Iterator[str]:
    """Yield absolute module names referenced by one import statement.

    ``parts`` is the full dotted module path of the importing file, starting
    at ``sdf_ui``; relative imports are resolved against it.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
        return
    if node.level > 0:
        base = list(parts[: len(parts) - node.level])
        if node.module:
            base.append(node.module)
        yield ".".join(base)
        return
    if node.module is None:
        return
    # `from sdf_ui import plot` binds the subpackage sdf_ui.plot.
    if node.module == "sdf_ui":
        for alias in node.names:
            yield f"sdf_ui.{alias.name}"
        return
    yield node.module


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def iter_runtime_imports(tree: ast.Module) -> Iterator[ast.Import | ast.ImportFrom]:
    """Yield every import node outside TYPE_CHECKING blocks."""
    checking_blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _is_type_checking(node.test)
    ]
    exempt = {id(child) for block in checking_blocks for child in ast.walk(block)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in exempt:
            yield node


def lint_file(path: Path) -> list[str]:
    """Return a violation per import that breaks the layer rules."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    source = path.relative_to(_SRC_ROOT).with_suffix("")
    layer_parts = tuple(source.parts[1:])  # strip the leading sdf_ui
    here = source_layer(layer_parts)
    if here is None:
        return []
    allowed = ALLOWED_IMPORTS.get(here, frozenset())

    violations: list[str] = []
    full_parts = tuple(source.parts)
    for node in iter_runtime_imports(tree):
        for module in resolve_import_node(node, full_parts):
            imported = target_layer(module)
            if imported is None or imported == here or imported in allowed:
                continue
            violations.append(
                f"layer violation: '{here}' imports '{imported}'"
                f" ({module}, line {node.lineno})"
            )
    return violations


def lint_tree(root: Path) -> Iterable[tuple[Path, list[str]]]:
    """Lint every module below root and yield (file, violations) pairs."""
    for path in sorted(root.rglob("*.py")):
        problems = lint_file(path)
        if problems:
            yield path, problems


def main() -> int:
    root = _SRC_ROOT
    if not root.is_dir():
        print(f"arch lint: package root not found: {root}", file=sys.stderr)
        return 1
    violations = list(lint_tree(root))
    for path, problems in violations:
        print(f"{path}")
        for problem in problems:
            print(f"  [!] {problem}")
    if violations:
        print(f"\narch lint: {sum(len(p) for _, p in violations)} violation(s)")
        return 1
    print("arch lint: no layer violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
