"""Check explicit shader rule entries against their implementation contracts."""

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _ROOT / "src" / "sdf_ui"
_RULES_ROOT = _SRC_ROOT / "learning" / "shaders"
_RULE_FILES = ("primitives.py", "shading.py", "effects.py")
_RULE = "shader-rule-coverage"


def _literal_string(node: ast.expr, assignments: dict[str, ast.expr]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        value = assignments.get(node.id)
        if value is not None and value is not node:
            return _literal_string(value, assignments)
    return None


def _module_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        value = _assignment_value(node)
        for name in _assignment_names(node):
            if value is not None:
                assignments[name] = value
    return assignments


def _assignment_value(node: ast.stmt) -> ast.expr | None:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    return None


def _assignment_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _rule_entries(tree: ast.Module) -> tuple[dict[str, ast.expr], ast.Dict | None]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RULES"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                return {
                    key.value: value
                    for key, value in zip(
                        node.value.keys, node.value.values, strict=True
                    )
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }, node.value
    return {}, None


def _implementation_path(node: ast.ImportFrom) -> Path | None:
    if node.module is None:
        return None
    tokens = node.module.split(".")
    if "gpu" not in tokens:
        return None
    marker = tokens.index("gpu")
    if tokens[marker : marker + 2] != ["gpu", "nodes"]:
        return None
    module_name = "implementation" if tokens[-1] == "reference" else tokens[-1]
    return _SRC_ROOT.joinpath(*tokens[marker:-1], f"{module_name}.py")


def _reference_imports(node: ast.ImportFrom, path: Path) -> dict[str, Path]:
    return {
        alias.asname or alias.name: path
        for alias in node.names
        if alias.name == "reference"
    }


def _imports(tree: ast.Module) -> dict[str, Path]:
    imports: dict[str, Path] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        path = _implementation_path(node)
        if path is not None:
            imports.update(_reference_imports(node, path))
    return imports


def _contract_name(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    assignments = _module_assignments(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "ShaderDifferentiation"
        ):
            continue
        if node.args:
            return _literal_string(node.args[0], assignments)
        for keyword in node.keywords:
            if keyword.arg == "name":
                return _literal_string(keyword.value, assignments)
    return None


def lint_rule_file(path: Path, source_root: Path = _SRC_ROOT) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"{path}: [{_RULE}] cannot parse rule module: {exc}"]
    entries, assignment = _rule_entries(tree)
    if assignment is None:
        return [f"{path}: [{_RULE}] missing literal RULES mapping"]
    if any(
        not isinstance(key, ast.Constant) or not isinstance(key.value, str)
        for key in assignment.keys
    ):
        return [f"{path}: [{_RULE}] RULES must declare literal string keys"]

    aliases = _imports(tree)
    violations: list[str] = []
    for name, value in entries.items():
        violation = _entry_violation(path, name, value, aliases, source_root)
        if violation is not None:
            violations.append(violation)
    return violations


def _entry_violation(
    path: Path, name: str, value: ast.expr, aliases: dict[str, Path], source_root: Path
) -> str | None:
    if not isinstance(value, ast.Name):
        return _unresolved_reference(path, name, value.lineno)
    implementation = aliases.get(value.id)
    if implementation is None:
        return _unresolved_reference(path, name, value.lineno)
    implementation = _source_path(implementation, source_root)
    contract = _contract_name(implementation)
    if contract is None:
        return (
            f"{path}:{value.lineno}: [{_RULE}] rule '{name}' has no statically named "
            "ShaderDifferentiation contract"
        )
    if contract != name:
        return f"{path}:{value.lineno}: [{_RULE}] rule '{name}' references contract '{contract}'"
    return None


def _unresolved_reference(path: Path, name: str, line: int) -> str:
    return (
        f"{path}:{line}: [{_RULE}] rule '{name}' must reference an imported "
        "shader reference"
    )


def _source_path(implementation: Path, source_root: Path) -> Path:
    try:
        relative = implementation.relative_to(_SRC_ROOT)
    except ValueError:
        return implementation
    return source_root / relative


def lint_tree(
    rules_root: Path = _RULES_ROOT, source_root: Path = _SRC_ROOT
) -> list[str]:
    violations: list[str] = []
    for filename in _RULE_FILES:
        path = rules_root / filename
        violations.extend(lint_rule_file(path, source_root))
    return violations


def main() -> int:
    if not _RULES_ROOT.is_dir():
        print(
            f"shader rule coverage lint: rule package not found: {_RULES_ROOT}",
            file=sys.stderr,
        )
        return 1
    violations = lint_tree()
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print("shader rule coverage lint: explicit rules match their shader contracts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
