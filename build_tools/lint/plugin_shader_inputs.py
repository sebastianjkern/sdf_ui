"""Check plugin texture inputs against shader interfaces and contracts."""

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_ROOT = _ROOT / "src" / "sdf_ui" / "gpu" / "nodes"
_RULE = "plugin-shader-input-contract"
_FORMAT_KINDS = {"r32f": "sdf", "rgba32float": "field", "rgba8": "color"}


def _call_keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _call_argument(call: ast.Call, name: str, position: int) -> ast.expr | None:
    return (
        call.args[position] if len(call.args) > position else _call_keyword(call, name)
    )


def _string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_strings(node: ast.expr | None) -> tuple[str, ...] | None:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    values: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.append(item.value)
    return tuple(values)


def _interface_expression(shader: ast.expr) -> ast.expr | None:
    if not isinstance(shader, ast.Call):
        return None
    return _call_keyword(shader, "interface")


def _binding_name(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if not isinstance(function, ast.Name) or function.id != "ImageBinding":
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    value = node.args[0].value
    return value if isinstance(value, str) else None


def _interface_inputs(node: ast.expr | None) -> tuple[str, ...] | None:
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if not isinstance(function, ast.Name) or function.id != "ShaderInterface":
        return None
    inputs = _call_keyword(node, "inputs")
    if inputs is None and len(node.args) > 1:
        inputs = node.args[1]
    if inputs is None:
        return ()
    if not isinstance(inputs, (ast.Tuple, ast.List)):
        return None
    names = tuple(_binding_name(item) for item in inputs.elts)
    if any(name is None for name in names):
        return None
    return tuple(name for name in names if name is not None)


def _imported_original(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module == "implementation":
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return alias.name
    return None


def _assignment(tree: ast.Module, name: str) -> ast.expr | None:
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            return statement.value
    return None


def _resolve(path: Path, tree: ast.Module, node: ast.expr | None) -> ast.expr | None:
    if not isinstance(node, ast.Name):
        return node
    original = _imported_original(tree, node.id)
    if original is None:
        return _assignment(tree, node.id)
    try:
        implementation = ast.parse(
            path.with_name("implementation.py").read_text(encoding="utf-8")
        )
    except (OSError, SyntaxError):
        return None
    return _assignment(implementation, original)


def _name_violation(
    uniforms: tuple[str, ...], interface: tuple[str, ...]
) -> str | None:
    if len(uniforms) == len(interface) and set(uniforms) == set(interface):
        return None
    missing = sorted(set(interface) - set(uniforms))
    undeclared = sorted(set(uniforms) - set(interface))
    duplicate_names = sorted(name for name in set(uniforms) if uniforms.count(name) > 1)
    return (
        f"input_uniforms do not match ShaderInterface.inputs names; "
        f"missing={missing}, undeclared={undeclared}, duplicates={duplicate_names}"
    )


def _texture_kind(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "TextureKind":
            return node.attr.lower()
    return _string(node)


def _shader_port(node: ast.expr) -> tuple[str, str] | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "ShaderArgument":
        return None
    name = _string(_call_argument(node, "name", 0))
    storage = _string(_call_argument(node, "kind", 2))
    kind = _string(_call_argument(node, "port_kind", 3)) or _FORMAT_KINDS.get(
        storage or ""
    )
    return (name, kind) if name is not None and kind is not None else None


def _kind_violation(
    plugin: ast.Call, contract: ast.expr | None, names: tuple[str, ...]
) -> str | None:
    if not isinstance(contract, ast.Call):
        return None
    inputs = _call_argument(contract, "inputs", 2)
    kinds = _call_argument(plugin, "input_kinds", 3)
    if not isinstance(inputs, (ast.Tuple, ast.List)) or not isinstance(
        kinds, (ast.Tuple, ast.List)
    ):
        return None
    ports = dict(
        port for item in inputs.elts if (port := _shader_port(item)) is not None
    )
    for name, value in zip(names, kinds.elts):
        actual, declared = _texture_kind(value), ports.get(name)
        if actual is not None and declared is not None and actual != declared:
            return f"input '{name}' expects {actual} in Plugin and {declared} in ShaderDifferentiation"
    return None


def _plugin_violation(path: Path, tree: ast.Module, node: ast.Call) -> str | None:
    uniforms = _literal_strings(_call_keyword(node, "input_uniforms"))
    shader = _call_keyword(node, "shader")
    if uniforms is None or not isinstance(shader, ast.Call):
        return None
    interface = _interface_inputs(_resolve(path, tree, _interface_expression(shader)))
    if interface is not None:
        violation = _name_violation(uniforms, interface)
        if violation is not None:
            return violation
    contract = _resolve(path, tree, _call_keyword(shader, "derivative_contract"))
    return _kind_violation(node, contract, uniforms)


def _plugin_violations(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "Plugin":
            continue
        detail = _plugin_violation(path, tree, node)
        if detail is not None:
            violations.append(f"{path}:{node.lineno}: [{_RULE}] {detail}")
    return violations


def lint_tree(root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("plugin.py")):
        violations.extend(_plugin_violations(path))
    return violations


def main() -> int:
    if not _PLUGIN_ROOT.is_dir():
        print(f"{_RULE}: plugin root not found: {_PLUGIN_ROOT}", file=sys.stderr)
        return 1
    violations = lint_tree(_PLUGIN_ROOT)
    if violations:
        print("\n".join(violations))
        print(f"{_RULE}: {len(violations)} violation(s)")
        return 1
    print(f"{_RULE}: shader input declarations match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
