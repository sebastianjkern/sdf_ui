import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "build_tools"
    / "lint"
    / "shader_rule_coverage.py"
)
_SPEC = importlib.util.spec_from_file_location("shader_rule_coverage", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
lint_rule_file = _MODULE.lint_rule_file


def _fixture(
    tmp_path: Path, *, rule_name: str, contract_name: str
) -> tuple[Path, Path]:
    source_root = tmp_path / "src" / "sdf_ui"
    implementation = (
        source_root / "gpu" / "nodes" / "primitives" / "circle" / "implementation.py"
    )
    implementation.parent.mkdir(parents=True)
    implementation.write_text(
        f'CONTRACT = ShaderDifferentiation("{contract_name}")\n', encoding="utf-8"
    )
    rule_file = tmp_path / "learning" / "shaders" / "primitives.py"
    rule_file.parent.mkdir(parents=True)
    rule_file.write_text(
        "from sdf_ui.gpu.nodes.primitives.circle.implementation import reference as _circle\n"
        f'RULES = {{"{rule_name}": _circle}}\n',
        encoding="utf-8",
    )
    return rule_file, source_root


def test_contract_name_mismatch_is_reported(tmp_path: Path) -> None:
    rule_file, source_root = _fixture(
        tmp_path, rule_name="circl", contract_name="circle"
    )

    violations = lint_rule_file(rule_file, source_root)

    assert len(violations) == 1
    assert "rule 'circl' references contract 'circle'" in violations[0]


def test_declared_contract_match_is_allowed(tmp_path: Path) -> None:
    rule_file, source_root = _fixture(
        tmp_path, rule_name="circle", contract_name="circle"
    )

    assert lint_rule_file(rule_file, source_root) == []


def test_dynamic_rule_extension_does_not_hide_declared_entries(tmp_path: Path) -> None:
    rule_file, source_root = _fixture(
        tmp_path, rule_name="circle", contract_name="circle"
    )
    rule_file.write_text(
        "from sdf_ui.gpu.nodes.primitives.circle.implementation import reference as _circle\n"
        'RULES = {"circle": _circle}\n'
        "RULES.update({name: rule for name, rule in generated_rules()})\n",
        encoding="utf-8",
    )

    assert lint_rule_file(rule_file, source_root) == []
