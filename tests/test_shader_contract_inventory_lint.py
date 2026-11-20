from pathlib import Path

from build_tools.lint.shader_contract_inventory import lint_source


def test_shader_builder_requires_derivative_contract():
    source = "from sdf_ui.gpu.nodes.common import shader as make_shader\nmake_shader('x', 'x.glsl')"
    violations = lint_source(source, Path("plugin.py"))
    assert len(violations) == 1
    assert "shader-contract-required" in violations[0]


def test_shader_builder_accepts_contract_keyword_forms():
    source = """
from sdf_ui.gpu.nodes.common import shader as make_shader
make_shader('x', 'x.glsl', derivative_contract=CONTRACT)
common.shader('y', 'y.glsl', derivative_contract=CONTRACT)
"""
    assert lint_source(source, Path("plugin.py")) == []


def test_shader_builder_rejects_explicit_none_contract():
    source = "shader('x', 'x.glsl', derivative_contract=None)"
    violations = lint_source(source, Path("plugin.py"))
    assert len(violations) == 1
