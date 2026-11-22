from pathlib import Path

from build_tools.lint.plugin_shader_inputs import lint_tree


def _fixture(
    root: Path,
    input_uniforms: str,
    *,
    kinds: str = "('sdf', 'height')",
    port_kind: str = ", port_kind='height'",
) -> None:
    plugin = root / "sample" / "plugin.py"
    implementation = root / "sample" / "implementation.py"
    plugin.parent.mkdir(parents=True)
    implementation.write_text(
        "from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface\n"
        "INTERFACE = ShaderInterface(\n"
        "    ImageBinding('dest', 0, write=True),\n"
        "    (ImageBinding('sdf', 1, read=True), ImageBinding('height', 2, read=True)),\n"
        ")\n"
        "CONTRACT = ShaderDifferentiation('sample', 'sample.wgsl', (\n"
        "    ShaderArgument('sdf', (), 'r32f'),\n"
        f"    ShaderArgument('height', (), 'r32f'{port_kind}),\n"
        "), (), ())\n",
        encoding="utf-8",
    )
    plugin_source = (
        "from .implementation import CONTRACT, INTERFACE\n"
        "def register_plugins(registry):\n"
        "    registry.register(Plugin(\n"
        f"        'sample', 'shading', 'color', {kinds},\n"
        "        shader=shader('sample', 'sample.wgsl', interface=INTERFACE,\n"
        "                      derivative_contract=CONTRACT),\n"
        f"        input_uniforms={input_uniforms},\n" + "    ))\n"
    )
    plugin.write_text(plugin_source, encoding="utf-8")


def test_lint_finds_plugin_input_name_missing_from_shader_interface(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, "('sdf', 'heightField')")

    violations = lint_tree(tmp_path)

    assert len(violations) == 1
    assert "missing=['height']" in violations[0]
    assert "undeclared=['heightField']" in violations[0]


def test_lint_allows_plugin_input_order_different_from_binding_order(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, "('height', 'sdf')", kinds="('height', 'sdf')")

    assert lint_tree(tmp_path) == []


def test_lint_finds_duplicate_plugin_input_names(tmp_path: Path) -> None:
    _fixture(tmp_path, "('sdf', 'sdf')")

    violations = lint_tree(tmp_path)

    assert len(violations) == 1
    assert "duplicates=['sdf']" in violations[0]


def test_lint_distinguishes_height_from_sdf_despite_shared_storage(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, "('sdf', 'height')", port_kind="")

    violations = lint_tree(tmp_path)

    assert len(violations) == 1
    assert (
        "input 'height' expects height in Plugin and sdf in ShaderDifferentiation"
        in violations[0]
    )
