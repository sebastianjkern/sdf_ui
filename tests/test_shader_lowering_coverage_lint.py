from pathlib import Path

from build_tools.lint.shader_lowering_coverage import lint_tree


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_generic_composite_plugins_require_explicit_lowering_dispatch(tmp_path):
    plugin_root = tmp_path / "nodes"
    lowering_file = tmp_path / "lowering.py"
    plugin_file = plugin_root / "plugin.py"
    _write(
        plugin_file,
        "for name in ('alpha', 'beta'):\n"
        "    registry.register(Plugin(name, lowering_func=lower_composite))\n",
    )
    _write(
        lowering_file,
        "class Lowerer:\n"
        "    def composite(self, name):\n"
        "        if name in {'alpha'}:\n"
        "            return self.emit(name)\n",
    )

    violations = lint_tree(plugin_root, lowering_file)

    assert len(violations) == 1
    assert "plugin.py:2" in violations[0]
    assert "Plugin 'beta'" in violations[0]


def test_explicit_composite_dispatch_and_shader_plugins_are_allowed(tmp_path):
    plugin_root = tmp_path / "nodes"
    lowering_file = tmp_path / "lowering.py"
    _write(
        plugin_root / "plugin.py",
        "registry.register(Plugin('alpha', lowering_func=lower_composite))\n"
        "registry.register(Plugin('raw_shader', shader=shader_file))\n",
    )
    _write(
        lowering_file,
        "class Lowerer:\n"
        "    def composite(self, name):\n"
        "        if name == 'alpha':\n"
        "            return self.emit(name)\n",
    )

    assert lint_tree(plugin_root, lowering_file) == []
