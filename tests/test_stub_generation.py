"""Generated stubs can be checked without modifying the checkout."""

from build_tools import generate_api_stubs as generator
from sdf_ui.gpu.nodes import loader


def test_stub_check_reports_stale_files_without_writing(tmp_path, monkeypatch, capsys):
    package = tmp_path / "sdf_ui"
    monkeypatch.setattr(generator, "PACKAGE", package)
    monkeypatch.setattr(generator, "CORE", package / "core")

    assert generator.generate_api_stubs()
    assert generator.generate_api_stubs(check=True)
    stub = package / "core" / "api.pyi"
    stub.write_text("outdated\n", encoding="utf-8")
    assert not generator.generate_api_stubs(check=True)
    assert stub.read_text(encoding="utf-8") == "outdated\n"
    assert "api.pyi" in capsys.readouterr().err
    assert generator.generate_api_stubs()
    assert generator.generate_api_stubs(check=True)


def test_plugin_loader_is_builtin_only():
    from sdf_ui.gpu.nodes.registry import PluginRegistry

    registry = PluginRegistry()
    loader.load_plugins(registry, packages=())
    assert registry._plugins == {}
