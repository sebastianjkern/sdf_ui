"""Generated stubs can be checked without modifying the checkout."""

from build_tools import generate_api_stubs as generator
from sdf_ui.core.plugins import loader


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


def test_stub_generation_ignores_external_plugins(monkeypatch):
    def fail_if_discovered():
        raise AssertionError("stub generation consulted installed entry points")

    monkeypatch.setattr(loader, "entry_points", fail_if_discovered)
    assert generator.generate_api_stubs(check=True)


def test_plugin_loader_uses_entry_point_group(monkeypatch):
    from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
    from sdf_ui.core.plugins.registry import PluginRegistry

    plugin = Plugin("external_test", PluginFamily.PRIMITIVE, TextureKind.SDF)

    class Entry:
        def load(self):
            return lambda registry: registry.register(plugin)

    class Entries:
        def select(self, *, group):
            assert group == "sdf_ui.core.plugins"
            return [Entry()]

    monkeypatch.setattr(loader, "entry_points", Entries)
    registry = PluginRegistry()
    loader.load_plugins(registry, packages=())
    assert registry._plugins["external_test"] is plugin
