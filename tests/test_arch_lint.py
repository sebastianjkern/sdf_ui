from pathlib import Path

import pytest

from build_tools.lint import arch_lint


def _lint_source(tmp_path, monkeypatch, relative_path, source):
    root = tmp_path / "src" / "sdf_ui"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(arch_lint, "_SRC_ROOT", root)
    return arch_lint.lint_file(path)


@pytest.mark.parametrize(
    "source",
    [
        "from sdf_ui import plot as plotting\n",
        "import sdf_ui.plot.api as plotting\n",
    ],
)
def test_arch_lint_detects_aliased_absolute_imports(tmp_path, monkeypatch, source):
    violations = _lint_source(tmp_path, monkeypatch, Path("core") / "engine.py", source)

    assert len(violations) == 1
    assert "imports 'plot' (sdf_ui.plot" in violations[0]


def test_arch_lint_detects_deferred_relative_imports(tmp_path, monkeypatch):
    source = "def load_plot():\n    from ..plot.api import Plot\n"

    violations = _lint_source(tmp_path, monkeypatch, Path("core") / "engine.py", source)

    assert len(violations) == 1
    assert "imports 'plot' (sdf_ui.plot.api, line 2)" in violations[0]


def test_arch_lint_detects_deferred_import_from_composition_layer(
    tmp_path, monkeypatch
):
    source = "def render():\n    from sdf_ui.gpu import Canvas\n"

    violations = _lint_source(tmp_path, monkeypatch, Path("plot") / "scene.py", source)

    assert len(violations) == 1
    assert "imports 'gpu' (sdf_ui.gpu, line 2)" in violations[0]


@pytest.mark.parametrize(
    "source",
    [
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from sdf_ui import plot\n",
        "import typing\n"
        "if typing.TYPE_CHECKING:\n"
        "    if True:\n"
        "        from sdf_ui import plot\n",
    ],
)
def test_arch_lint_exempts_nested_type_checking_imports(tmp_path, monkeypatch, source):
    violations = _lint_source(tmp_path, monkeypatch, Path("core") / "types.py", source)

    assert violations == []


def test_arch_lint_allows_declared_layer_edges(tmp_path, monkeypatch):
    violations = _lint_source(
        tmp_path,
        monkeypatch,
        Path("plot") / "scene.py",
        "from sdf_ui.learning import Parameter\n",
    )

    assert violations == []


def test_arch_lint_allows_animation_adapter_in_gpu_context_only(tmp_path, monkeypatch):
    violations = _lint_source(
        tmp_path,
        monkeypatch,
        Path("gpu") / "context.py",
        "def save_animation():\n    from ..anim import save_animation\n",
    )

    assert violations == []


def test_arch_lint_rejects_animation_imports_from_other_gpu_modules(
    tmp_path, monkeypatch
):
    violations = _lint_source(
        tmp_path,
        monkeypatch,
        Path("gpu") / "optim.py",
        "from ..anim import save_animation\n",
    )

    assert len(violations) == 1
    assert "imports 'anim' (sdf_ui.anim, line 1)" in violations[0]


def test_arch_lint_keeps_animation_exception_module_specific(tmp_path, monkeypatch):
    violations = _lint_source(
        tmp_path,
        monkeypatch,
        Path("gpu") / "context.py",
        "from ..anim.helpers import helper\n",
    )

    assert len(violations) == 1
    assert "imports 'anim' (sdf_ui.anim.helpers, line 1)" in violations[0]


def test_arch_lint_exempts_package_reexports(tmp_path, monkeypatch):
    violations = _lint_source(
        tmp_path,
        monkeypatch,
        Path("plot") / "__init__.py",
        "from sdf_ui.gpu import Canvas\n",
    )

    assert violations == []
