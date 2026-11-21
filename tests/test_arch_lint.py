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


def _assert_rejected_import(tmp_path, monkeypatch, relative_path, source, target):
    violations = _lint_source(tmp_path, monkeypatch, relative_path, source)
    assert len(violations) == 1
    assert target in violations[0]


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
    _assert_rejected_import(
        tmp_path,
        monkeypatch,
        Path("core") / "engine.py",
        "def load_plot():\n    from ..plot.api import Plot\n",
        "imports 'plot' (sdf_ui.plot.api, line 2)",
    )


def test_arch_lint_detects_deferred_import_from_composition_layer(
    tmp_path, monkeypatch
):
    _assert_rejected_import(
        tmp_path,
        monkeypatch,
        Path("plot") / "scene.py",
        "def render():\n    from sdf_ui.gpu import Canvas\n",
        "imports 'gpu' (sdf_ui.gpu, line 2)",
    )


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


@pytest.mark.parametrize(
    "relative_path,source",
    [
        (Path("plot") / "scene.py", "from sdf_ui.learning import Parameter\n"),
        (Path("plot") / "__init__.py", "from sdf_ui.gpu import Canvas\n"),
    ],
)
def test_arch_lint_allows_declared_edges_and_package_reexports(
    tmp_path, monkeypatch, relative_path, source
):
    assert _lint_source(tmp_path, monkeypatch, relative_path, source) == []


def test_arch_lint_allows_animation_adapter_in_gpu_context_only(tmp_path, monkeypatch):
    violations = _lint_source(
        tmp_path,
        monkeypatch,
        Path("gpu") / "context.py",
        "def save_animation():\n    from ..anim import save_animation\n",
    )

    assert violations == []


@pytest.mark.parametrize(
    "relative_path,source,target",
    [
        (
            Path("gpu") / "optim.py",
            "from ..anim import save_animation\n",
            "imports 'anim' (sdf_ui.anim, line 1)",
        ),
        (
            Path("gpu") / "context.py",
            "from ..anim.helpers import helper\n",
            "imports 'anim' (sdf_ui.anim.helpers, line 1)",
        ),
    ],
)
def test_arch_lint_limits_animation_exception_to_exact_module(
    tmp_path, monkeypatch, relative_path, source, target
):
    _assert_rejected_import(
        tmp_path,
        monkeypatch,
        relative_path,
        source,
        target,
    )
