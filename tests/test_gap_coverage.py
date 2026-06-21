from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import sdf_ui.ascii as ascii_mod
import sdf_ui.anim as anim_mod
import sdf_ui.core.context as context_mod
import sdf_ui.core.operations as operations_mod
import sdf_ui.core.shaders as shaders_mod
import sdf_ui.core.texture_utils as texture_utils_mod
from sdf_ui import color, sdf
from sdf_ui.core.color import ColorSpaceMode
from sdf_ui.core.texture import Renderer
from sdf_ui.text import glyph
from sdf_ui.util import collinear, rgb_col


class _DummyCodes:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __getattr__(self, name: str):
        return f"<{self.prefix}:{name}>"


class _FakeTexture:
    def __init__(self, *, size=(1, 1), components=4, dtype="f1", payload=b""):
        self.size = size
        self.components = components
        self.dtype = dtype
        self._payload = payload or (b"\x00" * (size[0] * size[1] * components))
        self.show_calls = 0

    def read(self):
        return self._payload

    def transpose(self, _mode):
        return self

    def show(self):
        self.show_calls += 1


def test_ascii_validation_rejects_invalid_requests():
    with pytest.raises(ValueError, match="cols must be greater than 0"):
        ascii_mod._validate_ascii_dimensions(0, 1, 8, 8, 1)

    with pytest.raises(ValueError, match="scale must be greater than 0"):
        ascii_mod._validate_ascii_dimensions(1, 0, 8, 8, 1)

    with pytest.raises(ValueError, match="larger than the source image"):
        ascii_mod._validate_ascii_dimensions(9, 1, 8, 8, 1)


def test_ascii_grayscale_conversion_prints_rows(capsys):
    image = Image.new("L", (2, 2), color=0)

    ascii_mod.convert_image_to_ascii(image, cols=1, scale=1.0, more_levels=False, invert=False, enhance=False)

    assert capsys.readouterr().out.strip() == "@"


def test_ascii_colored_conversion_uses_expected_terminal_codes(monkeypatch, capsys):
    monkeypatch.setattr(ascii_mod, "fg", _DummyCodes("fg"))
    monkeypatch.setattr(ascii_mod, "bg", _DummyCodes("bg"))
    monkeypatch.setattr(ascii_mod, "fx", SimpleNamespace(end="<end>"))

    image = Image.new("RGBA", (2, 2), (0, 0, 0, 255))

    ascii_mod.convert_image_to_ascii_colored(
        image,
        cols=1,
        scale=1.0,
        more_levels=False,
        invert=False,
        enhance=False,
        as_background=False,
    )
    ascii_mod.convert_image_to_ascii_colored(
        image,
        cols=1,
        scale=1.0,
        more_levels=False,
        invert=False,
        enhance=False,
        as_background=True,
    )

    output = capsys.readouterr().out.splitlines()
    assert output[0] == "<fg:t_000000>@<end>"
    assert output[1] == "<bg:t_000000> <end>"


def test_print_texture_routes_to_the_expected_converter(monkeypatch):
    calls = []

    def fake_grayscale(*args, **kwargs):
        calls.append(("grayscale", args, kwargs))

    def fake_colored(*args, **kwargs):
        calls.append(("colored", args, kwargs))

    monkeypatch.setattr(ascii_mod, "convert_image_to_ascii", fake_grayscale)
    monkeypatch.setattr(ascii_mod, "convert_image_to_ascii_colored", fake_colored)

    texture = SimpleNamespace(tex=_FakeTexture(size=(2, 2), components=4, payload=b"\x00" * 16))

    ascii_mod.print_texture(texture, colored=False)
    ascii_mod.print_texture(texture, colored=True)

    assert [call[0] for call in calls] == ["grayscale", "colored"]


def test_convert_to_video_writes_all_frames_and_logs(monkeypatch, tmp_path):
    written = []
    releases = []
    log_messages = []

    class FakeWriter:
        def __init__(self, name, fourcc, fps, size):
            self.name = name
            self.fourcc = fourcc
            self.fps = fps
            self.size = size

        def write(self, frame):
            written.append(frame)

        def release(self):
            releases.append(self.name)

    def fake_imread(path):
        return np.zeros((10, 20, 3), dtype=np.uint8)

    monkeypatch.setattr(anim_mod.cv2, "imread", fake_imread)
    monkeypatch.setattr(anim_mod.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(anim_mod.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(anim_mod, "logger", lambda: SimpleNamespace(info=log_messages.append))

    output_name = str(tmp_path / "movie")
    anim_mod.convert_to_video(output_name, ["frame1.png", "frame2.png"])

    assert len(written) == 2
    assert all(frame.shape == (10, 20, 3) for frame in written)
    assert releases == [output_name + ".mp4"]
    assert log_messages == ["Finished writing video"]


def _fake_context(size=(4, 4)):
    ctx = context_mod.Context.__new__(context_mod.Context)
    ctx.size = size
    ctx._closed = False
    ctx.local_size = (1, 1, 1)
    return ctx


def test_context_texture_loading_flips_and_converts(monkeypatch):
    captured = {}

    class FakeMGLContext:
        def texture(self, size, components, data):
            captured["size"] = size
            captured["components"] = components
            captured["data"] = np.array(data, copy=True)
            return SimpleNamespace(size=size, components=components, data=data)

    source = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )

    monkeypatch.setattr(context_mod.cv2, "imread", lambda _path: source.copy())
    monkeypatch.setattr(context_mod.cv2, "cvtColor", lambda arr, _code: arr[..., ::-1])

    ctx = _fake_context()
    ctx._mgl_ctx = FakeMGLContext()

    texture = context_mod.Context.texture_from_image(ctx, "ignored.png")

    assert captured["size"] == (2, 2)
    assert captured["components"] == 3
    assert captured["data"].tolist() == [
        [[9, 8, 7], [12, 11, 10]],
        [[3, 2, 1], [6, 5, 4]],
    ]
    assert texture.size == (2, 2)


def test_context_get_shader_close_and_init_helpers(monkeypatch):
    library_calls = []
    release_calls = []

    class FakeLibrary:
        def get(self, shader_name):
            library_calls.append(shader_name)
            return f"shader:{shader_name}"

    ctx = _fake_context()
    ctx._shader_library = FakeLibrary()
    ctx._mgl_ctx = SimpleNamespace(release=lambda: release_calls.append("release"))

    assert context_mod.Context.get_shader(ctx, "demo") == "shader:demo"
    assert context_mod.Context.__enter__(ctx) is ctx
    context_mod.Context.__exit__(ctx, None, None, None)
    context_mod.Context.close(ctx)

    created = []

    class FakeContextFactory:
        def __init__(self, size):
            created.append(size)

    monkeypatch.setattr(context_mod, "Context", FakeContextFactory)
    context_mod.init_sdf_ui((16, 9))

    assert library_calls == ["demo"]
    assert release_calls == ["release"]
    assert created == [(16, 9)]


def test_context_texture_allocators_increment_registry(monkeypatch):
    class FakeTex:
        def __init__(self, size, components, dtype=None):
            self.size = size
            self.components = components
            self.dtype = dtype
            self.filter = None

    class FakeMGLContext:
        def texture(self, size, components, data=None, dtype=None):
            return FakeTex(size, components, dtype=dtype)

    monkeypatch.setattr(context_mod, "tex_registry", texture_utils_mod.Counter(0))
    monkeypatch.setattr(context_mod, "mgl", SimpleNamespace(LINEAR="linear"))

    ctx = _fake_context((8, 8))
    ctx._mgl_ctx = FakeMGLContext()

    r32f = context_mod.Context.r32f(ctx)
    rgba8 = context_mod.Context.rgba8(ctx)

    assert r32f.size == (8, 8)
    assert rgba8.size == (8, 8)
    assert r32f.filter == ("linear", "linear")
    assert rgba8.filter == ("linear", "linear")
    assert int(context_mod.tex_registry) == 2


def test_shader_library_caches_compilation_and_reports_missing_files(tmp_path, monkeypatch):
    shader_file = tmp_path / "demo.glsl"
    shader_file.write_text("void main() {}", encoding="utf-8")

    descriptors = [SimpleNamespace(name="demo", path="demo.glsl")]
    monkeypatch.setattr(shaders_mod.registry, "shader_files", lambda: tuple(descriptors))

    compiled = []

    class FakeMGLContext:
        def compute_shader(self, code):
            compiled.append(code)
            return f"compiled:{code}"

    library = shaders_mod.ShaderLibrary(FakeMGLContext(), base_path=tmp_path)

    assert library.get("demo") == "compiled:void main() {}"
    assert library.get("demo") == "compiled:void main() {}"
    assert compiled == ["void main() {}"]

    with pytest.raises(KeyError, match="Unknown shader 'missing'"):
        library.get("missing")

    missing_library = shaders_mod.ShaderLibrary(FakeMGLContext(), base_path=tmp_path)
    monkeypatch.setattr(
        shaders_mod.registry,
        "shader_files",
        lambda: (SimpleNamespace(name="missing_file", path="absent.glsl"),),
    )
    with pytest.raises(FileNotFoundError, match="no file exists there"):
        missing_library.get("missing_file")


def test_run_shader_binds_uniforms_and_image_inputs():
    uniforms = []
    bind_calls = []
    run_calls = []

    class FakeShader:
        def __setitem__(self, key, value):
            uniforms.append((key, value))

        def run(self, *local_size):
            run_calls.append(local_size)

    class FakeTexture:
        def __init__(self):
            self.bindings = []

        def bind_to_image(self, location, read, write):
            bind_calls.append((location, read, write))

    class FakeCtx:
        local_size = (2, 3, 1)

        def get_shader(self, shader_name):
            assert shader_name == "demo"
            return FakeShader()

    shader = operations_mod.run_shader(
        FakeCtx(),
        "demo",
        uniforms={"alpha": 1},
        image_bindings=[(FakeTexture(), 0, False, True), (FakeTexture(), 1, True, False)],
    )

    assert uniforms == [("alpha", 1)]
    assert bind_calls == [(0, False, True), (1, True, False)]
    assert run_calls == [(2, 3, 1)]
    assert isinstance(shader, FakeShader)


def test_texture_counter_and_display_helpers(monkeypatch, capsys):
    counter = texture_utils_mod.Counter(2)
    counter += 3
    counter -= 1

    assert int(counter) == 4

    monkeypatch.setattr(texture_utils_mod, "tex_registry", counter)
    texture_utils_mod.decrease_tex_registry()
    assert int(texture_utils_mod.tex_registry) == 3

    calls = []

    def fake_frombytes(mode, size, data, _raw):
        calls.append((mode, size, data))
        return _FakeTexture(size=size, components=3 if mode == "RGB" else 4, payload=data)

    monkeypatch.setattr(texture_utils_mod.Image, "frombytes", fake_frombytes)

    texture_utils_mod.show_texture(_FakeTexture(size=(1, 1), components=3, payload=b"\x01\x02\x03"))
    texture_utils_mod.show_texture(_FakeTexture(size=(1, 1), components=4, payload=b"\x01\x02\x03\x04"))

    assert [call[0] for call in calls] == ["RGB", "RGBA"]

    with pytest.raises(NotImplementedError, match="not implemented"):
        texture_utils_mod.show_texture(_FakeTexture(size=(1, 1), components=2, payload=b"\x00\x00"))


def test_utilities_cover_rgb_normalization_and_collinearity():
    assert rgb_col(255, 0, 0, 128) == (1.0, 0.0, 0.0, 128 / 255)
    assert collinear(0, 0, 1, 1, 2, 2)
    assert not collinear(0, 0, 1, 2, 2, 5)


def test_glyph_rejects_multi_character_input():
    with pytest.raises(ValueError, match="exactly one character"):
        glyph("ab", 0.65, 25, 25)


def test_texture_namespace_aliases_cover_boolean_operators():
    left = sdf.circle((8, 8), 5)
    right = sdf.circle((10, 8), 5)

    assert (left | right).op == "union"
    assert (left & right).op == "intersection"
    assert (left - right).op == "subtract"
    assert left.intersect(right).op == "intersection"


def test_color_space_conversion_helpers_are_idempotent_when_already_in_target_mode():
    scene = color.clear("#ffffff")

    rgb_scene = scene.to_rgb()
    lab_scene = scene.to_lab()

    assert rgb_scene.mode == ColorSpaceMode.RGB
    assert lab_scene.mode == ColorSpaceMode.LAB
    assert rgb_scene.to_rgb() is rgb_scene
    assert lab_scene.to_lab() is lab_scene


def test_renderer_color_resolution_handles_hex_shorthand():
    renderer = Renderer(_fake_context())

    assert renderer._resolve_color("#fff") == (1.0, 1.0, 1.0, 1.0)
    assert renderer._resolve_color((1, 2, 3)) == (1, 2, 3)
