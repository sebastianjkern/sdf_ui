from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import sdf_ui.anim as anim_mod
import sdf_ui.ascii as ascii_mod
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

    ascii_mod.convert_image_to_ascii(
        image, cols=1, scale=1.0, more_levels=False, invert=False, enhance=False
    )

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

    texture = SimpleNamespace(
        tex=_FakeTexture(size=(2, 2), components=4, payload=b"\x00" * 16)
    )

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
    monkeypatch.setattr(
        anim_mod, "logger", lambda: SimpleNamespace(info=log_messages.append)
    )

    output_name = str(tmp_path / "movie")
    anim_mod.convert_to_video(output_name, ["frame1.png", "frame2.png"])

    assert len(written) == 2
    assert all(frame.shape == (10, 20, 3) for frame in written)
    assert releases == [output_name + ".mp4"]
    assert log_messages == ["Finished writing video"]


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
    renderer = Renderer(object())

    assert renderer._resolve_color("#fff") == (1.0, 1.0, 1.0, 1.0)
    assert renderer._resolve_color((1, 2, 3)) == (1, 2, 3)
