"""Render an imaginary light field seen through fluted red glass."""

import argparse
import shutil
import subprocess
from math import cos, sin, tau
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sdf_ui import Canvas, ColorTexture

SIZE = (768, 576)
OUTPUT = Path("out/examples/molten_reflection/molten_reflection.png")
VIDEO = OUTPUT.with_suffix(".mp4")
DEFAULT_SEED = 27
FPS = 24
FRAMES = 6 * FPS


def _falloff(distance):
    return np.exp(-0.5 * distance**2)


def _feather(distance, width):
    t = np.clip(0.5 + distance / (2 * width), 0, 1)
    return t * t * (3 - 2 * t)


def _refract(x, y, rng, sway, lift):
    """Bend the reflection through ribs with slightly different focal lengths."""
    grooves = (
        35 * x
        + 0.64 * np.sin(7.6 * x)
        + 0.26 * np.sin(22 * x + 1.9 * y)
        + 0.16 * np.sin(4 * y + 2 * x)
    )
    cell = np.floor(grooves + 0.5).astype(int)
    first_cell = cell.min()
    depths = rng.uniform(0.68, 1.32, cell.max() - first_cell + 1)
    depth = depths[cell - first_cell]
    grooves += 0.16 * sway * np.sin(6 * x - 4 * y) + 0.12 * lift * np.cos(7 * y)
    face = np.cos(tau * grooves)
    side = np.sin(tau * grooves)
    facets = (0.62 * face + 0.38 * face**3) * depth
    strength = 0.043 + 0.073 * _falloff((y - 0.70) / 0.29)
    strength = strength * (1 + 0.18 * sway * np.cos(4 * y - 2 * x))
    return x + 0.005 * side, y + strength * facets, face, side


def _reflected_light(x, y, face, side, sway, lift):
    """Paint broad color fields and the light sources reflected by the glass."""
    red = (0.61, 0.055, 0.026)
    ember = (0.17, 0.025, 0.008)
    bank = (0.015, 0.014, 0.011)
    violet = (0.071, 0.013, 0.033)

    rgb = np.empty((*y.shape, 3), dtype=np.float32)
    rgb[:] = red

    # The red surface folds over a dark bank, steepest near the center.
    fold = 0.78 - 0.22 * np.tanh(4.8 * (x - 0.51)) + 0.049 * np.sin(5.1 * x)
    fold += 0.022 * sway * np.sin(4.4 * x) + 0.012 * lift
    shadow = _falloff((y - (0.21 + 0.18 * x)) / 0.092)
    shadow *= _falloff((x - 0.03) / 0.34)
    _mix(rgb, ember, 0.81 * shadow)

    bruise = _falloff((y - (0.57 + 0.13 * np.sin(3.9 * x))) / 0.093)
    bruise *= 0.50 + 0.49 * _falloff((x - 0.45) / 0.14)
    _mix(rgb, violet, bruise)

    _mix(rgb, bank, _feather(y - fold, 0.021))
    _glow(rgb, (0.29, 0.029, 0.009), _falloff((y - fold + 0.018) / 0.045))
    _glow(rgb, (0.27, 0.042, 0.017), _falloff((x - 0.97) / 0.29) * _falloff(y / 0.17))

    # A window grazes the upper edge. Another breaks up along the lower left.
    upper = y - (0.064 + 0.25 * (x - 1) + 0.025 * sway)
    upper_light = _feather(-upper, 0.008) * _feather(x - 0.76, 0.10)
    _glow(rgb, (3.5, 2.1, 1.32), upper_light)
    _glow(rgb, (0.86, 0.25, 0.08), _falloff(upper / 0.04) * _feather(x - 0.76, 0.14))

    left = y - (0.71 + 0.48 * x + 0.027 * np.sin(13 * x + 0.6 * sway) - 0.025 * lift)
    left_edge = _feather(0.36 - x, 0.047)
    _glow(rgb, (2.8, 1.85, 1.12), left_edge * _falloff(left / 0.009))
    _glow(rgb, (0.82, 0.23, 0.085), left_edge * _falloff(left / 0.036))

    # Only the glass faces aligned with the light make long, bright streaks.
    lens = np.maximum(face, 0) ** 22
    lower_left = _falloff((y - (0.94 + 0.09 * x + 0.035 * sway)) / 0.062) * left_edge
    lower_left *= 0.77 + 0.23 * np.cos(26 * x)
    _glow(rgb, (2.65, 1.75, 1.03), lens * lower_left)
    _glow(rgb, (0.63, 0.13, 0.047), np.maximum(face, 0) ** 4 * lower_left)

    cool_band = _falloff((y - fold - 0.094) / 0.096)
    cool_band *= _falloff((x - 0.64) / 0.23) * _feather(y - fold, 0.029)
    _glow(rgb, (0.12, 0.34, 0.23), lens * cool_band)

    blue = _falloff((y - (0.55 - 0.023 * np.sin(8 * x))) / 0.033) * (
        0.62 * _falloff((x - 0.18) / 0.074) + 1.3 * _falloff((x - 0.99) / 0.066)
    )
    _glow(rgb, (0.33, 0.44, 0.58), lens * blue)
    _glow(
        rgb,
        (0.17, 0.15, 0.12),
        np.maximum(side, 0) ** 45 * _falloff((y - fold) / 0.017),
    )
    copper = _falloff((y - 0.99) / 0.024) * _falloff((x - 0.94) / 0.18)
    _glow(rgb, (0.46, 0.13, 0.06), copper * lens)
    return rgb


def _mix(rgb, color, amount):
    rgb += (np.asarray(color, dtype=np.float32) - rgb) * amount[..., None]


def _glow(rgb, color, amount):
    rgb += np.asarray(color, dtype=np.float32) * amount[..., None]


def molten_reflection(size=SIZE, *, seed=DEFAULT_SEED, progress=0.0):
    """Return a top-down linear RGBA frame. Progress wraps every full turn."""
    width, height = size
    x = (np.arange(width, dtype=np.float32)[None, :] + 0.5) / width
    y = (np.arange(height, dtype=np.float32)[:, None] + 0.5) / height
    rng = np.random.default_rng(seed)
    phase = tau * (progress % 1)
    sway, lift = sin(phase), 1 - cos(phase)
    refracted_x, refracted_y, face, side = _refract(x, y, rng, sway, lift)
    rgb = _reflected_light(refracted_x, refracted_y, face, side, sway, lift)
    # The darker side of each flute also attenuates the underlying color.
    shading = (
        1 - (0.12 + 0.10 * _falloff((y - 0.66) / 0.29)) * np.maximum(-side, 0) ** 5
    )
    rgb *= shading[..., None]
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    return rgba


def render(output=OUTPUT, *, seed=DEFAULT_SEED):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pixels = molten_reflection(seed=seed)
    with Canvas(SIZE) as canvas:
        image = canvas.rgba32f()
        with ColorTexture(tex=image, context=canvas, mode="LINEAR") as source:
            image.write(pixels[::-1])
            canvas.save(_finish(source), output)
    return output


def _finish(source):
    return (
        source.bloom(threshold=0.54, amount=0.44, passes=2)
        .film_grain(0.08, size=1.5)
        .finish(tone=0.26, dither=0.65)
    )


def render_animation(output=VIDEO, *, seed=DEFAULT_SEED):
    """Stream a six-second loop to FFmpeg while reusing one canvas."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Animation export requires ffmpeg on PATH")
    encoders = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], text=True)
    available = {
        line.split()[1] for line in encoders.splitlines() if line.startswith(" V")
    }
    if "libx264" in available:
        codec = ["-c:v", "libx264", "-crf", "17"]
    elif "libopenh264" in available:
        codec = ["-c:v", "libopenh264", "-b:v", "7M"]
    else:
        codec = ["-c:v", "mpeg4", "-q:v", "2"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = SIZE
    with TemporaryDirectory(dir=output.parent) as staging:
        video = Path(staging) / "loop.mp4"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(FPS),
            "-i",
            "pipe:0",
            "-an",
            *codec,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ]
        with Canvas(SIZE) as canvas:
            image = canvas.rgba32f()
            with ColorTexture(tex=image, context=canvas, mode="LINEAR") as source:
                scene = _finish(source)
                with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                    try:
                        with encoder.stdin as stream:
                            for frame in range(FRAMES):
                                # Omit the repeated endpoint at progress=1.
                                pixels = molten_reflection(
                                    seed=seed, progress=frame / FRAMES
                                )
                                image.write(pixels[::-1])
                                with canvas.render(scene) as rendered:
                                    stream.write(rendered.read()[::-1].tobytes())
                    except BrokenPipeError as exc:
                        raise RuntimeError(
                            "FFmpeg stopped while encoding the loop"
                        ) from exc
                    if encoder.wait() != 0:
                        raise RuntimeError("FFmpeg could not encode the loop")
        video.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Override the default output path")
    parser.add_argument(
        "--animate", action="store_true", help="Render a six-second MP4 loop"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Vary the depth of the glass ribs",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.animate:
        print(render_animation(args.output or VIDEO, seed=args.seed))
    else:
        print(render(args.output or OUTPUT, seed=args.seed))


if __name__ == "__main__":
    main()
