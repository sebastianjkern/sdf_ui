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
RIB_WAVE_AMOUNT = 0.64
RIB_WAVE_FREQUENCY = 7.6
RIB_CROSS_AMOUNT = 0.26
RIB_CROSS_PHASE = 1.9
RIB_SURFACE_AMOUNT = 0.16
RIB_LIFT_AMOUNT = 0.12
RIB_DEPTH_FACE_WEIGHT = 0.62
RIB_DEPTH_SIDE_WEIGHT = 0.38
RIB_BASE_REFRACTION = 0.043
RIB_REFRACTION_GAIN = 0.073
RIB_REFRACTION_CENTER = 0.70
RIB_REFRACTION_WIDTH = 0.29
RIB_DYNAMIC_GAIN = 0.18
RIB_PIXEL_SHIFT = 0.005
FOLD_HEIGHT = 0.78
FOLD_SLOPE = 0.22
FOLD_SLOPE_RATE = 4.8
FOLD_CENTER = 0.51
FOLD_WAVE_AMOUNT = 0.049
FOLD_WAVE_RATE = 5.1
FOLD_SWAY_AMOUNT = 0.022
FOLD_SWAY_RATE = 4.4
FOLD_LIFT_AMOUNT = 0.012
SHADOW_CENTER = 0.21
SHADOW_SLOPE = 0.18
SHADOW_WIDTH = 0.092
SHADOW_X_CENTER = 0.03
SHADOW_X_WIDTH = 0.34
SHADOW_MIX = 0.81
BRUISE_CENTER = 0.57
BRUISE_WAVE = 0.13
BRUISE_RATE = 3.9
BRUISE_WIDTH = 0.093
BRUISE_BASE = 0.50
BRUISE_X_GAIN = 0.49
BRUISE_X_CENTER = 0.45
BRUISE_X_WIDTH = 0.14
FOLD_EDGE_WIDTH = 0.021
FOLD_GLOW_OFFSET = 0.018
FOLD_GLOW_WIDTH = 0.045
RIGHT_GLOW_CENTER = 0.97
RIGHT_GLOW_WIDTH = 0.29
RIGHT_GLOW_Y_WIDTH = 0.17
UPPER_EDGE_HEIGHT = 0.064
UPPER_EDGE_SLOPE = 0.25
UPPER_EDGE_SWAY = 0.025
UPPER_EDGE_WIDTH = 0.008
UPPER_WINDOW_START = 0.76
UPPER_WINDOW_WIDTH = 0.10
UPPER_GLOW_WIDTH = 0.04
UPPER_WINDOW_GLOW_WIDTH = 0.14
LEFT_EDGE_HEIGHT = 0.71
LEFT_EDGE_SLOPE = 0.48
LEFT_EDGE_WAVE = 0.027
LEFT_EDGE_SWAY = 0.6
LEFT_EDGE_LIFT = 0.025
LEFT_EDGE_CLIP = 0.36
LEFT_EDGE_WIDTH = 0.047
LEFT_HOT_GLOW_WIDTH = 0.009
LEFT_WARM_GLOW_WIDTH = 0.036
LOWER_STREAK_HEIGHT = 0.94
LOWER_STREAK_SLOPE = 0.09
LOWER_STREAK_SWAY = 0.035
LOWER_STREAK_WIDTH = 0.062
LOWER_STREAK_BASE = 0.77
LOWER_STREAK_VARIATION = 0.23
LOWER_STREAK_RATE = 26
COOL_BAND_OFFSET = 0.094
COOL_BAND_WIDTH = 0.096
COOL_BAND_X_CENTER = 0.64
COOL_BAND_X_WIDTH = 0.23
COOL_BAND_EDGE_WIDTH = 0.029
BLUE_BAND_HEIGHT = 0.55
BLUE_BAND_WAVE = 0.023
BLUE_BAND_RATE = 8
BLUE_BAND_WIDTH = 0.033
BLUE_LEFT_GAIN = 0.62
BLUE_LEFT_CENTER = 0.18
BLUE_LEFT_WIDTH = 0.074
BLUE_RIGHT_GAIN = 1.3
BLUE_RIGHT_CENTER = 0.99
BLUE_RIGHT_WIDTH = 0.066
SIDE_GLOW_WIDTH = 0.017
COPPER_BAND_HEIGHT = 0.99
COPPER_BAND_WIDTH = 0.024
COPPER_X_CENTER = 0.94
COPPER_X_WIDTH = 0.18
SURFACE_SHADING_BASE = 0.12
SURFACE_SHADING_AMOUNT = 0.10
SURFACE_SHADING_CENTER = 0.66
SURFACE_SHADING_WIDTH = 0.29


def _falloff(distance):
    return np.exp(-0.5 * distance**2)


def _feather(distance, width):
    t = np.clip(0.5 + distance / (2 * width), 0, 1)
    return t * t * (3 - 2 * t)


def _refract(x, y, rng, sway, lift):
    """Bend the reflection through ribs with slightly different focal lengths."""
    grooves = (
        35 * x
        + RIB_WAVE_AMOUNT * np.sin(RIB_WAVE_FREQUENCY * x)
        + RIB_CROSS_AMOUNT * np.sin(22 * x + RIB_CROSS_PHASE * y)
        + RIB_SURFACE_AMOUNT * np.sin(4 * y + 2 * x)
    )
    cell = np.floor(grooves + 0.5).astype(int)
    first_cell = cell.min()
    depths = rng.uniform(0.68, 1.32, cell.max() - first_cell + 1)
    depth = depths[cell - first_cell]
    grooves += RIB_SURFACE_AMOUNT * sway * np.sin(6 * x - 4 * y)
    grooves += RIB_LIFT_AMOUNT * lift * np.cos(7 * y)
    face = np.cos(tau * grooves)
    side = np.sin(tau * grooves)
    facets = (RIB_DEPTH_FACE_WEIGHT * face + RIB_DEPTH_SIDE_WEIGHT * face**3) * depth
    strength = RIB_BASE_REFRACTION + RIB_REFRACTION_GAIN * _falloff(
        (y - RIB_REFRACTION_CENTER) / RIB_REFRACTION_WIDTH
    )
    strength = strength * (1 + RIB_DYNAMIC_GAIN * sway * np.cos(4 * y - 2 * x))
    return x + RIB_PIXEL_SHIFT * side, y + strength * facets, face, side


def _reflected_light(x, y, face, side, sway, lift):
    """Paint broad color fields and the light sources reflected by the glass."""
    rgb = np.empty((*y.shape, 3), dtype=np.float32)
    rgb[:] = (0.61, 0.055, 0.026)
    fold = _paint_fold(rgb, x, y, sway, lift)
    left_edge = _paint_edge_lights(rgb, x, y, sway, lift)
    _paint_reflection_streaks(rgb, x, y, face, sway, fold, left_edge)
    _paint_color_bands(rgb, x, y, face, side, fold)
    return rgb


def _paint_fold(rgb, x, y, sway, lift):
    fold = FOLD_HEIGHT - FOLD_SLOPE * np.tanh(FOLD_SLOPE_RATE * (x - FOLD_CENTER))
    fold += FOLD_WAVE_AMOUNT * np.sin(FOLD_WAVE_RATE * x)
    fold += (
        FOLD_SWAY_AMOUNT * sway * np.sin(FOLD_SWAY_RATE * x) + FOLD_LIFT_AMOUNT * lift
    )
    shadow = _falloff((y - (SHADOW_CENTER + SHADOW_SLOPE * x)) / SHADOW_WIDTH)
    shadow *= _falloff((x - SHADOW_X_CENTER) / SHADOW_X_WIDTH)
    _mix(rgb, (0.17, 0.025, 0.008), SHADOW_MIX * shadow)
    bruise = _falloff(
        (y - (BRUISE_CENTER + BRUISE_WAVE * np.sin(BRUISE_RATE * x))) / BRUISE_WIDTH
    )
    bruise *= BRUISE_BASE + BRUISE_X_GAIN * _falloff(
        (x - BRUISE_X_CENTER) / BRUISE_X_WIDTH
    )
    _mix(rgb, (0.071, 0.013, 0.033), bruise)
    _mix(rgb, (0.015, 0.014, 0.011), _feather(y - fold, FOLD_EDGE_WIDTH))
    _glow(
        rgb,
        (0.29, 0.029, 0.009),
        _falloff((y - fold + FOLD_GLOW_OFFSET) / FOLD_GLOW_WIDTH),
    )
    _glow(
        rgb,
        (0.27, 0.042, 0.017),
        _falloff((x - RIGHT_GLOW_CENTER) / RIGHT_GLOW_WIDTH)
        * _falloff(y / RIGHT_GLOW_Y_WIDTH),
    )
    return fold


def _paint_edge_lights(rgb, x, y, sway, lift):
    upper = y - (
        UPPER_EDGE_HEIGHT + UPPER_EDGE_SLOPE * (x - 1) + UPPER_EDGE_SWAY * sway
    )
    upper_light = _feather(-upper, UPPER_EDGE_WIDTH) * _feather(
        x - UPPER_WINDOW_START, UPPER_WINDOW_WIDTH
    )
    _glow(rgb, (3.5, 2.1, 1.32), upper_light)
    upper_glow = _falloff(upper / UPPER_GLOW_WIDTH) * _feather(
        x - UPPER_WINDOW_START, UPPER_WINDOW_GLOW_WIDTH
    )
    _glow(rgb, (0.86, 0.25, 0.08), upper_glow)
    left = y - (
        LEFT_EDGE_HEIGHT
        + LEFT_EDGE_SLOPE * x
        + LEFT_EDGE_WAVE * np.sin(13 * x + LEFT_EDGE_SWAY * sway)
        - LEFT_EDGE_LIFT * lift
    )
    left_edge = _feather(LEFT_EDGE_CLIP - x, LEFT_EDGE_WIDTH)
    _glow(rgb, (2.8, 1.85, 1.12), left_edge * _falloff(left / LEFT_HOT_GLOW_WIDTH))
    _glow(rgb, (0.82, 0.23, 0.085), left_edge * _falloff(left / LEFT_WARM_GLOW_WIDTH))
    return left_edge


def _paint_reflection_streaks(rgb, x, y, face, sway, fold, left_edge):
    lens = np.maximum(face, 0) ** 22
    lower_left = (
        _falloff(
            (
                y
                - (
                    LOWER_STREAK_HEIGHT
                    + LOWER_STREAK_SLOPE * x
                    + LOWER_STREAK_SWAY * sway
                )
            )
            / LOWER_STREAK_WIDTH
        )
        * left_edge
    )
    lower_left *= LOWER_STREAK_BASE + LOWER_STREAK_VARIATION * np.cos(
        LOWER_STREAK_RATE * x
    )
    _glow(rgb, (2.65, 1.75, 1.03), lens * lower_left)
    _glow(rgb, (0.63, 0.13, 0.047), np.maximum(face, 0) ** 4 * lower_left)


def _paint_color_bands(rgb, x, y, face, side, fold):
    lens = np.maximum(face, 0) ** 22
    cool_band = _falloff((y - fold - COOL_BAND_OFFSET) / COOL_BAND_WIDTH)
    cool_band *= _falloff((x - COOL_BAND_X_CENTER) / COOL_BAND_X_WIDTH)
    cool_band *= _feather(y - fold, COOL_BAND_EDGE_WIDTH)
    _glow(rgb, (0.12, 0.34, 0.23), lens * cool_band)
    blue_envelope = _falloff(
        (y - (BLUE_BAND_HEIGHT - BLUE_BAND_WAVE * np.sin(BLUE_BAND_RATE * x)))
        / BLUE_BAND_WIDTH
    )
    blue = blue_envelope * (
        BLUE_LEFT_GAIN * _falloff((x - BLUE_LEFT_CENTER) / BLUE_LEFT_WIDTH)
        + BLUE_RIGHT_GAIN * _falloff((x - BLUE_RIGHT_CENTER) / BLUE_RIGHT_WIDTH)
    )
    _glow(rgb, (0.33, 0.44, 0.58), lens * blue)
    _glow(
        rgb,
        (0.17, 0.15, 0.12),
        np.maximum(side, 0) ** 45 * _falloff((y - fold) / SIDE_GLOW_WIDTH),
    )
    copper = _falloff((y - COPPER_BAND_HEIGHT) / COPPER_BAND_WIDTH)
    copper *= _falloff((x - COPPER_X_CENTER) / COPPER_X_WIDTH)
    _glow(rgb, (0.46, 0.13, 0.06), copper * lens)


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
        1
        - (
            SURFACE_SHADING_BASE
            + SURFACE_SHADING_AMOUNT
            * _falloff((y - SURFACE_SHADING_CENTER) / SURFACE_SHADING_WIDTH)
        )
        * np.maximum(-side, 0) ** 5
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
    ffmpeg = _ffmpeg_binary()
    codec = _video_codec(ffmpeg)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        video = Path(staging) / "loop.mp4"
        command = _encoder_command(ffmpeg, codec, video)
        with Canvas(SIZE) as canvas:
            image = canvas.rgba32f()
            with ColorTexture(tex=image, context=canvas, mode="LINEAR") as source:
                scene = _finish(source)
                with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                    _stream_animation(encoder, canvas, image, scene, seed)
        video.replace(output)
    return output


def _ffmpeg_binary():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Animation export requires ffmpeg on PATH")
    return ffmpeg


def _video_codec(ffmpeg):
    encoders = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], text=True)
    available = {
        line.split()[1] for line in encoders.splitlines() if line.startswith(" V")
    }
    if "libx264" in available:
        return ["-c:v", "libx264", "-crf", "17"]
    if "libopenh264" in available:
        return ["-c:v", "libopenh264", "-b:v", "7M"]
    return ["-c:v", "mpeg4", "-q:v", "2"]


def _encoder_command(ffmpeg, codec, video):
    width, height = SIZE
    return [
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


def _stream_animation(encoder, canvas, image, scene, seed):
    try:
        with encoder.stdin as stream:
            for frame in range(FRAMES):
                pixels = molten_reflection(seed=seed, progress=frame / FRAMES)
                image.write(pixels[::-1])
                with canvas.render(scene) as rendered:
                    stream.write(rendered.read()[::-1].tobytes())
    except BrokenPipeError as exc:
        raise RuntimeError("FFmpeg stopped while encoding the loop") from exc
    if encoder.wait() != 0:
        raise RuntimeError("FFmpeg could not encode the loop")


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
