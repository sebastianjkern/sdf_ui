"""Render a one-minute narrated visual intuition for Lagrange multipliers.

All visual layers and frames are rendered by SDF UI. The optional ``math``
extra typesets equations; ``espeak-ng`` and ``ffmpeg`` create the narration
and final MP4. Run from the repository root with::

    uv run --extra math python -m examples.lagrange_intuition_short
"""

import argparse
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from time import monotonic, strftime

from sdf_ui import Canvas, Plot, color, sdf
from sdf_ui.widgets import Panel, progress_bar

OUTPUT = Path("out/learning/lagrange_intuition_short.mp4")
SIZE = (1280, 720)
FPS = 24
DURATION = 60.0

BG = "#10151f"
PANEL = "#171e2b"
GRID = (0.17, 0.21, 0.28, 0.72)
WHITE = "#f4f1e9"
MUTED = "#a4adbc"
TEAL = "#62dfcf"
GOLD = "#ffc978"
PURPLE = "#b9a5ff"
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)

# The plot uses the lower-left canvas origin used by SDF UI geometry.
PLOT_ORIGIN = (116.0, 92.0)
PLOT_SCALE = 454.0
PLOT = Plot((*PLOT_ORIGIN, 1.2 * PLOT_SCALE, 1.2 * PLOT_SCALE), (0, 1.2), (0, 1.2))

NARRATION = (
    "Can you maximize a function if you can only move along a curve?",
    "Think of f equals x times y as a landscape, with x squared plus y squared equals one as a circular rail.",
    "The curves x times y equals c are level sets. As c grows, they move uphill. The highest reachable one just touches the rail.",
    "At that touch, the curves share a tangent. Gradients point perpendicular to curves, so the objective and constraint gradients are parallel.",
    "That is the Lagrange condition: gradient f equals lambda gradient g. Here, y comma x equals lambda times two x comma two y.",
    "Together with the circle, this gives x equals y equals one over root two, and the maximum product is one half.",
    "Lambda is the exchange rate: if the squared-radius budget grows a little, the best product grows by lambda times that amount.",
)

STORY = (
    (
        "THE QUESTION",
        "A function can rise in many directions.\nWhat if your path is fixed?",
        (r"f(x,y)=xy", r"x^2+y^2=1"),
        "",
    ),
    (
        "SET THE EXAMPLE",
        "Move along the circular rail while\nseeking the largest product x times y.",
        (r"f(x,y)=xy", r"g(x,y)=x^2+y^2=1"),
        "",
    ),
    (
        "LEVEL SETS",
        "Each curve xy=c has one constant\nvalue. Larger c means a higher contour.",
        (r"xy=c", r"c\ \text{increases}\ \nearrow"),
        "",
    ),
    (
        "THE LAST TOUCH",
        "The best reachable contour just\nkisses the constraint. It cannot cross.",
        (r"xy=\frac{1}{2}",),
        "Tangent at the maximum.",
    ),
    (
        "WHY GRADIENTS?",
        "A gradient points normal to its\ncontour. At a shared tangent, the\ntwo normals are parallel.",
        (r"\nabla f\ \parallel\ \nabla g", r"\nabla f=\lambda\nabla g"),
        "",
    ),
    (
        "SOLVE THE CONDITION",
        "The multiplier balances the two\ngradients. Solve this with the\noriginal constraint.",
        (r"(y,x)=\lambda(2x,2y)", r"x=y=\frac{1}{\sqrt{2}}", r"f_{\max}=\frac{1}{2}"),
        "At the endpoints, xy=0. The\ninterior point gives the maximum 1/2.",
    ),
    (
        "THE MULTIPLIER",
        "Lambda is the exchange rate: how\nmuch the best value changes when\nthe constraint loosens.",
        (r"x^2+y^2=b", r"f_{\max}(b)=\frac{b}{2}", r"\lambda=\frac{df_{\max}}{db}=\frac{1}{2}"),
        "The multiplier is the marginal value\nof a little extra budget.",
    ),
)


def _log(message):
    """Print a timestamped progress message immediately."""
    print(f"[{strftime('%H:%M:%S')}] {message}", flush=True)


def _speak_clips(directory, speed=150):
    """Synthesize narration clips and return their timings and sample format."""
    executable = shutil.which("espeak-ng")
    if executable is None:
        raise RuntimeError("espeak-ng is required to render the narrated short")

    clips = []
    durations = []
    wave_params = None
    _log(f"Synthesizing {len(NARRATION)} narration clips with espeak-ng")
    for index, sentence in enumerate(NARRATION):
        _log(f"Narration {index + 1}/{len(NARRATION)}: starting synthesis")
        clip_started = monotonic()
        path = directory / f"voice_{index:02d}.wav"
        subprocess.run(
            [
                executable,
                "-v",
                "en-us+m3",
                "-s",
                str(speed),
                "-p",
                "44",
                "-a",
                "165",
                "-w",
                str(path),
                sentence,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        with wave.open(str(path), "rb") as clip:
            params = clip.getparams()
            samples = clip.readframes(clip.getnframes())
        if wave_params is None:
            wave_params = params
        elif (params.nchannels, params.sampwidth, params.framerate) != (
            wave_params.nchannels,
            wave_params.sampwidth,
            wave_params.framerate,
        ):
            raise RuntimeError("speech clips were synthesized with mismatched formats")
        clips.append(samples)
        clip_duration = (
            len(samples)
            / (params.framerate * params.nchannels * params.sampwidth)
        )
        durations.append(clip_duration)
        _log(
            f"Narration {index + 1}/{len(NARRATION)}: "
            f"{clip_duration:.1f}s audio ready in {monotonic() - clip_started:.1f}s"
        )

    speech_duration = sum(durations)
    if speech_duration >= DURATION - 1.0:
        raise RuntimeError(
            f"Narration is too long ({speech_duration:.1f}s); increase speech speed"
        )

    pause = (DURATION - speech_duration) / len(clips)
    _log(
        f"Narration totals {speech_duration:.1f}s; inserting "
        f"{pause:.1f}s pauses to fill {DURATION:.0f}s"
    )
    starts = []
    silence_frames = round(pause * wave_params.framerate)
    silence = bytes(
        silence_frames * wave_params.nchannels * wave_params.sampwidth
    )
    audio = bytearray()
    elapsed = 0.0
    for samples, clip_duration in zip(clips, durations):
        starts.append(elapsed)
        audio.extend(samples)
        audio.extend(silence)
        elapsed += clip_duration + pause

    narration_path = directory / "lagrange_intuition_narration.wav"
    with wave.open(str(narration_path), "wb") as output:
        output.setparams(wave_params._replace(nframes=0))
        output.writeframes(audio)
    return narration_path, starts, durations, pause


def _pixel(point):
    return PLOT.point(*point)


def _overlay(scene, layer, *, cache=True):
    composite = scene.alpha_overlay(layer)
    return composite if cache else composite.uncached()


def _segment(a, b, radius, tint, *, cache=True):
    segment = sdf.segment(a, b, radius)
    if not cache:
        segment = segment.uncached()
    layer = segment.fill(tint, TRANSPARENT)
    return layer if cache else layer.uncached()


def _fill_type(node, tint, *, inflate=0.0):
    # Keep the ink color through the antialiased edge so straight-alpha
    # compositing attenuates the glyph only once.
    return node.fill(tint, f"{tint}00", inflate=inflate, inner=-0.5, outer=0.5)


def _text(
    value,
    box,
    *,
    size=22,
    tint=WHITE,
    align="left",
    valign="middle",
    cache=True,
):
    text_node = sdf.text(
        value,
        size=size,
        box=box,
        align=align,
        valign=valign,
        overflow="clip",
    )
    if not cache:
        text_node = text_node.uncached()
    layer = _fill_type(text_node, tint, inflate=0.2)
    return layer if cache else layer.uncached()


def _formula(value, center, size=34, tint=GOLD):
    node = sdf.math(value, center=center, height=None, font_size=size, inflate=0.2)
    return _fill_type(node, tint)


def _level_curve(level, tint, width=2.2, *, cache=True):
    # Keep the endpoints a little inside the viewport, clear of the labels.
    curve = PLOT.curve(lambda x: level / x, domain=(level / 1.18, 1.18))
    if not cache:
        curve = curve.uncached()
    layer = curve.stroke(width, tint)
    return layer if cache else layer.uncached()


def _dot(center, radius, tint):
    return sdf.circle(center, radius).uncached().fill(tint, TRANSPARENT).uncached()


def _build_static_scenes():
    """Build common plot layers and the seven reusable explanation panels."""
    scene = color.clear(BG)

    # Main heading and a compact problem statement.
    scene = _overlay(
        scene,
        _text(
            "A GEOMETRIC GUIDE TO OPTIMIZATION",
            (66, 665, 740, 28),
            size=17,
            tint=MUTED,
        ),
    )
    scene = _overlay(
        scene,
        _formula(
            r"\text{Maximize }f(x,y)=xy\quad\text{on}\quad x^2+y^2=1",
            (640, 641),
            size=34,
            tint=WHITE,
        ),
    )

    # Plot grid, axes, tick labels, and the unit-circle constraint.
    scene = _overlay(
        scene,
        PLOT.axes(
            x_ticks=(0, 0.5, 1),
            y_ticks=(0.5, 1),
            axis_color=MUTED,
            grid_color=GRID,
            x_label="x",
            y_label="y",
        ),
    )

    constraint = sdf.arc(PLOT_ORIGIN, PLOT_SCALE, 0.0, math.pi / 2).stroke(
        3.0, TEAL
    )
    scene = _overlay(scene, constraint)
    scene = _overlay(
        scene,
        _formula(r"x^2+y^2=1", _pixel((0.62, 1.04)), size=23, tint=TEAL),
    )
    scene = _overlay(
        scene,
        _text("constraint", (450, 100, 125, 25), size=14, tint=TEAL),
    )

    common = scene.cache("lagrange_short_plot")
    contour_scene = color.clear(TRANSPARENT)
    for level in (0.08, 0.16, 0.24, 0.32, 0.40, 0.46):
        contour = _level_curve(level, (1.0, 0.78, 0.47, 0.42), width=1.35)
        contour_scene = _overlay(contour_scene, contour)
    contour_scene = contour_scene.cache("lagrange_short_contours")

    card = Panel((700, 85, 520, 540), padding=(35, 24), radius=20)
    left, _, width, _ = card.content_box
    panels = []
    for index, (header, body, formulas, note) in enumerate(STORY):
        panel = card.layer(PANEL)
        panel = _overlay(
            panel,
            _text(header, (left, 585, width, 36), size=19, tint=TEAL),
        )
        panel = _overlay(
            panel,
            _text(body, (left, 425, width, 132), size=25, tint=WHITE, valign="top"),
        )
        for formula_index, expression in enumerate(formulas):
            tint = (GOLD, TEAL, PURPLE)[formula_index % 3]
            center_y = 330 - formula_index * 64
            panel = _overlay(
                panel,
                _formula(expression, (960, center_y), tint=tint),
            )
        if note:
            panel = _overlay(
                panel,
                _text(note, (left, 98, width, 62), size=18, tint=MUTED, valign="top"),
            )
        panels.append(panel.cache(f"lagrange_short_panel_{index}"))

    scenes = []
    for index, panel in enumerate(panels):
        base = _overlay(common, panel)
        if index >= 2:
            base = _overlay(base, contour_scene)
        scenes.append(base.cache(f"lagrange_short_section_{index}"))

    optimum = 1 / math.sqrt(2)
    short_end = _pixel((optimum + 0.20, optimum + 0.20))
    long_end = _pixel((optimum + 0.40, optimum + 0.40))
    gradient_labels = color.clear(TRANSPARENT)
    gradient_labels = _overlay(
        gradient_labels,
        _formula(r"\nabla f", (short_end[0] + 30, short_end[1] + 12), size=24, tint=GOLD),
    )
    gradient_labels = _overlay(
        gradient_labels,
        _formula(r"\nabla g", (long_end[0] + 30, long_end[1] + 12), size=24, tint=PURPLE),
    ).cache("lagrange_short_gradient_labels")
    return scenes, gradient_labels


def _section_at(time, starts):
    return max(index for index, start in enumerate(starts) if start <= time)


def _local_progress(time, section, starts, durations):
    return min(
        1.0,
        max(0.0, (time - starts[section]) / max(durations[section], 0.001)),
    )


def _frame_scene(time, starts, durations, static_scenes, gradient_labels):
    section = _section_at(time, starts)
    local = _local_progress(time, section, starts, durations)
    if section == 0:
        angle = 0.16
    elif section == 1:
        angle = 0.16 + 0.26 * local
    elif section == 2:
        angle = 0.42 + 0.28 * local
    elif section == 3:
        angle = 0.70 + (math.pi / 4 - 0.70) * local
    else:
        angle = math.pi / 4

    point = (math.cos(angle), math.sin(angle))
    level = 0.5 * math.sin(2 * angle)
    scene = static_scenes[section]

    if section >= 1:
        current = _level_curve(level, GOLD, width=3.0, cache=False)
        scene = _overlay(scene, current, cache=False)
    if section >= 3:
        optimum = 1 / math.sqrt(2)
        optimum_pixel = _pixel((optimum, optimum))
        ring = sdf.circle(optimum_pixel, 12).uncached().stroke(2, GOLD).uncached()
        tangent = _segment(
            _pixel((optimum - 0.24, optimum + 0.24)),
            _pixel((optimum + 0.24, optimum - 0.24)),
            1.5,
            PURPLE,
            cache=False,
        )
        scene = _overlay(scene, tangent, cache=False)
        scene = _overlay(scene, ring, cache=False)
        if section == 3 and level < 0.499:
            other_angle = math.pi / 2 - angle
            other_point = _pixel((math.cos(other_angle), math.sin(other_angle)))
            scene = _overlay(
                scene,
                _dot(other_point, 6, GOLD),
                cache=False,
            )
        if section >= 4:
            start = optimum_pixel
            short_end = _pixel((optimum + 0.20, optimum + 0.20))
            long_end = _pixel((optimum + 0.40, optimum + 0.40))
            scene = _overlay(
                scene, sdf.arrow(start, long_end, width=3.2).fill(PURPLE, TRANSPARENT),
                cache=False,
            )
            scene = _overlay(
                scene, sdf.arrow(start, short_end, width=3.2).fill(GOLD, TRANSPARENT),
                cache=False,
            )
            scene = _overlay(scene, gradient_labels, cache=False)

    point_pixel = _pixel(point)
    scene = _overlay(
        scene,
        _dot(point_pixel, 8, WHITE),
        cache=False,
    )
    if section >= 2:
        label_x = min(1.08, max(0.76, point[0] + 0.10))
        label_y = min(1.13, max(0.13, level / label_x + 0.05))
        scene = _overlay(
            scene,
            PLOT.annotate(
                (label_x, label_y),
                "xy = c",
                offset=(0, 7),
                size=16,
                tint=GOLD,
                cache=False,
            ),
            cache=False,
        )

    scene = _overlay(
        scene,
        progress_bar(
            (67.5, 32.5, 1145, 5), time / DURATION,
            track_color=GRID, fill_color=TEAL, cache=False,
        ),
        cache=False,
    )
    clock = f"{int(time // 60):01d}:{int(time % 60):02d}"
    scene = _overlay(
        scene,
        _text(
            clock,
            (1150, 8, 62, 20),
            size=13,
            tint=MUTED,
            align="right",
            cache=False,
        ),
        cache=False,
    )
    return scene


def _timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _write_subtitles(path, starts, durations):
    with Path(path).open("w", encoding="utf-8") as subtitles:
        for index, (text, start, duration) in enumerate(
            zip(NARRATION, starts, durations), start=1
        ):
            subtitles.write(
                f"{index}\n{_timestamp(start)} --> "
                f"{_timestamp(start + duration)}\n{text}\n\n"
            )


def _select_video_encoder():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the video")
    # FFmpeg builds differ by platform: minimal distro builds often omit
    # libx264 while retaining the built-in MPEG-4 Part 2 encoder.
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    available = {
        line.split()[1]
        for line in encoders.splitlines()
        if len(line.split()) >= 2 and line.split()[0][:1] in {"V", "."}
    }
    codec = next(
        (
            candidate
            for candidate in ("libx264", "libopenh264", "mpeg4")
            if candidate in available
        ),
        None,
    )
    if codec is None:
        raise RuntimeError(
            "This FFmpeg build has no supported video encoder "
            "(tried libx264, libopenh264, and mpeg4). Install an FFmpeg build "
            "with one of these encoders and rerun the render."
        )
    return ffmpeg, codec


def _encode_silent_video(frame_directory, output, ffmpeg, codec):
    _log(f"Encoding {FPS} fps silent MP4 from {frame_directory} with {codec}")
    encode_started = monotonic()
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                str(frame_directory / "frame_%06d.png"),
                "-c:v",
                codec,
                "-b:v",
                "5000k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        raise RuntimeError(
            f"FFmpeg failed to encode the silent MP4: {detail}"
        ) from error
    _log(f"Silent MP4 encoded in {monotonic() - encode_started:.1f}s: {output}")


def _mux_audio(video_path, audio_path, output_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to mux the narration into the short")
    _log(f"Muxing narration into {output_path}")
    mux_started = monotonic()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            str(DURATION),
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    _log(f"Audio mux completed in {monotonic() - mux_started:.1f}s")


def render_short(output=OUTPUT):
    """Render the narrated short with SDF UI, plus a poster and subtitles."""
    render_started = monotonic()
    ffmpeg, video_codec = _select_video_encoder()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    poster = output.with_name(output.stem + "_poster.png")
    subtitles = output.with_suffix(".srt")
    _log(
        f"Starting {SIZE[0]}x{SIZE[1]} SDF UI render: "
        f"{DURATION:.0f}s at {FPS} fps ({round(FPS * DURATION)} frames)"
    )
    _log(f"Outputs: video={output}, poster={poster}, subtitles={subtitles}")
    with tempfile.TemporaryDirectory(prefix="lagrange-short-") as temp_name:
        temp_dir = Path(temp_name)
        _log("Stage 1/5: synthesize narration")
        audio_path, starts, durations, pause = _speak_clips(temp_dir)
        _log("Stage 2/5: write subtitle file")
        _write_subtitles(subtitles, starts, durations)
        _log("Stage 3/5: build reusable SDF UI plot and section scenes")
        scene_started = monotonic()
        static_scenes, gradient_labels = _build_static_scenes()
        _log(f"Static SDF UI scenes built in {monotonic() - scene_started:.1f}s")
        silent_path = temp_dir / "lagrange_silent.mp4"
        frame_directory = temp_dir / "frames"
        frame_directory.mkdir()

        _log("Stage 4/5: render and save SDF UI frames")
        with Canvas(SIZE) as canvas:
            with canvas.session(cache={}) as renderer:
                poster_saved = False
                frame_count = round(FPS * DURATION)
                frame_started = monotonic()
                for frame_index in range(frame_count):
                    time = frame_index / FPS
                    section = _section_at(time, starts)
                    current_frame = frame_index + 1
                    _log(
                        f"Frame {current_frame}/{frame_count} "
                        f"({time:.2f}s, {STORY[section][0]}): rendering"
                    )
                    one_frame_started = monotonic()
                    scene = _frame_scene(
                        time, starts, durations, static_scenes, gradient_labels
                    )
                    frame = renderer.render(scene)
                    try:
                        frame_path = frame_directory / f"frame_{frame_index:06d}.png"
                        frame.save(frame_path)
                        if not poster_saved and time >= 31.0:
                            frame.save(poster)
                            poster_saved = True
                    finally:
                        frame.close()
                    elapsed = monotonic() - frame_started
                    average_frame_time = elapsed / current_frame
                    eta = average_frame_time * (frame_count - current_frame)
                    stats = renderer.stats
                    cache_entries = renderer.cache_info().entries
                    print(
                        f"[{strftime('%H:%M:%S')}] Frame {current_frame}/{frame_count} "
                        f"complete in {monotonic() - one_frame_started:.2f}s; "
                        f"elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m; "
                        f"GPU images allocated/reused="
                        f"{stats.texture_allocations}/{stats.texture_reuses}, "
                        f"cache entries={cache_entries}",
                        flush=True,
                    )

        _log("Stage 5/5: encode MP4 and mux audio")
        _encode_silent_video(frame_directory, silent_path, ffmpeg, video_codec)
        _mux_audio(silent_path, audio_path, output)

    _log(f"Render complete in {(monotonic() - render_started) / 60:.1f}m")
    return output, poster, subtitles, starts, durations, pause


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    video, poster, subtitles, _starts, _durations, pause = render_short(args.output)
    print(video)
    print(poster)
    print(subtitles)
    print(f"Render: {SIZE[0]}x{SIZE[1]}, {DURATION:.0f}s, narration pauses: {pause:.2f}s")


if __name__ == "__main__":
    main()
