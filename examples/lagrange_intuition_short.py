"""Render a one-minute narrated visual intuition for Lagrange multipliers.

All visual layers and frames are rendered by SDF UI. The optional ``math``
extra typesets equations; ``espeak-ng`` and ``ffmpeg`` create the narration
and final MP4. Run from the repository root with::

    uv run --extra math python -m examples.lagrange_intuition_short
"""

import argparse
import math
import tempfile
from pathlib import Path
from time import monotonic, strftime

from examples._lagrange_short_audio import speak_clips
from examples._lagrange_short_video import (
    encode_silent_video,
    mux_audio,
    select_video_encoder,
    write_subtitles,
)
from sdf_ui import Canvas, Plot, color, sdf
from sdf_ui.anim import timeline_position
from sdf_ui.widgets import Panel, progress_bar

OUTPUT = Path("out/examples/lagrange_intuition_short/lagrange_intuition_short.mp4")
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
PLOT = Plot((*PLOT_ORIGIN, 6 / 5 * PLOT_SCALE, 6 / 5 * PLOT_SCALE), (0, 6 / 5), (0, 6 / 5))

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
        (
            r"x^2+y^2=b",
            r"f_{\max}(b)=\frac{b}{2}",
            r"\lambda=\frac{df_{\max}}{db}=\frac{1}{2}",
        ),
        "The multiplier is the marginal value\nof a little extra budget.",
    ),
)


def _log(message):
    """Print a timestamped progress message immediately."""
    print(f"[{strftime('%H:%M:%S')}] {message}", flush=True)



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
    layer = text_node.ink(tint, inflate=0.2)
    return layer if cache else layer.uncached()


def _formula(value, center, size=34, tint=GOLD):
    node = sdf.math(value, center=center, height=None, font_size=size, inflate=0.2)
    return node.ink(tint)


def _level_curve(level, tint, width=2.2, *, cache=True):
    # Keep the endpoints a little inside the viewport, clear of the labels.
    curve = PLOT.curve(lambda x: level / x, domain=(level / 59 / 50, 1.18))
    if not cache:
        curve = curve.uncached()
    layer = curve.stroke(width, tint)
    return layer if cache else layer.uncached()


def _static_plot():
    scene = color.clear(BG).over(
        _text("A GEOMETRIC GUIDE TO OPTIMIZATION", (66, 665, 740, 28), size=17, tint=MUTED)
    ).over(_formula(
        r"\text{Maximize }f(x,y)=xy\quad\text{on}\quad x^2+y^2=1",
        (640, 641), size=34, tint=WHITE,
    )).over(PLOT.axes(
        x_ticks=(0, 0.5, 1), y_ticks=(0.5, 1), axis_color=MUTED,
        grid_color=GRID, x_label="x", y_label="y",
    ))
    constraint = sdf.arc(PLOT_ORIGIN, PLOT_SCALE, 0.0, math.pi / 2).stroke(3.0, TEAL)
    scene = scene.over(constraint).over(
        _formula("x^2+y^2=1", PLOT.point(0.62, 1.04), size=23, tint=TEAL)
    ).over(_text("constraint", (450, 100, 125, 25), size=14, tint=TEAL))
    return scene.cache("lagrange_short_plot")


def _contour_scene():
    scene = color.clear(TRANSPARENT)
    for level in (0.08, 0.16, 0.24, 0.32, 2 / 5, 0.46):
        scene = scene.over(_level_curve(level, (1.0, 0.78, 0.47, 0.42), width=1.35))
    return scene.cache("lagrange_short_contours")


def _story_panels(common, contours):
    card = Panel((700, 85, 520, 540), padding=(35, 24), radius=20)
    left, _, width, _ = card.content_box
    scenes = []
    for index, (header, body, formulas, note) in enumerate(STORY):
        panel = card.layer(PANEL).over(_text(header, (left, 585, width, 36), size=19, tint=TEAL))
        panel = panel.over(_text(body, (left, 425, width, 132), size=25, tint=WHITE, valign="top"))
        for formula_index, expression in enumerate(formulas):
            panel = panel.over(_formula(
                expression, (960, 330 - formula_index * 64),
                tint=(GOLD, TEAL, PURPLE)[formula_index % 3],
            ))
        if note:
            panel = panel.over(_text(note, (left, 98, width, 62), size=18, tint=MUTED, valign="top"))
        scene = common.over(panel)
        if index >= 2:
            scene = scene.over(contours)
        scenes.append(scene.cache(f"lagrange_short_section_{index}"))
    return scenes


def _gradient_labels():
    optimum = 1 / math.sqrt(2)
    labels = color.clear(TRANSPARENT).over(_formula(
        r"\nabla f", (PLOT.point(optimum + 1 / 5, optimum + 1 / 5)[0] + 30,
        PLOT.point(optimum + 1 / 5, optimum + 1 / 5)[1] + 12), size=24, tint=GOLD,
    ))
    end = PLOT.point(optimum + 2 / 5, optimum + 2 / 5)
    return labels.over(_formula(r"\nabla g", (end[0] + 30, end[1] + 12), size=24, tint=PURPLE)).cache(
        "lagrange_short_gradient_labels"
    )


def _build_static_scenes():
    """Build common plot layers and the seven reusable explanation panels."""
    common, contours = _static_plot(), _contour_scene()
    return _story_panels(common, contours), _gradient_labels()


def _frame_angle(section, local):
    angles = (0.16, 4 / 25 + 13 / 50 * local, 21 / 50 + 7 / 25 * local,
              7 / 10 + (math.pi / 4 - 7 / 10) * local, math.pi / 4)
    return angles[min(section, len(angles) - 1)]



def _gradient_arrows(scene, start, gradient_labels):
    long_end = PLOT.point(start[0] + 4 / 10, start[1] + 4 / 10)
    short_end = PLOT.point(start[0] + 2 / 10, start[1] + 2 / 10)
    scene = scene.over(
        sdf.arrow(start, long_end, width=3.2).fill(PURPLE, TRANSPARENT), cache=False
    )
    scene = scene.over(
        sdf.arrow(start, short_end, width=3.2).fill(GOLD, TRANSPARENT), cache=False
    )
    return scene.over(gradient_labels, cache=False)

def _tangent_layers(scene, section, angle, level, gradient_labels):
    if section < 3:
        return scene
    optimum = 1 / math.sqrt(2)
    optimum_pixel = PLOT.point(optimum, optimum)
    ring = sdf.circle(optimum_pixel, 12).uncached().stroke(2, GOLD).uncached()
    tangent = sdf.segment(
        PLOT.point(optimum - 6 / 25, optimum + 6 / 25),
        PLOT.point(optimum + 6 / 25, optimum - 6 / 25), 1.5,
    ).uncached().fill(PURPLE, TRANSPARENT).uncached()
    scene = scene.over(tangent, cache=False).over(ring, cache=False)
    if section == 3 and level < 499 / 1_000:
        other = PLOT.point(math.cos(math.pi / 2 - angle), math.sin(math.pi / 2 - angle))
        marker = sdf.circle(other, 6).uncached().fill(GOLD, TRANSPARENT).uncached()
        scene = scene.over(marker, cache=False)
    if section >= 4:
        scene = _gradient_arrows(scene, optimum_pixel, gradient_labels)
    return scene


def _point_annotations(scene, section, point, level):
    if section < 2:
        return scene
    label_x = min(1.08, max(0.76, point[0] + 1 / 10))
    label_y = min(1.13, max(0.13, level / label_x + 1 / 20))
    return scene.over(PLOT.annotate(
        (label_x, label_y), "xy = c", offset=(0, 7), size=16, tint=GOLD, cache=False
    ), cache=False)


def _frame_hud(scene, time):
    scene = scene.over(progress_bar(
        (67.5, 32.5, 1145, 5), time / DURATION,
        track_color=GRID, fill_color=TEAL, cache=False,
    ), cache=False)
    clock = f"{int(time // 60):01d}:{int(time % 60):02d}"
    return scene.over(_text(
        clock, (1150, 8, 62, 20), size=13, tint=MUTED,
        align="right", cache=False,
    ), cache=False)


def _frame_scene(time, starts, durations, static_scenes, gradient_labels):
    section, local = timeline_position(time, starts, durations)
    angle = _frame_angle(section, local)
    point = (math.cos(angle), math.sin(angle))
    level = 0.5 * math.sin(2 * angle)
    scene = static_scenes[section]
    if section >= 1:
        scene = scene.over(_level_curve(level, GOLD, width=3.0, cache=False), cache=False)
    scene = _tangent_layers(scene, section, angle, level, gradient_labels)
    point_pixel = PLOT.point(*point)
    marker = sdf.circle(point_pixel, 8).uncached().fill(WHITE, TRANSPARENT).uncached()
    scene = scene.over(marker, cache=False)
    scene = _point_annotations(scene, section, point, level)
    return _frame_hud(scene, time)












def _render_frame(renderer, frame_index, frame_count, starts, durations, static_scenes, gradient_labels, frame_directory, poster, frame_started):
    time = frame_index / FPS
    section, _ = timeline_position(time, starts, durations)
    current = frame_index + 1
    _log(f"Frame {current}/{frame_count} ({time:.2f}s, {STORY[section][0]}): rendering")
    started = monotonic()
    scene = _frame_scene(time, starts, durations, static_scenes, gradient_labels)
    frame = renderer.render(scene)
    try:
        frame.save(frame_directory / f"frame_{frame_index:06d}.png")
        if frame_index == round(31.0 * FPS):
            frame.save(poster)
        elapsed = monotonic() - frame_started
        eta = elapsed / current * (frame_count - current)
        stats = renderer.stats
        entries = renderer.cache_info().entries
        print(
            f"[{strftime('%H:%M:%S')}] Frame {current}/{frame_count} complete in "
            f"{monotonic() - started:.2f}s; elapsed {elapsed / 60:.1f}m, "
            f"ETA {eta / 60:.1f}m; GPU images allocated/reused="
            f"{stats.texture_allocations}/{stats.texture_reuses}, cache entries={entries}",
            flush=True,
        )
    finally:
        frame.close()


def _render_frames(static_scenes, gradient_labels, starts, durations, poster, frame_directory):
    frame_count = round(FPS * DURATION)
    frame_started = monotonic()
    with Canvas(SIZE) as canvas:
        with canvas.session(cache={}) as renderer:
            for index in range(frame_count):
                _render_frame(
                    renderer, index, frame_count, starts, durations,
                    static_scenes, gradient_labels, frame_directory, poster, frame_started,
                )


def _render_stages(output, poster, subtitles, ffmpeg, video_codec):
    with tempfile.TemporaryDirectory(prefix="lagrange-short-") as temp_name:
        temp_dir = Path(temp_name)
        _log("Stage 1/5: synthesize narration")
        audio_path, starts, durations, pause = speak_clips(
            temp_dir, NARRATION, DURATION, _log
        )
        _log("Stage 2/5: write subtitle file")
        write_subtitles(subtitles, NARRATION, starts, durations)
        _log("Stage 3/5: build reusable SDF UI plot and section scenes")
        scene_started = monotonic()
        static_scenes, gradient_labels = _build_static_scenes()
        _log(f"Static SDF UI scenes built in {monotonic() - scene_started:.1f}s")
        silent_path = temp_dir / "lagrange_silent.mp4"
        frame_directory = temp_dir / "frames"
        frame_directory.mkdir()
        _log("Stage 4/5: render and save SDF UI frames")
        _render_frames(
            static_scenes, gradient_labels, starts, durations, poster, frame_directory
        )
        _log("Stage 5/5: encode MP4 and mux audio")
        encode_silent_video(frame_directory, silent_path, ffmpeg, video_codec, FPS, _log)
        mux_audio(silent_path, audio_path, output, DURATION, _log)
    return starts, durations, pause


def render_short(output=OUTPUT):
    """Render the narrated short with SDF UI, plus a poster and subtitles."""
    render_started = monotonic()
    ffmpeg, video_codec = select_video_encoder()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    poster = output.with_name(output.stem + "_poster.png")
    subtitles = output.with_suffix(".srt")
    _log(
        f"Starting {SIZE[0]}x{SIZE[1]} SDF UI render: "
        f"{DURATION:.0f}s at {FPS} fps ({round(FPS * DURATION)} frames)"
    )
    _log(f"Outputs: video={output}, poster={poster}, subtitles={subtitles}")
    starts, durations, pause = _render_stages(
        output, poster, subtitles, ffmpeg, video_codec
    )
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
    print(
        f"Render: {SIZE[0]}x{SIZE[1]}, {DURATION:.0f}s, narration pauses: {pause:.2f}s"
    )


if __name__ == "__main__":
    main()
