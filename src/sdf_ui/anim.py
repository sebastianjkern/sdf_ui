"""Render animated scene parameters to GIF or MP4 files."""

__docformat__ = "google"

from contextlib import ExitStack
from math import isfinite
from numbers import Integral, Real

import cv2

from .log import logger

_MAX_DIRTY_ANIMATION_NODES = 96
_DIRTY_FRAME_AREA_RATIO = 0.35


def timeline_position(time, starts, durations):
    """Return the active section and its clamped progress for a timeline.

    ``starts`` must be strictly increasing and pair with positive ``durations``.
    Time before the first section maps to its start; time in gaps or after the
    final section maps to the preceding section at progress one.
    """
    starts, durations = _validate_timeline(time, starts, durations)
    section = max(
        (index for index, start in enumerate(starts) if start <= time),
        default=0,
    )
    progress = (time - starts[section]) / durations[section]
    return section, min(1.0, max(0.0, progress))


def _validate_timeline(time, starts, durations):
    if isinstance(time, bool) or not isinstance(time, Real) or not isfinite(time):
        raise ValueError("time must be a finite number")
    starts = tuple(starts)
    durations = tuple(durations)
    if not starts or len(starts) != len(durations):
        raise ValueError("starts and durations must have the same nonzero length")
    _validate_section_starts(starts)
    _validate_section_durations(durations)
    return starts, durations


def _validate_section_starts(starts):
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
        for value in starts
    ):
        raise ValueError("section starts must be finite numbers")
    if any(first >= second for first, second in zip(starts, starts[1:])):
        raise ValueError("section starts must be strictly increasing")


def _validate_section_durations(durations):
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0
        for value in durations
    ):
        raise ValueError("section durations must be finite positive numbers")


def _dirty_animation_node_count(scene):
    from .core.texture import MultiOutputResult, TextureNode

    pending = [scene]
    visited = set()
    while pending:
        node = pending.pop()
        if id(node) in visited or not isinstance(
            node, (TextureNode, MultiOutputResult)
        ):
            continue
        visited.add(id(node))
        if isinstance(node, TextureNode) and node.tex is not None:
            continue
        pending.extend(node.inputs)
    return len(visited)


def convert_to_video(name, image_paths, *, fps=60):
    """
    Convert a sequence of images into a video.

    Parameters:
    - name (str): The name of the output video file (without extension).
    - image_paths (list): List of file paths of input images.

    Example:
    >>> images = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
    >>> convert_to_video("output_video", images)
    """
    _write_mp4(name + ".mp4", image_paths, fps)
    logger().info("Finished writing video")


def _write_mp4(path, image_paths, fps):
    """Write equally sized image frames to an MP4 file."""

    image_paths = _validated_video_inputs(fps, image_paths)

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read video frame: {image_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        writer.write(first)
        for image_path in image_paths[1:]:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read video frame: {image_path}")
            _validate_video_frame(frame, image_path, width, height)
            writer.write(frame)
    finally:
        writer.release()


def _validated_video_inputs(fps, image_paths):
    if isinstance(fps, bool) or not isinstance(fps, Real) or not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    values = list(image_paths)
    if not values:
        raise ValueError("at least one image frame is required")
    return values


def _validate_video_frame(frame, image_path, width, height):
    if frame.shape[:2] != (height, width):
        raise ValueError(
            f"Video frame {image_path} has size {frame.shape[1]}x{frame.shape[0]}, "
            f"expected {width}x{height}"
        )


def save_frame_sequence(
    canvas,
    scene,
    directory,
    *,
    frames=60,
    params=None,
    prefix="frame",
    start_index=0,
    dirty_regions=True,
):
    """Render a progress sweep to numbered PNG files and return their paths."""
    from pathlib import Path

    _validate_frame_sequence(frames, start_index, prefix, Path)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        directory / f"{prefix}_{int(start_index) + index:06d}.png"
        for index in range(int(frames))
    )
    with canvas.session(params=params, dirty_regions=dirty_regions) as renderer:
        for index, path in enumerate(paths):
            progress = index / max(int(frames) - 1, 1)
            renderer.render(scene, progress=progress).save(path)
    return paths


def _validate_frame_sequence(frames, start_index, prefix, Path):
    if isinstance(frames, bool) or not isinstance(frames, Integral) or frames < 1:
        raise ValueError("frames must be a positive integer")
    if isinstance(start_index, bool) or not isinstance(start_index, Integral) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a nonempty filename prefix")


def save_animation(
    canvas, scene, path, *, frames=60, params=None, fps=30, dirty_regions=True
):
    """Render an animated scene to a GIF or MP4 by sweeping progress from 0 to 1."""
    import tempfile
    from numbers import Integral, Real
    from pathlib import Path

    if isinstance(frames, bool) or not isinstance(frames, Integral) or frames < 1:
        raise ValueError("frames must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, Real) or fps <= 0:
        raise ValueError("fps must be a positive number")

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("animation path must end in .gif or .mp4")
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sdf-ui-frames-") as temp_dir:
        frame_paths = _AnimationRenderer.render_frames(
            canvas, scene, params, int(frames), dirty_regions, Path(temp_dir)
        )

        if suffix == ".gif":
            write_gif(frame_paths, path, duration=max(1, round(1000 / fps)))
        else:
            _write_mp4(path, frame_paths, fps)
    return path


class _AnimationRenderer:
    @staticmethod
    def render_frames(canvas, scene, params, frames, dirty_regions, directory):
        frame_paths = []
        with ExitStack() as stack:
            graph = _AnimationRenderer._compile_graph(canvas, scene, params, stack)
            renderer = _AnimationRenderer._dirty_renderer(canvas, scene, params, dirty_regions, stack)
            full_renderer = _AnimationRenderer._full_renderer(canvas, params, graph, stack)
            try:
                for index in range(frames):
                    progress = index / max(frames - 1, 1)
                    frame_path = directory / f"{index:06d}.png"
                    _AnimationRenderer._render_frame(
                        index, progress, frame_path, scene, canvas, graph,
                        renderer, full_renderer,
                    )
                    frame_paths.append(frame_path)
            finally:
                _AnimationRenderer._close_dirty_outputs(renderer)
        return frame_paths

    @staticmethod
    def _compile_graph(canvas, scene, params, stack):
        try:
            return stack.enter_context(canvas.compile({"frame": scene}, params=params))
        except NotImplementedError:
            return None
        except ValueError as error:
            if "unsupported custom render function" not in str(error):
                raise
            return None

    @staticmethod
    def _dirty_renderer(canvas, scene, params, dirty_regions, stack):
        if not dirty_regions or _dirty_animation_node_count(scene) > _MAX_DIRTY_ANIMATION_NODES:
            return None
        return stack.enter_context(canvas.session(params=params, dirty_regions=True))

    @staticmethod
    def _full_renderer(canvas, params, graph, stack):
        if graph is not None:
            return None
        return stack.enter_context(canvas.session(params=params, dirty_regions=False))

    @staticmethod
    def _render_frame(index, progress, frame_path, scene, canvas, graph, renderer, full_renderer):
        if renderer is not None and _AnimationRenderer._should_render_dirty(index, progress, renderer, scene, canvas):
            renderer.render(scene, progress=progress).save(frame_path)
        elif graph is None:
            full_renderer.render(scene, progress=progress).save(frame_path)
        else:
            with graph.run(progress=progress) as execution:
                execution.save("frame", frame_path)

    @staticmethod
    def _should_render_dirty(index, progress, renderer, scene, canvas):
        if index == 0:
            return True
        renderer.progress = progress
        rect = renderer._resolve_dirty_rect(scene)
        area = 0 if rect is None else rect[2] * rect[3]
        return (
            area <= _DIRTY_FRAME_AREA_RATIO * canvas.size[0] * canvas.size[1]
            and renderer.stats.dirty_region_fallbacks <= 10
        )

    @staticmethod
    def _close_dirty_outputs(renderer):
        if renderer is None:
            return
        owned = {}
        for _, state in renderer._dirty_layer_cache.values():
            outputs = state[1] if isinstance(state[1], tuple) else (state[1],)
            for output in outputs:
                texture = getattr(output, "tex", None)
                if texture is not None:
                    owned[id(texture)] = texture
        for texture in owned.values():
            texture.close()


def write_gif(image_paths, path, *, duration=100, loop=0):
    """Encode existing image frames as a GIF and return its path.

    ``duration`` is the frame delay in milliseconds. ``loop=0`` repeats
    forever, as specified by the GIF format.
    """
    from pathlib import Path

    from PIL import Image

    image_paths = _validate_gif_arguments(image_paths, duration, loop)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        frames = _load_gif_frames(image_paths, Image)
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration),
            loop=int(loop),
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()
    return path


def _validate_gif_arguments(image_paths, duration, loop):
    if isinstance(duration, bool) or not isinstance(duration, Integral) or duration < 1:
        raise ValueError("duration must be a positive integer number of milliseconds")
    if isinstance(loop, bool) or not isinstance(loop, Integral) or loop < 0:
        raise ValueError("loop must be a non-negative integer")
    paths = tuple(image_paths)
    if not paths:
        raise ValueError("at least one image frame is required")
    return paths


def _load_gif_frames(image_paths, Image):
    frames = []
    expected_size = None
    try:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                if expected_size is not None and image.size != expected_size:
                    raise ValueError(
                        f"GIF frame {image_path} has size {image.width}x{image.height}, "
                        f"expected {expected_size[0]}x{expected_size[1]}"
                    )
                expected_size = image.size
                frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
    except BaseException:
        for frame in frames:
            frame.close()
        raise
    return frames
