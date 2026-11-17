"""Render animated scene parameters to GIF or MP4 files."""

__docformat__ = "google"

from contextlib import ExitStack

import cv2

from .log import logger

_MAX_DIRTY_ANIMATION_NODES = 96


def _dirty_animation_node_count(scene):
    from .core.texture import MultiOutputResult, TextureNode

    pending = [scene]
    visited = set()
    while pending:
        node = pending.pop()
        if id(node) in visited or not isinstance(node, (TextureNode, MultiOutputResult)):
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
    from math import isfinite
    from numbers import Real

    if isinstance(fps, bool) or not isinstance(fps, Real) or not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("at least one image frame is required")

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
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Video frame {image_path} has size {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}"
                )
            writer.write(frame)
    finally:
        writer.release()


def save_animation(
    canvas, scene, path, *, frames=60, params=None, fps=30, dirty_regions=True
):
    """Render an animated scene to a GIF or MP4 by sweeping progress from 0 to 1."""
    import tempfile
    from numbers import Integral, Real
    from pathlib import Path

    from PIL import Image

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
        frame_paths = []
        with ExitStack() as stack:
            graph = stack.enter_context(canvas.compile({"frame": scene}, params=params))
            renderer_scope = (
                stack.enter_context(canvas.session(params=params, dirty_regions=True))
                if dirty_regions
                and _dirty_animation_node_count(scene) <= _MAX_DIRTY_ANIMATION_NODES
                else None
            )
            try:
                for index in range(int(frames)):
                    progress = index / max(int(frames) - 1, 1)
                    frame_path = Path(temp_dir) / f"{index:06d}.png"
                    use_dirty = renderer_scope is not None
                    if use_dirty and index:
                        renderer_scope.progress = progress
                        rect = renderer_scope._resolve_dirty_rect(scene)
                        area = 0 if rect is None else rect[2] * rect[3]
                        use_dirty = (
                            area <= 0.35 * canvas.size[0] * canvas.size[1]
                            and renderer_scope.stats.dirty_region_fallbacks <= 10
                        )
                    if use_dirty:
                        frame = renderer_scope.render(scene, progress=progress)
                        frame.save(frame_path)
                    else:
                        with graph.run(progress=progress) as execution:
                            execution.save("frame", frame_path)
                    frame_paths.append(frame_path)
            finally:
                if renderer_scope is not None:
                    # The dirty cache owns intermediate render targets for the session.
                    owned = {}
                    for _, state in renderer_scope._dirty_layer_cache.values():
                        outputs = state[1] if isinstance(state[1], tuple) else (state[1],)
                        for output in outputs:
                            texture = getattr(output, "tex", None)
                            if texture is not None:
                                owned[id(texture)] = texture
                    for texture in owned.values():
                        texture.close()

        if suffix == ".gif":
            images = [
                Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE)
                for frame in frame_paths
            ]
            try:
                images[0].save(
                    path,
                    save_all=True,
                    append_images=images[1:],
                    duration=max(1, round(1000 / fps)),
                    loop=0,
                    optimize=False,
                )
            finally:
                for image in images:
                    image.close()
        else:
            _write_mp4(path, frame_paths, fps)
    return path
