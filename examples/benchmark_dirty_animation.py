"""Compare full and automatic dirty rendering for every animation example.

Run with ``uv run python -m examples.benchmark_dirty_animation --frames 6``.
The baseline uses the compiled graph exporter path; the candidate uses a
persistent dirty-region renderer session. Timings include GPU readback so both
paths are measured on the same basis.
"""

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from examples.generative_art_animation import SIZE, tidal_bloom_scene
from examples.kinetic_city_animation import HEIGHT, WIDTH, kinetic_city_scene
from examples.learned_rule_art import build_score, make_learned_layout
from sdf_ui import Canvas, color, param, sdf
from sdf_ui.anim import _dirty_animation_node_count
from sdf_ui.learning import Parameter


def _scenes():
    center_x = Parameter(
        "center_x",
        0.0,
        animate=lambda progress: 640 * (0.35 + 0.25 * math.sin(progress * 2 * math.pi)),
    )
    moving_disc = sdf.circle((param("center_x"), 180), 54)
    simple = color.clear("#ffffff").alpha_overlay(
        sdf.circle((64, 180), 126)
        .smooth_union(moving_disc)
        .fill("#62bb47", (0.0, 0.0, 0.0, 0.0))
    )

    center = Parameter(
        "center_x", 0.0, animate=lambda progress: 120 + 400 * progress - 20
    )
    div = sdf.rounded_rect((param("center_x"), 180), (90, 56), 18).fill(
        "#62bb47", "#14251b"
    )
    center_scene = color.clear("#111827").over(div).to_rgb()

    phase = Parameter("phase", 0.0, animate=lambda progress: progress * 2 * math.pi)
    tidal_scene = tidal_bloom_scene(None)

    city = kinetic_city_scene()
    layout, parameters, widths, heights, names, _, _ = make_learned_layout()
    parameters = dict(parameters.items())
    learned_phase = Parameter(
        "phase", 0.0, animate=lambda progress: progress * 2 * math.pi
    )
    parameters["phase"] = learned_phase
    learned = build_score(layout, parameters, widths, heights, names)
    return (
        ("animation_example", (640, 360), simple, {"center_x": center_x}),
        ("center_div_motion", (640, 360), center_scene, {"center_x": center}),
        ("tidal_bloom", (SIZE, SIZE), tidal_scene, {"phase": phase}),
        ("blue_hour", (WIDTH, HEIGHT), city, {"phase": phase}),
        ("learned_score", (WIDTH, HEIGHT), learned, parameters),
    )


def _render(spec, frames, dirty):
    name, size, scene, params = spec
    snapshots = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sdf-ui-dirty-bench-") as directory:
        with Canvas(size) as canvas:
            use_dirty_renderer = dirty and _dirty_animation_node_count(scene) <= 96
            if use_dirty_renderer:
                with canvas.compile({"frame": scene}, params=params) as graph, canvas.session(
                    params=params, dirty_regions=True
                ) as renderer:
                    fallback_frames = 0
                    for index in range(frames):
                        progress = index / max(frames - 1, 1)
                        path = Path(directory) / f"{index:04d}.png"
                        use_dirty = True
                        if index:
                            renderer.progress = progress
                            rect = renderer._resolve_dirty_rect(scene)
                            area = 0 if rect is None else rect[2] * rect[3]
                            use_dirty = (
                                area <= 0.35 * size[0] * size[1]
                                and renderer.stats.dirty_region_fallbacks <= 10
                            )
                        if use_dirty:
                            renderer.render(scene, progress=progress).save(path)
                        else:
                            fallback_frames += 1
                            with graph.run(progress=progress) as execution:
                                execution.save("frame", path)
                        snapshots.append(np.asarray(Image.open(path).convert("RGBA")))
                    stats = renderer.stats.as_dict()
                    stats["full_frame_fallbacks"] = fallback_frames
                    stats["dirty_animation_enabled"] = True
                    return snapshots, time.perf_counter() - started, stats
            with canvas.compile({"frame": scene}, params=params) as graph:
                for index in range(frames):
                    progress = index / max(frames - 1, 1)
                    with graph.run(progress=progress) as execution:
                        path = Path(directory) / f"{index:04d}.png"
                        execution.save("frame", path)
                        snapshots.append(np.asarray(Image.open(path).convert("RGBA")))
    return snapshots, time.perf_counter() - started, {
        "dirty_animation_enabled": False
    }


def benchmark(frames=6):
    report = []
    for spec in _scenes():
        baseline, baseline_seconds, _ = _render(spec, frames, dirty=False)
        candidate, candidate_seconds, stats = _render(spec, frames, dirty=True)
        maximum = max(
            int(np.max(np.abs(left.astype(np.int16) - right.astype(np.int16))))
            for left, right in zip(baseline, candidate)
        )
        outside_tolerance_fraction = sum(
            float(
                np.any(
                    np.abs(left.astype(np.int16) - right.astype(np.int16)) > 5,
                    axis=2,
                ).mean()
            )
            for left, right in zip(baseline, candidate)
        ) / frames
        if maximum > 5:
            raise AssertionError(
                f"{spec[0]} differs by {maximum} channel values; expected <= 5"
            )
        report.append(
            {
                "example": spec[0],
                "frames": frames,
                "baseline_seconds": baseline_seconds,
                "dirty_seconds": candidate_seconds,
                "speedup": baseline_seconds / candidate_seconds,
                "pixel_identical": maximum == 0,
                "within_5_lsb": maximum <= 5,
                "max_channel_delta": maximum,
                "pixels_over_5_lsb_fraction": outside_tolerance_fraction,
                "dirty_region_pixels": stats.get("dirty_region_pixels", 0),
                "dirty_region_fallbacks": stats.get("dirty_region_fallbacks", 0),
                "dirty_layer_updates": stats.get("dirty_layer_updates", 0),
                "full_frame_fallbacks": stats.get("full_frame_fallbacks", 0),
                "dirty_animation_enabled": stats["dirty_animation_enabled"],
                "shader_dispatches": stats.get("shader_dispatches", 0),
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    print(json.dumps(benchmark(args.frames), indent=2))


if __name__ == "__main__":
    main()
