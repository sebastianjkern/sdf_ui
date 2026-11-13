"""Measure synchronized graph execution on the active WebGPU device.

Run from a checkout: python -m examples.benchmark_graph --size 64 --steps 20
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from sdf_ui import Canvas, sdf
from sdf_ui.graph import GraphTrainer
from sdf_ui.learning import Objective, ParameterStore


def benchmark(size=64, warmup=3, steps=20):
    if size < 1 or warmup < 0 or steps < 1:
        raise ValueError("size and steps must be positive; warmup must be nonnegative")
    weights = ParameterStore()
    radius = weights.bind("radius", size * 0.2)
    field = sdf.circle((size * 0.5, size * 0.5), radius)
    scene = field.fill("#6db59a", "#15212b").to_rgb()
    target = np.zeros((size, size, 4), dtype=np.float32)
    with (
        Canvas((size, size)) as canvas,
        canvas.compile({"image": scene}, weights) as graph,
    ):
        objective = Objective().mse("image", target)
        timings = {"forward_loss_ms": [], "backward_ms": [], "training_step_ms": []}
        for iteration in range(warmup + steps):
            for parameter in graph.parameters:
                parameter.zero_grad()
            graph.clear_cache()
            canvas.queue.on_submitted_work_done_sync()
            start = perf_counter()
            with graph.run(objective) as execution:
                canvas.queue.on_submitted_work_done_sync()
                forward = perf_counter() - start
                start = perf_counter()
                execution.backward()
                canvas.queue.on_submitted_work_done_sync()
                backward = perf_counter() - start
            if iteration >= warmup:
                timings["forward_loss_ms"].append(forward * 1000)
                timings["backward_ms"].append(backward * 1000)
        with GraphTrainer(graph, objective, lr=0.001) as trainer:
            for iteration in range(warmup + steps):
                canvas.queue.on_submitted_work_done_sync()
                start = perf_counter()
                trainer.step()
                canvas.queue.on_submitted_work_done_sync()
                if iteration >= warmup:
                    timings["training_step_ms"].append((perf_counter() - start) * 1000)
        return {
            "device": canvas.device_info,
            "size": [size, size],
            "warmup": warmup,
            "steps": steps,
            "timings_ms": {
                key: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for key, values in timings.items()
            },
            "notes": "Host wall time with WebGPU queue synchronization; forward includes loss seeding, backward includes scalar gradient transfer.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(benchmark(args.size, args.warmup, args.steps), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
