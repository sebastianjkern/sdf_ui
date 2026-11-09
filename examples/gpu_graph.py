"""Native SDF graph training with GPU targets, combined losses, and checkpoints."""

import argparse
import json
from pathlib import Path

from sdf_ui import Canvas, param, sdf
from sdf_ui.graph import GraphTrainer
from sdf_ui.learning import (
    Bounds,
    Element,
    Layout,
    Objective,
    ParameterStore,
)


def train(output, steps=150, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parameters = ParameterStore()
    parameters.define("x", 19.0)
    parameters.define("y", 16.0)
    parameters.define("radius", 5.0)
    parameters.define("color", [0.3, 0.6, 0.5, 1.0])
    radius = param("radius").softplus()
    field = sdf.circle((param("x"), param("y")), radius)
    image = field.fill(param("color"), "#15212b").to_rgb()
    layout = Layout(
        [
            Element(
                "disc",
                Bounds(
                    param("x") - radius, param("y") - radius, radius * 2, radius * 2
                ),
                minimum_size=(16, 16),
            )
        ],
        (48, 36),
    )
    target_field = sdf.circle((27, 20), 9)
    target_image = target_field.fill((0.7, 0.35, 0.5, 1.0), "#15212b").to_rgb()
    with Canvas((48, 36)) as canvas:
        with canvas.compile(
            {"field": target_field, "image": target_image}
        ) as target_graph:
            with (
                target_graph.snapshot("field") as target_sdf,
                target_graph.snapshot("image") as target_rgb,
            ):
                target_graph.save("image", output / "target.png")
                with canvas.compile(
                    {"field": field, "image": image}, params=parameters
                ) as graph:
                    objective = (
                        Objective()
                        .mse("field", target_sdf, weight=0.02, name="field")
                        .mse("image", target_rgb, name="pixels")
                        .geometry(
                            lambda: (
                                layout.offscreen_loss(parameters) * 0.001
                                + layout.minimum_size_loss(parameters) * 0.001
                                + layout.alignment_loss(
                                    ["disc"],
                                    axis="center_y",
                                    target=20,
                                    params=parameters,
                                )
                                * 0.01
                            ),
                            name="geometry",
                            identity={
                                "version": 1,
                                "layout": "disc",
                                "center_y": 20,
                                "weights": [0.001, 0.001, 0.01],
                            },
                        )
                    )
                    graph.save("image", output / "before.png")
                    with GraphTrainer(graph, objective, lr=0.15) as trainer:
                        if resume:
                            trainer.load(resume)
                        with graph.run(objective) as execution:
                            initial = execution.loss
                        history = trainer.fit(steps)
                        with graph.run(objective) as execution:
                            final = execution.loss
                            execution.save("image", output / "after.png")
                        trainer.save(output / "checkpoint.json")
                        report = {
                            "initial_loss": initial,
                            "final_loss": final,
                            "steps": trainer.steps,
                            "parameters": parameters.state_dict()["parameters"],
                            "training_image_readbacks": sum(
                                h["stats"]["image_readbacks"] for h in history
                            ),
                            "gpu_backward_dispatches": sum(
                                h["stats"]["backward_dispatches"] for h in history
                            ),
                            "cache": graph.cache_info(),
                            "last_step": history[-1] if history else None,
                        }
                        (output / "metrics.json").write_text(
                            json.dumps(report, indent=2), encoding="utf-8"
                        )
                        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("out/learning/gpu_graph"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    print(json.dumps(train(args.output, args.steps, args.resume), indent=2))


if __name__ == "__main__":
    main()
