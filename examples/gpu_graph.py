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

_IMAGE_TEXT = 'image'

def _training_scene():
    parameters = ParameterStore()
    for name, value, trainable in (
        ("x", 19.0, True), ("y", 16.0, True), ("radius", 5.0, True),
        ("color", [0.3, 0.6, 0.5, 1.0], False),
    ):
        parameters.define(name, value, trainable=trainable)
    radius = param("radius").softplus()
    field = sdf.circle((param("x"), param("y")), radius)
    image = field.fill(param("color"), "#15212b").to_rgb()
    layout = Layout(
        [Element("disc", Bounds(
            param("x") - radius, param("y") - radius, radius * 2, radius * 2
        ), minimum_size=(16, 16))],
        (48, 36),
    )
    return parameters, field, image, layout


def _target_scene():
    field = sdf.circle((27, 20), 9)
    return field, field.fill((0.7, 0.35, 0.5, 1.0), "#15212b").to_rgb()


def _geometry_loss(layout, parameters):
    return (
        layout.offscreen_loss(parameters) / 1_000
        + layout.minimum_size_loss(parameters) / 1_000
        + layout.alignment_loss(
            ["disc"], axis="center_y", target=20, params=parameters
        ) / 100
    )


def _objective(layout, parameters, target_sdf, target_rgb):
    return (
        Objective()
        .mse("field", target_sdf, weight=0.02, name="field")
        .mse(_IMAGE_TEXT, target_rgb, name="pixels")
        .geometry(
            lambda: _geometry_loss(layout, parameters),
            name="geometry",
            identity={
                "version": 1, "layout": "disc", "center_y": 20,
                "weights": [0.001, 0.001, 0.01],
            },
        )
    )


def _training_report(trainer, graph, parameters, history, initial, final):
    return {
        "initial_loss": initial,
        "final_loss": final,
        "steps": trainer.steps,
        "parameters": parameters.state_dict()["parameters"],
        "training_image_readbacks": sum(
            entry["stats"]["image_readbacks"] for entry in history
        ),
        "gpu_backward_dispatches": sum(
            entry["stats"]["backward_dispatches"] for entry in history
        ),
        "cache": graph.cache_info(),
        "last_step": history[-1] if history else None,
    }


def _train_graph(graph, objective, parameters, output, steps, resume):
    graph.save(_IMAGE_TEXT, output / "before.png")
    with GraphTrainer(graph, objective, lr=0.15) as trainer:
        if resume:
            trainer.load(resume)
        with graph.run(objective) as execution:
            initial = execution.loss
        history = trainer.fit(steps)
        with graph.run(objective) as execution:
            final = execution.loss
            execution.save(_IMAGE_TEXT, output / "after.png")
        trainer.save(output / "checkpoint.json")
        report = _training_report(trainer, graph, parameters, history, initial, final)
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def train(output, steps=150, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parameters, field, image, layout = _training_scene()
    target_field, target_image = _target_scene()
    with Canvas((48, 36)) as canvas:
        with canvas.compile({"field": target_field, _IMAGE_TEXT: target_image}) as target_graph:
            with (
                target_graph.snapshot("field") as target_sdf,
                target_graph.snapshot(_IMAGE_TEXT) as target_rgb,
            ):
                target_graph.save(_IMAGE_TEXT, output / "target.png")
                with canvas.compile({"field": field, _IMAGE_TEXT: image}, params=parameters) as graph:
                    objective = _objective(layout, parameters, target_sdf, target_rgb)
                    return _train_graph(graph, objective, parameters, output, steps, resume)


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
