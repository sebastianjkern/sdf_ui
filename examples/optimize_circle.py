"""Fit a circle radius with the compiled render-graph API."""

import argparse
from pathlib import Path

from sdf_ui import Canvas, sdf
from sdf_ui.graph import GraphTrainer
from sdf_ui.learning import Objective, ParameterStore


def optimize(steps=150, output=None):
    parameters = ParameterStore()
    radius = parameters.bind("radius", 4.0)
    field = sdf.circle((16, 16), radius)
    image = field.fill("#72c497", "#17212b").to_rgb()
    with Canvas((32, 32)) as canvas:
        with canvas.capture(sdf.circle((16, 16), 9)) as target:
            objective = Objective().mse("field", target, name="field_fit")
            with canvas.compile({"field": field, "image": image}, parameters) as graph:
                with GraphTrainer(graph, objective, lr=0.1) as trainer:
                    history = trainer.fit(steps)
                    with graph.run(objective) as execution:
                        final_loss = execution.loss
                    if output is not None:
                        output = Path(output)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        graph.save("image", output)
                    return {
                        "radius": parameters["radius"].item(),
                        "initial_loss": history[0]["loss"] if history else final_loss,
                        "final_loss": final_loss,
                        "gradient_info": graph.explain()["parameters"],
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(optimize(args.steps, args.output))


if __name__ == "__main__":
    main()
