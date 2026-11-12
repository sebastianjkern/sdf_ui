"""Fit a ring with one combined field, pixel, and semantic geometry objective."""

import argparse
import json
from pathlib import Path

from sdf_ui import Canvas, param, sdf
from sdf_ui.backends.opengl.nodes.registry import registry
from sdf_ui.learning import (
    Adam,
    Bounds,
    DifferentiableRenderer,
    Element,
    Layout,
    ParameterStore,
)


def optimize(output, steps=200, gpu=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parameters = ParameterStore()
    parameters.define("x", 12.0)
    parameters.define("y", 12.0)
    parameters.define("radius", 4.0)
    radius = param("radius").softplus()
    scene = sdf.ring((param("x"), param("y")), radius, thickness=3.0)
    display = scene.fill("#72c497", "#17212b").to_rgb()
    layout = Layout(
        [
            Element(
                "ring",
                Bounds(
                    param("x") - radius, param("y") - radius, radius * 2, radius * 2
                ),
            )
        ],
        (48, 40),
    )
    renderer = DifferentiableRenderer(layout.size, params=parameters)
    target = sdf.ring((23, 20), 8, thickness=3)
    target_field = renderer.render(target).detach()
    target_pixels = renderer.render(target.fill("#72c497", "#17212b").to_rgb()).detach()

    def objective():
        field = renderer.render(scene)
        pixels = renderer.render(display)
        return (
            ((field - target_field) ** 2).mean() * 0.01
            + ((pixels - target_pixels) ** 2).mean()
            + layout.offscreen_loss(parameters) * 0.001
            + layout.alignment_loss(
                ["ring"], axis="center_y", target=20, params=parameters
            )
            * 0.01
        )

    renderer.save(display, output / "before.png")
    initial = objective().item()
    optimizer = Adam(parameters.parameters(), lr=0.15)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
    final = objective().item()
    renderer.save(display, output / "after.png")
    parameters.save(output / "parameters.json")
    if gpu:
        with Canvas((48, 40)) as canvas:
            display.save(output / "gpu_after.png", ctx=canvas, params=parameters)
    shader = registry.get("ring").shader
    # A local backward pass is also available independently of a render graph.
    import numpy as np

    local = shader.backward(
        (48, 40),
        {},
        {
            "center": [23.0, 20.0],
            "radius": 8.0,
            "thickness": 3.0,
        },
        np.ones((40, 48)),
    )
    report = {
        "initial_loss": initial,
        "final_loss": final,
        "parameters": parameters.state_dict()["parameters"],
        "local_radius_gradient": float(local.parameters["radius"]),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("out/learning/shader_optimization")
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Also display the learned scene through the GPU renderer",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    print(json.dumps(optimize(args.output, args.steps, args.gpu), indent=2))


if __name__ == "__main__":
    main()
