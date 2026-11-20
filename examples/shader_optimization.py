"""Fit a ring with one combined field, pixel, and semantic geometry objective."""

import argparse
import json
from pathlib import Path

from sdf_ui import Canvas, param, sdf
from sdf_ui.gpu.nodes.registry import registry
from sdf_ui.learning import (
    Adam,
    Bounds,
    DifferentiableRenderer,
    Element,
    Layout,
    ParameterStore,
)


def _ring_scene(parameters):
    radius = param("radius").softplus()
    scene = sdf.ring((param("x"), param("y")), radius, thickness=3.0)
    display = scene.fill("#72c497", "#17212b").to_rgb()
    layout = Layout(
        [Element("ring", Bounds(
            param("x") - radius, param("y") - radius, radius * 2, radius * 2
        ))],
        (48, 40),
    )
    return scene, display, layout


def _target_images(renderer):
    target = sdf.ring((23, 20), 8, thickness=3)
    field = renderer.render(target).detach()
    pixels = renderer.render(target.fill("#72c497", "#17212b").to_rgb()).detach()
    return field, pixels


def _objective(renderer, scene, display, layout, parameters, targets):
    target_field, target_pixels = targets

    def loss():
        field = renderer.render(scene)
        pixels = renderer.render(display)
        geometry = (
            layout.offscreen_loss(parameters) / 1_000
            + layout.alignment_loss(
                ["ring"], axis="center_y", target=20, params=parameters
            ) / 100
        )
        return ((field - target_field) ** 2).mean() / 100 + (
            (pixels - target_pixels) ** 2
        ).mean() + geometry

    return loss


def _fit(parameters, loss, steps):
    optimizer = Adam(parameters.parameters(), lr=0.15)
    initial = loss().item()
    for _ in range(steps):
        optimizer.zero_grad()
        value = loss()
        value.backward()
        optimizer.step()
    return initial, loss().item()


def _local_radius_gradient():
    import numpy as np

    shader = registry.get("ring").shader
    local = shader.backward(
        (48, 40), {},
        {"center": [23.0, 20.0], "radius": 8.0, "thickness": 3.0},
        np.ones((40, 48)),
    )
    return float(local.parameters["radius"])


def optimize(output, steps=200, gpu=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parameters = ParameterStore()
    for name, value in (("x", 12.0), ("y", 12.0), ("radius", 4.0)):
        parameters.define(name, value)
    scene, display, layout = _ring_scene(parameters)
    renderer = DifferentiableRenderer(layout.size, params=parameters)
    targets = _target_images(renderer)
    loss = _objective(renderer, scene, display, layout, parameters, targets)
    renderer.save(display, output / "before.png")
    initial, final = _fit(parameters, loss, steps)
    renderer.save(display, output / "after.png")
    parameters.save(output / "parameters.json")
    if gpu:
        with Canvas((48, 40)) as canvas:
            display.save(output / "gpu_after.png", canvas=canvas, params=parameters)
    report = {
        "initial_loss": initial,
        "final_loss": final,
        "parameters": parameters.state_dict()["parameters"],
        "local_radius_gradient": _local_radius_gradient(),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("out/examples/shader_optimization")
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
