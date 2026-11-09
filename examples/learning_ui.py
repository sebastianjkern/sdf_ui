"""Train layouts and fit SDF graphics; run with ``python -m examples.learning_ui``."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from sdf_ui import color, param, sdf
from sdf_ui.learning import (
    SGD,
    Adam,
    Bounds,
    CategoricalPolicy,
    DifferentiableRenderer,
    Element,
    GaussianPolicy,
    Layout,
    LayoutTrainer,
    ParameterStore,
    PolicyGradientTrainer,
)


def save_buttons(renderer, scene, layout, path, params=None):
    renderer.save(scene, path)
    # Labels are display annotations; measured bounds drive the learning loss.
    snapshot = layout.snapshot(params=params)
    with Image.open(path) as image:
        image = image.convert("RGBA")
        draw = ImageDraw.Draw(image)
        for name, bounds in snapshot.bounds.items():
            draw.text(
                (
                    bounds.x + bounds.width / 2,
                    image.height - bounds.y - bounds.height / 2,
                ),
                name.title(),
                fill="white",
                anchor="mm",
            )
        image.save(path)


def geometry_demo(output, steps):
    parameters = ParameterStore()
    elements = []
    for index, name in enumerate(("save", "cancel", "help")):
        x = parameters.define(name + "_x", 5.0 + index * 25)
        y = parameters.define(name + "_y", 8.0 + index * 47)
        elements.append(
            Element(name, Bounds(x, y, 72, 28), role="button", minimum_size=(64, 24))
        )
    layout = Layout(elements, (320, 160))
    names = [element.name for element in elements]
    scene = color.clear("#17212b")
    for element in elements:
        scene = scene.over(element.shape(corner_radius=6).fill("#3278ba", (0, 0, 0, 0)))
    renderer = DifferentiableRenderer(layout.size)
    save_buttons(renderer, scene, layout, output / "geometry_before.png")

    def loss():
        return (
            layout.overlap_loss() / 100
            + layout.offscreen_loss() / 100
            + layout.minimum_size_loss()
            + layout.alignment_loss(names, axis="top", target=64)
            + layout.gap_loss(names, gap=16, axis="x")
            + (parameters["save_x"] - 20) ** 2
        )

    trainer = LayoutTrainer(
        parameters.parameters(),
        loss,
        SGD(parameters.parameters(), lr=0.08, momentum=0.8),
    )
    initial = loss().item()
    trainer.fit(steps)
    final = loss().item()
    save_buttons(renderer, scene, layout, output / "geometry_after.png")
    parameters.save(output / "geometry_weights.json")
    return {"initial_loss": initial, "final_loss": final}


def image_demo(output, steps):
    parameters = ParameterStore()
    parameters.define("x", 21.0)
    parameters.define("y", 25.0)
    parameters.define("radius", 6.0)
    scene = sdf.circle((param("x"), param("y")), param("radius"))
    renderer = DifferentiableRenderer((64, 64), params=parameters)
    target_scene = sdf.circle((36.0, 32.0), 13.0)
    target = renderer.render(target_scene).detach()
    display = scene.fill("#62bb47", "#17212b").to_rgb()
    renderer.save(display, output / "image_before.png")
    renderer.save(
        target_scene.fill("#62bb47", "#17212b").to_rgb(), output / "image_target.png"
    )
    optimizer = Adam(parameters.parameters(), lr=0.3)
    initial = ((renderer.render(scene) - target) ** 2).mean().item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((renderer.render(scene) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    final = ((renderer.render(scene) - target) ** 2).mean().item()
    renderer.save(display, output / "image_after.png")
    parameters.save(output / "image_weights.json")
    return {
        "initial_loss": initial,
        "final_loss": final,
        "parameters": parameters.state_dict()["parameters"],
    }


def reward_demo(output, steps):
    import numpy as np

    policy = GaussianPolicy(2, 3, seed=7, initial_std=0.12)
    trainer = PolicyGradientTrainer(
        policy, optimizer=Adam(policy.parameters(), lr=0.01), seed=9
    )
    names = ("save", "cancel", "help")

    def observation(rng):
        return np.array([rng.uniform(240, 400) / 320, 1.0])

    def make_layout(obs, action):
        width = float(obs[0] * 320)
        return Layout(
            [
                Element(name, Bounds(float(x * width), 64, 64, 28), role="button")
                for name, x in zip(names, action)
            ],
            (width, 160),
        )

    def reward(obs, action):
        snapshot = make_layout(obs, action).snapshot()
        placement = float(np.sum((action - np.array([0.06, 0.38, 0.70])) ** 2))
        return -placement - (snapshot.overlap_area + snapshot.offscreen_area) / (
            snapshot.size[0] * 28
        )

    held_out = [np.array([width / 320, 1.0]) for width in (256, 320, 384)]

    def score():
        return float(
            np.mean([reward(obs, policy.deterministic_action(obs)) for obs in held_out])
        )

    def save_policy_image(suffix):
        obs = np.array([1.0, 1.0])
        layout = make_layout(obs, policy.deterministic_action(obs))
        renderer = DifferentiableRenderer((320, 160))
        scene = color.clear("#17212b")
        for element in layout.elements.values():
            scene = scene.over(element.shape(6).fill("#ba6432", (0, 0, 0, 0)))
        save_buttons(renderer, scene, layout, output / f"reward_{suffix}.png")

    initial = score()
    save_policy_image("before")
    history = trainer.train(reward, observation, steps=steps, batch_size=48)
    final = score()
    save_policy_image("after")

    # Learn a discrete responsive choice with the same callback interface.
    orientation = CategoricalPolicy(1, 2, seed=3)
    discrete_trainer = PolicyGradientTrainer(
        orientation, optimizer=Adam(orientation.parameters(), lr=0.02), seed=4
    )
    discrete_trainer.train(
        lambda obs, action: 1.0 if action == int(obs[0] < 0.7) else -1.0,
        lambda rng: np.array([rng.uniform(0.4, 1.2)]),
        steps=steps,
        batch_size=32,
    )
    widths = np.array([0.45, 0.55, 0.65, 0.8, 1.0, 1.15])
    accuracy = np.mean(
        [
            orientation.deterministic_action([width]) == int(width < 0.7)
            for width in widths
        ]
    )
    # Policy/trainer state dictionaries contain only serializable values.
    (output / "policy_checkpoint.json").write_text(
        json.dumps(trainer.state_dict(), indent=2), encoding="utf-8"
    )
    return {
        "initial_reward": initial,
        "final_reward": final,
        "orientation_accuracy": float(accuracy),
        "last_training_batch": history[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("geometry", "image", "reward", "all"), default="all"
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("out/learning"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, demo in (
        ("geometry", geometry_demo),
        ("image", image_demo),
        ("reward", reward_demo),
    ):
        if args.mode in (name, "all"):
            results[name] = demo(args.output, args.steps)
    (args.output / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(f"Images and checkpoints: {args.output.resolve()}")


if __name__ == "__main__":
    main()
