"""Train a bounded SDF parameter with the resource-owning session API.

Run with: python -m examples.high_level_learning
"""

from pathlib import Path

from sdf_ui import Canvas, sdf
from sdf_ui.learning import ParameterStore


def main():
    parameters = ParameterStore()
    radius = parameters.scalar("radius", initial=3, min=0)
    field = sdf.circle((16, 16), radius)
    image = field.fill("#72c497", "#17212b").to_rgb()
    destination = Path("out/learning/high_level")
    destination.mkdir(parents=True, exist_ok=True)

    with Canvas((32, 32)) as canvas:
        with canvas.training({"field": field, "image": image}, parameters) as session:
            session.match("field", sdf.circle((16, 16), 9))
            print("Initial:", session.evaluate())
            session.fit(150, lr=0.1)
            print("Final:", session.evaluate())
            print("Diagnostics:", session.report())
            session.save("image", destination / "learned.png")
            session.save_checkpoint(destination / "checkpoint.json")


if __name__ == "__main__":
    main()
