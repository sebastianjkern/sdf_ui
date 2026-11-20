"""Fit a circle radius with a custom loss over scene output and parameters."""



import argparse
from pathlib import Path

from sdf_ui import Canvas, param, sdf
from sdf_ui.learning import Parameter

_RADIUS_TEXT = 'radius'

def optimize(steps=150, output=None):
    radius = Parameter(_RADIUS_TEXT, 4.0)
    field = sdf.circle((16, 16), param(_RADIUS_TEXT))

    with Canvas((32, 32)) as canvas:
        with canvas.capture(sdf.circle((16, 16), 9)) as target:
            target_pixels = target.read()

            def loss(graph):
                image_error = ((graph["field"].output - target_pixels) ** 2).mean()
                radius_prior = (radius - 9) ** 2
                return image_error + radius_prior * 1 / 100

            result = canvas.fit(
                {"field": field},
                loss,
                params={_RADIUS_TEXT: radius},
                steps=steps,
                lr=0.1,
            )

        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            field.save(output, canvas=canvas, params={_RADIUS_TEXT: radius})

    return {
        _RADIUS_TEXT: radius.item(),
        "loss_before_last_update": result.final_loss,
        "steps": result.steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(optimize(args.steps, args.output))


if __name__ == "__main__":
    main()
