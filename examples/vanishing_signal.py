"""Draw an abstract tunnel from projected right-angle paths."""

import argparse
from pathlib import Path

from sdf_ui import Canvas, param, sdf

SIZE = (800, 1240)
FOCUS = (402, 619)
OUTPUT = Path("out/examples/vanishing_signal/vanishing_signal.png")


def perspective_paths(paths, focus, *, count, far, near, width):
    """Project a set of open paths onto a geometric sequence of depth planes."""
    fx, fy = focus
    points = []
    for index in range(count):
        scale = far * (near / far) ** (index / (count - 1))
        for path in paths:
            points.extend(
                (x * scale + fx * (1 - scale), SIZE[1] - y * scale - fy * (1 - scale))
                for x, y in path
            )
            points.append(None)
    return sdf.polyline(points, radius=width / 2)


def _architecture_paths(focus):
    structure = perspective_paths(
        (
            ((-88, 240), (72, 240), (72, 951), (674, 951), (674, 458)),
            ((536, 497), (733, 497), (733, 295), (884, 295)),
            ((-36, 644), (19, 644), (19, 713)),
        ), focus, count=23, far=0.12, near=1.04, width=3.7,
    )
    fold = perspective_paths(
        (((268, -90), (268, 371), (117, 371), (117, 810), (639, 810), (639, 1125)),),
        focus, count=25, far=0.13, near=1.02, width=2.15,
    )
    ticks = perspective_paths(
        (((478, 193), (647, 193), (647, 311)), ((286, 941), (386, 941))),
        focus, count=21, far=0.13, near=1.0, width=1.65,
    )
    return structure, fold, ticks


def _signal_paths(focus):
    rail = perspective_paths(
        (((907, 443), (757, 443), (757, 754), (907, 754)),),
        focus, count=25, far=0.13, near=1.14, width=3.25,
    )
    markers = perspective_paths(
        (((297, 42), (349, 42), (349, 12)), ((378, 88), (397, 88)), ((23, 983), (23, 1001))),
        focus, count=19, far=0.19, near=1.06, width=2.15,
    )
    return rail, markers


def vanishing_signal():
    focus = (param("focus_x", FOCUS[0]), param("focus_y", FOCUS[1]))
    backdrop = sdf.circle((400, 604), 0).fill_linear(
        (0.097, 0.099, 0.093, 1), (0.021, 0.026, 0.032, 1), inner=12, outer=646
    )
    structure, fold, ticks = _architecture_paths(focus)
    rail, markers = _signal_paths(focus)
    layers = (
        structure.ink_linear((0.004, 0.006, 0.008, 0.93)),
        fold.ink_linear((0.36, 0.92, 0.97, 0.96)),
        ticks.ink_linear((0.45, 0.87, 0.9, 0.86)),
        rail.ink_linear((1.45, 0.017, 0.029, 0.98)),
        markers.ink_linear((1.12, 0.025, 0.033, 0.93)),
    )
    return backdrop.over(*layers).bloom(
        threshold=0.72, amount=0.26, passes=1
    ).film_grain(0.15).finish(tone=0.18, dither=0.65)


def render(output=OUTPUT, *, focus=FOCUS):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas(SIZE) as canvas:
        canvas.save(
            vanishing_signal(), output, params=dict(zip(("focus_x", "focus_y"), focus))
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--focus",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=FOCUS,
        help="Vanishing point in pixels from the top-left corner",
    )
    args = parser.parse_args()
    if not all(0 <= coordinate <= size for coordinate, size in zip(args.focus, SIZE)):
        parser.error("--focus must lie inside the canvas")
    print(render(args.output, focus=args.focus))


if __name__ == "__main__":
    main()
