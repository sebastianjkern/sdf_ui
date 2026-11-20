"""Train layout rules, then render them as an animated graphic score."""

import json
import math
from pathlib import Path

from sdf_ui import Canvas, param, sdf, sin
from sdf_ui.learning import (
    Adam,
    Bounds,
    Element,
    Layout,
    LayoutTrainer,
    Parameter,
    ParameterStore,
)

WIDTH = 960
HEIGHT = 600
BASELINE = 500
BAR_COUNT = 17
PHASE = param("phase")

PALETTE = (
    (0.055, 0.22, 0.72, 1.0),
    (0.92, 0.25, 0.16, 1.0),
    (0.02, 0.48, 0.42, 1.0),
    (0.91, 0.58, 0.12, 1.0),
)


def _make_layout_inputs():
    parameters = ParameterStore()
    elements = []
    widths = []
    heights = []
    names = []

    # The sequence of widths and heights comes from a repeating harmonic rule.
    # Only the starting x/y values are seeded; the trainer arranges them.
    for index in range(BAR_COUNT):
        wave = math.sin(index * 117 / 100 + 7 / 20)
        width = 19 + 7 * (0.5 + 0.5 * math.cos(index * 83 / 100))
        height = 82 + 188 * (0.5 + 0.5 * wave)
        ideal_x = 116 + index * 43
        ideal_y = BASELINE - height
        jitter_x = 24 * math.sin(index * 413 / 100 + 3 / 5)
        jitter_y = 30 * math.sin(index * 271 / 100 + 6 / 5)
        name = f"mark_{index:02d}"
        x_name, y_name = f"{name}_x", f"{name}_y"
        parameters.define(x_name, ideal_x + jitter_x)
        parameters.define(y_name, ideal_y + jitter_y)
        elements.append(
            Element(
                name,
                Bounds(param(x_name), param(y_name), width, height),
                role="rhythmic mark",
            )
        )
        widths.append(width)
        heights.append(height)
        names.append(name)

    return parameters, elements, widths, heights, names


def _coverage_balance(parameters, names, widths, heights):
    span = parameters[f"{names[-1]}_x"] + widths[-1] - parameters[f"{names[0]}_x"]
    coverage = ((span - WIDTH * 19 / 25) / WIDTH) ** 2
    center = sum(
        (parameters[f"{name}_x"] + width / 2) * height
        for name, width, height in zip(names, widths, heights)
    ) / sum(heights)
    return coverage, ((center - WIDTH / 2) / 90) ** 2


def _boundary_penalties(layout, parameters):
    scale = WIDTH * HEIGHT
    return layout.overlap_loss(params=parameters) / scale, layout.offscreen_loss(params=parameters) / scale


def _rule_loss(layout, parameters, names, widths, heights):
    # A score of shared values and constraints, rather than hand-picked
    # final coordinates: regular gaps, one ground line, page coverage,
    # centered ink mass, no collisions, and no cropped marks.
    gaps = [
        parameters[f"{names[index + 1]}_x"]
        - parameters[f"{names[index]}_x"]
        - widths[index]
        for index in range(BAR_COUNT - 1)
    ]
    mean_gap = sum(gaps) / len(gaps)
    rhythmic_spacing = layout.gap_loss(
        names, gap=mean_gap, axis="x", params=parameters
    ) / (BAR_COUNT * 22**2)
    common_baseline = layout.alignment_loss(
        names, axis="bottom", target=BASELINE, params=parameters
    ) / (BAR_COUNT * 28**2)

    page_coverage, visual_balance = _coverage_balance(
        parameters, names, widths, heights
    )
    overlap, offscreen = _boundary_penalties(layout, parameters)

    return (
        4.0 * rhythmic_spacing
        + 3.0 * common_baseline
        + 24.0 * page_coverage
        + 45.0 * visual_balance
        + 8.0 * overlap
        + 16.0 * offscreen
    )


def make_learned_layout():
    """Find a rhythmic, centered layout from four composition rules."""
    parameters, elements, widths, heights, names = _make_layout_inputs()
    layout = Layout(elements, (WIDTH, HEIGHT))

    def rule_loss():
        return _rule_loss(layout, parameters, names, widths, heights)

    trainer = LayoutTrainer(
        parameters,
        rule_loss,
        optimizer=Adam(parameters.parameters(), lr=0.35),
        identity={"art_rules": "rhythm-baseline-balance-coverage-v1"},
    )
    initial_loss = trainer.evaluate()["loss"]
    trainer.fit(180)
    final_loss = trainer.evaluate()["loss"]
    return layout, parameters, widths, heights, names, initial_loss, final_loss


def _score_bars(snapshot, widths, heights, names):
    bars = []
    outlines = []
    caps = []
    tops = []
    for index, (name, width, height) in enumerate(zip(names, widths, heights)):
        bounds = snapshot.bounds[name]
        center_x = bounds.center_x
        breathing_height = height * (
            1.0 + 4 / 25 * sin(PHASE * 2.0 + index * 57 / 100)
        )
        center_y = HEIGHT - BASELINE + breathing_height / 2
        bar = sdf.rounded_rect(
            (center_x, center_y),
            (width / 2, breathing_height / 2),
            corner_radius=min(width * 11 / 50, 6),
        )
        bars.append(bar.fill(PALETTE[index % len(PALETTE)], (0.0, 0.0, 0.0, 0.0)))
        outlines.append(
            bar.outline(
                (0.99, 0.97, 0.89, 0.72),
                (0.0, 0.0, 0.0, 0.0),
                inflate=-7 / 5,
            )
        )
        top_y = HEIGHT - BASELINE + breathing_height
        caps.append(sdf.circle((center_x, top_y), 4.2))
        tops.append((center_x, top_y))
    return bars, outlines, caps, tops

def _score_backdrop():
    background = sdf.rect((WIDTH / 2, HEIGHT / 2), (WIDTH / 2, HEIGHT / 2)).fill(
        (0.94, 0.91, 0.83, 1.0), (0.94, 0.91, 0.83, 1.0)
    )
    guides = sdf.union_all([
        sdf.rect((WIDTH / 2, HEIGHT - y), (WIDTH / 2 - 52, 0.55))
        for y in (54, 104, 150, BASELINE, 542)
    ]).fill((0.16, 0.22, 0.30, 0.15), (0.0, 0.0, 0.0, 0.0))
    ground = sdf.rect((WIDTH / 2, HEIGHT - BASELINE), (WIDTH / 2 - 42, 1.2)).fill(
        (0.06, 0.10, 0.16, 0.82), (0.0, 0.0, 0.0, 0.0)
    )
    return background, guides, ground


def _score_connectors(tops):
    connectors = [
        sdf.capsule(tops[index], tops[index + 1], 1.55)
        for index in range(BAR_COUNT - 1)
    ]
    return sdf.union_all(connectors).fill((0.08, 0.47, 0.59, 0.68), (0.0, 0.0, 0.0, 0.0))


def build_score(layout, parameters, widths, heights, names):
    """Render the learned layout as a breathing, connected visual score."""
    snapshot = layout.snapshot(params=parameters)
    background, guides, ground = _score_backdrop()
    bars, outlines, caps, tops = _score_bars(snapshot, widths, heights, names)
    cap_layer = sdf.union_all(caps).fill(
        (0.98, 0.72, 0.27, 0.98), (0.0, 0.0, 0.0, 0.0)
    )
    connector_layer = _score_connectors(tops)

    scene = background
    for layer in (guides, ground, connector_layer, *bars, *outlines, cap_layer):
        scene = scene.alpha_overlay(layer)
    return scene


def learned_rule_art(
    frames=72, output=Path("out/generative_art/learned_score")
):
    """Train compositional rules, then export a still and looping MP4."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    layout, parameters, widths, heights, names, initial_loss, final_loss = (
        make_learned_layout()
    )
    phase = Parameter(
        "phase",
        0.0,
        animate=lambda progress: progress * 2.0 * math.pi,
    )
    scene = build_score(layout, parameters, widths, heights, names)
    render_params = dict(parameters.items())
    render_params["phase"] = phase

    with Canvas((WIDTH, HEIGHT)) as canvas:
        scene.save(
            output.with_suffix(".png"),
            canvas=canvas,
            params=render_params,
            progress=0.5,
        )
        canvas.save_animation(
            scene,
            output.with_suffix(".mp4"),
            frames=frames,
            params=render_params,
            fps=30,
        )

    report = {
        "initial_rule_loss": initial_loss,
        "final_rule_loss": final_loss,
        "learned_layout": layout.snapshot(params=parameters).as_dict(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Rule loss: {initial_loss:.5f} -> {final_loss:.5f}")
    return report


if __name__ == "__main__":
    learned_rule_art()
