import numpy as np

from sdf_ui import Canvas, param, sdf
from sdf_ui.gpu.nodes.common import build
from sdf_ui.learning import Objective, Parameter, ParameterStore

_PHASE_OFFSET = 0.37
_ORIGIN_X = 1.25
_ORIGIN_Y_OFFSET = 0.75


def _hatch(shape, **overrides):
    values = {
        "fg_color": (0.15, 0.45, 0.7, 0.9),
        "inflate": 0.0,
        "angle": 0.23,
        "spacing": 5.3,
        "phase": -_PHASE_OFFSET,
        "width": 1.4,
        "origin_x": _ORIGIN_X,
        "origin_y": -_ORIGIN_Y_OFFSET,
    }
    values.update(overrides)
    return build("hatch_fill", shape, **values)


def test_hatch_fill_renders_identically_in_a_canvas_crop():
    shape = sdf.circle((13.5, 9.5), 8.2)
    image = _hatch(shape).to_rgb()
    with Canvas((28, 20)) as canvas:
        with (
            canvas.render(image) as full,
            canvas.render(image, evaluation_rect=(7, 4, 11, 9)) as cropped,
        ):
            np.testing.assert_array_equal(cropped.read(), full.read()[4:13, 7:18])


def test_antialiased_hatch_keeps_the_ink_color():
    image = _hatch(sdf.circle((13.5, 9.5), 8.2)).to_rgb()
    with Canvas((28, 20)) as canvas, canvas.render(image) as rendered:
        pixels = rendered.read()
    alpha = pixels[..., 3]
    edges = pixels[(alpha > 20) & (alpha < 200), :3]
    assert len(edges)
    np.testing.assert_allclose(
        edges, np.broadcast_to((38, 115, 178), edges.shape), atol=3
    )


def test_hatch_fill_native_gpu_gradients_reach_angle_phase_and_geometry():
    phase = Parameter("phase", -_PHASE_OFFSET)
    angle = Parameter("angle", 0.23)
    inflate = Parameter("inflate", 0.0)
    params = ParameterStore([phase, angle, inflate])
    image = _hatch(
        sdf.circle((7.5, 5.5), 4.2),
        phase=param("phase"),
        angle=param("angle"),
        inflate=param("inflate"),
    ).to_rgb()
    target = (
        np.random.default_rng(194).uniform(0.05, 0.95, (12, 16, 4)).astype(np.float32)
    )
    with Canvas((16, 12)) as canvas, canvas.compile({"image": image}, params) as graph:
        with graph.run(Objective().mse("image", target)) as execution:
            execution.backward()
    assert all(
        parameter.grad is not None and np.all(np.isfinite(parameter.grad))
        for parameter in (phase, angle, inflate)
    )
    assert all(np.any(parameter.grad != 0) for parameter in (phase, angle, inflate))
