"""Shader contracts stay usable through public render-graph conventions."""

import numpy as np
import pytest

from sdf_ui import Canvas, color, param, sdf
from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.learning import (
    Adam,
    Bounds,
    DifferentiableRenderer,
    Element,
    Layout,
    Parameter,
)
from sdf_ui.learning.shaders.plugins import supports_plugin


@pytest.mark.parametrize(
    "factory",
    [
        lambda p: sdf.segment((0.2, 1.3), (13.1, 9.7), p + 1),
        lambda p: sdf.ellipse((7.3, 5.4), (p + 3, 2.7)),
        lambda p: sdf.arc((7.3, 5.4), p + 3, 0.2, 4.3),
        lambda p: sdf.wedge((7.3, 5.4), p + 3, 0.2, 4.3),
        lambda p: sdf.grid((p, 0.3), (3.7, 5.1), angle=0.13),
        lambda p: sdf.half_plane((p, 2.1), (0.4, 1.0)),
        lambda p: sdf.line((p, 1.3), (13.1, 9.7)),
        lambda p: sdf.bezier((0.3, p + 1), (8.13, 8.71), (14.27, 1.38)),
        lambda p: sdf.triangle((p, 1.3), (13.1, 2.7), (5.2, 10.1)),
        lambda p: sdf.annulus((7.3, 5.4), p + 3, thickness=1.7),
        lambda p: sdf.circle((7.3, 5.4), 3.1).skew((p * 0.1, -0.13), center=(7.3, 5.4)),
    ],
)
def test_new_primitive_adapters_preserve_parameter_expression_gradients(factory):
    parameter = Parameter("p", 0.7)
    renderer = DifferentiableRenderer((16, 12), {"p": parameter})
    output = renderer.render(factory(param("p")))
    assert output.shape == (12, 16)
    (output * np.linspace(0.1, 1.0, 192).reshape(12, 16)).mean().backward()
    assert parameter.grad is not None and np.isfinite(parameter.grad)


@pytest.mark.parametrize("operation", ["outline", "isolines", "light"])
def test_new_shading_parameters_receive_gradients(operation):
    parameter = Parameter("p", 0.7)
    field = sdf.circle((7.3, 5.4), 3.1)
    if operation == "outline":
        scene = field.outline((param("p"), 0.3, 0.4, 0.8))
    elif operation == "isolines":
        scene = field.isolines("#88bbcc", spacing=param("p") + 4.0)
    else:
        scene = field.light(
            (0.6, 0.4, 0.3, 1.0),
            shade_background=param("p"),
            specular=0.1,
            shininess=3.0,
        )
    output = DifferentiableRenderer((16, 12), {"p": parameter}).render(scene.to_rgb())
    output.mean().backward()
    assert parameter.grad is not None and np.isfinite(parameter.grad)


@pytest.mark.parametrize("operation", ["blur", "blur_9", "blur_13", "shadow"])
def test_composite_passes_propagate_source_geometry_gradients(operation):
    radius = Parameter("radius", 3.17)
    field = sdf.circle((7.23, 5.61), radius)
    scene = (
        field.shadow(distance=0)
        if operation == "shadow"
        else getattr(field.fill("#b97654", "#123456"), operation)(n=0)
    )
    output = DifferentiableRenderer((16, 12)).render(scene.to_rgb())
    output.mean().backward()
    assert radius.grad is not None and np.isfinite(radius.grad)


def test_fill_and_mask_adapters_convert_lab_inputs_to_rgb():
    alpha = Parameter("alpha", 0.4)
    renderer = DifferentiableRenderer((12, 10))
    field = sdf.circle((5.1, 4.3), 3.1)
    foreground = color.clear((0.7, 0.3, 0.2, alpha))
    filled = field.fill_from_texture(foreground, background=(0.1, 0.2, 0.3, 0.5))
    masked = color.clear("#112233").mask(filled, field.generate_mask())
    output = renderer.render(masked)
    output.mean().backward()
    assert alpha.grad is not None and np.isfinite(alpha.grad)
    assert output.shape == (10, 12, 4)


def test_masked_union_outputs_both_reach_shared_parameter_leaves():
    radius = Parameter("radius", 2.7)
    field, mask = sdf.circle((3.2, 4.1), radius).masked_union(
        sdf.circle((6.3, 4.2), 3.1)
    )
    renderer = DifferentiableRenderer((10, 9))
    distance, color_output = renderer.render(field), renderer.render(mask)
    loss = distance.mean() + color_output.mean()
    loss.backward()
    combined = radius.grad.copy()
    radius.zero_grad()
    distance.mean().backward()
    first = radius.grad.copy()
    radius.zero_grad()
    color_output.mean().backward()
    np.testing.assert_allclose(combined, first + radius.grad)


def test_combined_pixel_field_and_geometry_objective_optimizes_shared_parameters():
    radius = Parameter("radius", 2.0)
    parameters = {"radius": radius}
    renderer = DifferentiableRenderer((20, 16), parameters)
    field = sdf.ring((9.3, 7.2), param("radius"), thickness=2.0)
    layout = Layout(
        [
            Element(
                "ring",
                Bounds(
                    9.3 - param("radius"),
                    7.2 - param("radius"),
                    param("radius") * 2,
                    param("radius") * 2,
                ),
            )
        ],
        (20, 16),
    )
    target = sdf.ring((9.3, 7.2), 4.0, thickness=2.0)
    target_field = renderer.render(target).detach()
    target_image = renderer.render(target.fill("#aabbcc", "#112233").to_rgb()).detach()

    def objective():
        return (
            ((renderer.render(field) - target_field) ** 2).mean()
            + (
                (
                    renderer.render(field.fill("#aabbcc", "#112233").to_rgb())
                    - target_image
                )
                ** 2
            ).mean()
            + layout.size_loss({"ring": (8, 8)}, parameters) * 0.1
        )

    initial = objective().item()
    optimizer = Adam([radius], lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        objective().backward()
        optimizer.step()
    assert objective().item() < initial * 0.001
    assert radius.item() == pytest.approx(4, abs=0.03)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: sdf.ring((7.3, 5.4), 3.1, thickness=1.7),
        lambda: sdf.circle((7.3, 5.4), 3.1).skew((0.17, -0.13), center=(7.3, 5.4)),
    ],
)
def test_public_plugin_graph_matches_existing_gpu_renderer(factory):
    scene = factory()
    expected = DifferentiableRenderer((16, 12)).render(scene).numpy()
    with Canvas((16, 12)) as canvas:
        gpu = canvas.render(scene)
        actual = np.frombuffer(gpu.tex.read(), dtype=np.float32).reshape(12, 16)
    np.testing.assert_allclose(actual, expected, atol=0.04)


def test_arbitrary_host_callbacks_are_not_implicitly_differentiated():
    custom = Plugin(
        "opaque",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        render_func=lambda *args: None,
    )
    assert not supports_plugin(custom)
