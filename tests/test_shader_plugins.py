"""Shader contracts stay usable through public render-graph conventions."""



import numpy as np
import pytest

from sdf_ui import Canvas, color, param, sdf
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.learning import (
    Adam,
    Bounds,
    DifferentiableRenderer,
    Element,
    Layout,
    Parameter,
)
from sdf_ui.learning.shaders.plugins import supports_plugin

_RADIUS_TEXT = 'radius'

class TestSlopLintCasesOne:
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
                lambda p: sdf.circle((7.3, 5.4), 3.1).skew((p * 1 / 10, -13 / 100), center=(7.3, 5.4)),
            ],
        )
        def test_new_primitive_adapters_preserve_parameter_expression_gradients(self, factory):
            parameter = Parameter("p", 0.7)
            renderer = DifferentiableRenderer((16, 12), {"p": parameter})
            output = renderer.render(factory(param("p")))
            assert output.shape == (12, 16)
            (output * np.linspace(0.1, 1.0, 192).reshape(12, 16)).mean().backward()
            assert parameter.grad is not None and np.isfinite(parameter.grad)

        @pytest.mark.parametrize("operation", ["blur", "blur_9", "blur_13", "shadow"])
        def test_composite_passes_propagate_source_geometry_gradients(self, operation):
            radius = Parameter(_RADIUS_TEXT, 3.17)
            field = sdf.circle((7.23, 5.61), radius)
            scene = (
                field.shadow(distance=0)
                if operation == "shadow"
                else getattr(field.fill("#b97654", "#123456"), operation)(n=0)
            )
            output = DifferentiableRenderer((16, 12)).render(scene.to_rgb())
            output.mean().backward()
            assert radius.grad is not None and np.isfinite(radius.grad)

        def test_mask_layer_blend_modes_execute_on_gpu(self):
            base = color.clear((1.0, 0.0, 0.0, 0.6)).to_rgb()
            top = color.clear((0.0, 0.0, 1.0, 0.3)).to_rgb()
            mask = color.clear((0.5, 0.5, 0.5, 1.0)).to_rgb()
            with Canvas((8, 8)) as canvas:
                rendered = {
                    mode: base.mask(top, mask, mode=mode).to_rgb().render(canvas)
                    for mode in ("mix", "multiply", "screen", "over")
                }
                results = {mode: image.tex.read() for mode, image in rendered.items()}
            center = (4, 4)
            pixels = {mode: image[center] / 255.0 for mode, image in results.items()}
            assert pixels["mix"][3] == pytest.approx(0.45, abs=1 / 255)
            assert pixels["over"][3] == pytest.approx(0.51, abs=1 / 255)
            assert pixels["multiply"][0] == pytest.approx(0.0, abs=1 / 255)
            assert pixels["screen"][3] > pixels["mix"][3]

        def test_masked_union_outputs_both_reach_shared_parameter_leaves(self):
            radius = Parameter(_RADIUS_TEXT, 2.7)
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

        @pytest.mark.parametrize(
            "factory",
            [
                lambda: sdf.ring((7.3, 5.4), 3.1, thickness=1.7),
                lambda: sdf.circle((7.3, 5.4), 3.1).skew((0.17, -13 / 100), center=(7.3, 5.4)),
            ],
        )
        def test_public_plugin_graph_matches_existing_gpu_renderer(self, factory):
            scene = factory()
            expected = DifferentiableRenderer((16, 12)).render(scene).numpy()
            with Canvas((16, 12)) as canvas:
                gpu = canvas.render(scene)
                actual = np.frombuffer(gpu.tex.read(), dtype=np.float32).reshape(12, 16)
            np.testing.assert_allclose(actual, expected, atol=0.04)


class TestSlopLintCasesTwoA:
            @pytest.mark.parametrize("operation", ["outline", "isolines", "light"])
            def test_new_shading_parameters_receive_gradients(self, operation):
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

            def test_stroke_width_creates_a_composable_color_layer(self):
                field = sdf.circle((16, 16), 8)
                thin = field.stroke(2, (1.0, 0.0, 0.0, 1.0)).to_rgb()
                thick = field.stroke(6, (1.0, 0.0, 0.0, 1.0)).to_rgb()
                combined = thin.alpha_overlay(thick).to_rgb()
                with Canvas((32, 32)) as canvas:
                    rendered_thin = thin.render(canvas)
                    rendered_thick = thick.render(canvas)
                    rendered_combined = combined.render(canvas)
                    thin_pixels = rendered_thin.tex.read()
                    thick_pixels = rendered_thick.tex.read()
                    combined_pixels = rendered_combined.tex.read()
                # Sample 2.5 units inside the zero contour: the wider stroke reaches here.
                assert thick_pixels[16, 21, 3] > thin_pixels[16, 21, 3]
                assert combined_pixels[16, 21, 3] >= thick_pixels[16, 21, 3]

            def test_arbitrary_host_callbacks_are_not_implicitly_differentiated(self):
                custom = Plugin(
                    "opaque",
                    PluginFamily.PRIMITIVE,
                    TextureKind.SDF,
                    render_func=lambda *args: None,
                )
                assert not supports_plugin(custom)


class TestSlopLintCasesTwoB:
            def test_fill_and_mask_adapters_convert_lab_inputs_to_rgb(self):
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

            def test_combined_pixel_field_and_geometry_objective_optimizes_shared_parameters(self):
                radius = Parameter(_RADIUS_TEXT, 2.0)
                parameters = {_RADIUS_TEXT: radius}
                renderer = DifferentiableRenderer((20, 16), parameters)
                field = sdf.ring((9.3, 7.2), param(_RADIUS_TEXT), thickness=2.0)
                layout = Layout(
                    [
                        Element(
                            "ring",
                            Bounds(
                                93 / 10 - param(_RADIUS_TEXT),
                                36 / 5 - param(_RADIUS_TEXT),
                                param(_RADIUS_TEXT) * 2,
                                param(_RADIUS_TEXT) * 2,
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
                        + layout.size_loss({"ring": (8, 8)}, parameters) * 1 / 10
                    )

                initial = objective().item()
                optimizer = Adam([radius], lr=0.1)
                for _ in range(100):
                    optimizer.zero_grad()
                    objective().backward()
                    optimizer.step()
                assert objective().item() < initial * 1 / 1_000
                assert radius.item() == pytest.approx(4, abs=0.03)
