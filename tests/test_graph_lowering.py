"""Native graph plans contain descriptors and fresh uniform bindings, not pixels."""



from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui import color, param, sdf, surface
from sdf_ui import field as vector_field
from sdf_ui.core.expressions import evaluate_expr
from sdf_ui.core.sdf import SDFTexture
from sdf_ui.core.texture import MultiOutputResult
from sdf_ui.gpu.nodes.base import (
    Plugin,
    PluginFamily,
    ShaderFile,
    TextureKind,
)
from sdf_ui.gpu.nodes.differentiation import ShaderDifferentiation
from sdf_ui.gpu.nodes.registry import registry
from sdf_ui.graph.lowering import GraphPlan, Pass, Port, lower_graph
from sdf_ui.learning import Parameter
from sdf_ui.learning.autodiff import Tensor
from sdf_ui.util import hex_col

_ALPHA_TEXT = 'alpha'

_CENTER_TEXT = 'center'

_CIRCLE_TEXT = 'circle'

_COLOR_TEXT = 'color'

_CORNER_RADIUS_TEXT = 'corner_radius'

_DISTANCE_TEXT = 'distance'

_FIELD_TEXT = 'field'

_IMAGE_TEXT = 'image'

_INFLATE_TEXT = 'inflate'

_NORMAL_TEXT = 'normal'

_OFFSET_TEXT = 'offset'

_RADIUS_TEXT = 'radius'

_SECOND_TEXT = 'second'

def resolver(parameters=None):
    def resolve(value):
        value = evaluate_expr(
            value, SimpleNamespace(size=(16, 12)), parameters or {}, track_grad=True
        )
        if isinstance(value, str) and value.startswith("#"):
            if len(value) == 4:
                value = "#" + "".join(c * 2 for c in value[1:])
            return hex_col(value)
        return value

    return resolve


def names(plan):
    return tuple(p.shader.name for p in plan.passes)


def execute_reference(plan, parameters=None, size=(8, 6)):
    """A test-only executor: production lowering must never evaluate contracts."""
    outputs = []
    for p in plan.passes:
        inputs = {
            arg.name: outputs[port.pass_index][port.output_index]
            for arg, port in zip(p.contract.inputs, p.inputs)
        }
        result = p.contract.evaluate(size, inputs, p.uniforms(resolver(parameters)))
        outputs.append(result if isinstance(result, tuple) else (result,))
    return {
        name: outputs[port.pass_index][port.output_index]
        for name, port in plan.outputs.items()
    }




class TestSlopLintCasesTwo:
        def test_shared_multi_output_dag_is_lowered_once_in_topological_order(self):
            base = sdf.circle((3.2, 4.1), param(_RADIUS_TEXT))
            field, mask = base.masked_union(base.translate((2, 0)))
            plan = lower_graph({_DISTANCE_TEXT: field, "mask": mask, "base": base, "same": field})
            assert names(plan) == (_CIRCLE_TEXT, "transform", "masked_union")
            assert plan.outputs[_DISTANCE_TEXT] == Port(2, 0)
            assert plan.outputs["mask"] == Port(2, 1, _COLOR_TEXT, "RGB")
            assert plan.outputs["same"] is plan.outputs[_DISTANCE_TEXT]
            assert plan.passes[1].inputs == (Port(0),)
            assert plan.passes[2].inputs == (Port(0), Port(1))
            for index, p in enumerate(plan.passes):
                assert all(port.pass_index < index for port in p.inputs)
            assert plan.parameter_names == (_RADIUS_TEXT,)

        @pytest.mark.parametrize(_RADIUS_TEXT, [2, param("r"), (1, 2, 3, 4)])
        def test_rect_scalar_corner_radius_and_vector_conventions(self, radius):
            r = Parameter("r", 2)
            plan = lower_graph({_FIELD_TEXT: sdf.rect((3, 4), (5, 6), corner_radius=radius)})
            u = plan.passes[0].uniforms(resolver({"r": r}))
            assert u[_OFFSET_TEXT] == (3, 4)
            assert u["size"] == (5, 6)
            if isinstance(u[_CORNER_RADIUS_TEXT], Tensor):
                np.testing.assert_array_equal(u[_CORNER_RADIUS_TEXT].numpy(), [2] * 4)
                u[_CORNER_RADIUS_TEXT].sum().backward()
                assert r.grad == 4
            else:
                assert u[_CORNER_RADIUS_TEXT] == (
                    (1, 2, 3, 4) if isinstance(radius, tuple) else (2,) * 4
                )

        def test_trainable_color_channels_and_light_float_cast_preserve_gradients(self):
            shade = Parameter("shade", 0.4)
            red = Parameter("red", 128)
            plan = lower_graph(
                {
                    "light": sdf.circle((3, 4), 2).light(
                        (param("red"), 64, 32), shade_background=param("shade")
                    )
                }
            )
            u = plan.passes[-1].uniforms(resolver({"shade": shade, "red": red}))
            assert u["shadeBackground"] is shade
            np.testing.assert_allclose(u["fgColor"].numpy(), [128 / 255, 64 / 255, 32 / 255, 1])
            (u["fgColor"].sum() + u["shadeBackground"]).backward()
            assert shade.grad == 1
            assert red.grad == pytest.approx(1 / 255)
            assert plan.outputs["light"].mode == "RGB"

        @pytest.mark.parametrize(
            "operation,kwargs,message",
            [
                ("scale", {"factor": param("p")}, "finite and non-zero"),
                ("skew", {"skew": (param("p"), 0.5)}, "singular"),
            ],
        )
        def test_transform_validation_uses_current_values(self, operation, kwargs, message):
            scene = getattr(sdf.circle((3, 4), 2), operation)(**kwargs)
            p = lower_graph({_FIELD_TEXT: scene}).passes[-1]
            p.uniforms(resolver({"p": 1}))
            with pytest.raises(ValueError, match=message):
                p.uniforms(resolver({"p": 0 if operation == "scale" else 2}))

        @pytest.mark.parametrize(
            "value", [param("n", 0), Parameter("n", 0), Tensor(0), -1, 0.5, True]
        )
        @pytest.mark.parametrize("selector", ["n", "base", _DISTANCE_TEXT, "index"])
        def test_control_flow_selectors_are_compile_time_constants(self, value, selector):
            field = sdf.circle((3, 4), 2)
            if selector == _DISTANCE_TEXT:
                scene = field.shadow(distance=value)
            elif selector == "index":
                multi = field.masked_union(field)
                scene = SDFTexture(
                    op="output", inputs=(multi,), params={"index": value, "kind": "sdf"}
                )
            else:
                scene = color.clear("#abc").blur(**{selector: value})
            with pytest.raises(ValueError, match="fixed compile-time integer"):
                lower_graph({"out": scene})

        def test_layer_ports_follow_contract_order_and_share_inserted_conversions(self):
            base, top = color.clear("#abc"), color.clear("#123")
            field = sdf.circle((3, 4), 2)
            plan = lower_graph(
                {
                    "overlay": base.over(top),
                    "filled": field.fill_from_texture(base),
                    "masked": base.mask(top, field.generate_mask()),
                }
            )
            assert names(plan).count("to_rgb") == 3
            overlay = plan.passes[plan.outputs["overlay"].pass_index]
            assert tuple(a.name for a in overlay.contract.inputs) == ("tex0", "tex1")
            assert overlay.inputs[1] == plan.passes[plan.outputs["filled"].pass_index].inputs[0]
            filled = plan.passes[plan.outputs["filled"].pass_index]
            assert tuple(a.name for a in filled.contract.inputs) == ("origTex", "sdf")
            assert tuple(p.kind for p in filled.inputs) == (_COLOR_TEXT, "sdf")
            assert all(p.mode == "RGB" for p in plan.outputs.values())

        def test_radial_gradient_uses_circle_and_fill_with_fresh_colors(self):
            scene = color.radial_gradient(
                (2, 3), param("fg"), "#123", inner=param("inner"), outer=10
            )
            plan = lower_graph({_IMAGE_TEXT: scene})
            assert names(plan) == (_CIRCLE_TEXT, "fill")
            assert plan.passes[0].uniforms(resolver({"fg": "#abc", "inner": 2})) == {
                _OFFSET_TEXT: (2, 3),
                _RADIUS_TEXT: 0,
            }
            u = plan.passes[1].uniforms(resolver({"fg": "#abc", "inner": 2}))
            assert u["first"] == 2 and u[_SECOND_TEXT] == 10
            np.testing.assert_allclose(u[_COLOR_TEXT].numpy(), [170 / 255, 187 / 255, 204 / 255, 1])

        def test_plan_fingerprint_is_stable_and_records_expression_parameter_names(self):
            radius = Parameter(_RADIUS_TEXT, 2)

            def scene():
                return sdf.circle((param("x", 3), 4), radius)

            node = scene()
            first = lower_graph({"z": node.fill("#abc"), "a": node})
            second = lower_graph({"a": node, "z": node.fill("#abc")})
            assert first.fingerprint == second.fingerprint
            assert first.parameter_names == (_RADIUS_TEXT, "x")
            radius.assign(5)
            assert (
                lower_graph({"a": scene(), "z": node.fill("#abc")}).fingerprint
                != first.fingerprint
            )  # sharing changed
            assert (
                lower_graph({"a": node, "z": node.fill("#abc")}).fingerprint
                == first.fingerprint
            )
            assert (
                lower_graph({"a": node, "z": node.fill("#def")}).fingerprint
                != first.fingerprint
            )

        def test_custom_shader_plugins_are_allowed_but_opaque_callbacks_are_rejected(self,
            monkeypatch,
        ):
            registry.ensure_loaded()
            shader = registry.get(_CIRCLE_TEXT).shader
            custom = Plugin(
                "custom_circle",
                PluginFamily.PRIMITIVE,
                TextureKind.SDF,
                params=(_CENTER_TEXT, _RADIUS_TEXT),
                shader=shader,
                make_uniforms=lambda p: {_OFFSET_TEXT: p[_CENTER_TEXT], _RADIUS_TEXT: p[_RADIUS_TEXT]},
            )
            monkeypatch.setitem(registry._plugins, custom.name, custom)
            assert names(lower_graph({_FIELD_TEXT: registry.build(custom.name, (3, 4), 2)})) == (
                _CIRCLE_TEXT,
            )
            for plugin in [
                replace(custom, render_func=lambda *args: None),
                replace(custom, shader=ShaderFile(_CIRCLE_TEXT, "custom.wgsl")),
            ]:
                monkeypatch.setitem(registry._plugins, custom.name, plugin)
                with pytest.raises(
                    ValueError, match="custom render function|known differentiation"
                ):
                    lower_graph({_FIELD_TEXT: registry.build(custom.name, (3, 4), 2)})
            builtin = registry.get("translate")
            monkeypatch.setitem(
                registry._plugins,
                "translate",
                replace(builtin, render_func=lambda *args: None, lowering_func=None),
            )
            with pytest.raises(ValueError, match="custom render function"):
                lower_graph({_FIELD_TEXT: sdf.circle((3, 4), 2).translate((1, 2))})

        def test_empty_plan_is_valid(self):
            plan = lower_graph({})
            assert plan.passes == () and dict(plan.outputs) == {}

        def test_reference_execution_accumulates_both_outputs_and_reuses_plan_after_update(self):
            radius = Parameter(_RADIUS_TEXT, 2)
            field = sdf.circle((2.1, 2.3), param(_RADIUS_TEXT))
            distance, mask = field.masked_union(field.translate((1.7, 0.3)))
            plan = lower_graph({_DISTANCE_TEXT: distance, "mask": mask})
            outputs = execute_reference(plan, {_RADIUS_TEXT: radius})
            (outputs[_DISTANCE_TEXT].mean() + outputs["mask"].mean()).backward()
            assert np.isfinite(radius.grad) and abs(radius.grad) > 0
            previous = outputs[_DISTANCE_TEXT].numpy()
            radius.assign(3)
            radius.zero_grad()
            fresh = execute_reference(plan, {_RADIUS_TEXT: radius})
            assert not np.allclose(previous, fresh[_DISTANCE_TEXT].numpy())
            fresh[_DISTANCE_TEXT].mean().backward()
            assert np.isfinite(radius.grad)

        @pytest.mark.parametrize("clockwise", [False, True])
        def test_convex_polygon_winding_and_uniform_gradients(self, clockwise):
            x = Parameter("x", 1)
            points = ((param("x"), 1), (6, 1), (6, 5), (param("x"), 5))
            if clockwise:
                points = tuple(reversed(points))
            plan = lower_graph({_FIELD_TEXT: sdf.convex_polygon(points)})
            output = execute_reference(plan, {"x": x})[_FIELD_TEXT]
            assert output.numpy()[3, 3] < 0 and output.numpy()[0, 0] > 0
            output.mean().backward()
            assert np.isfinite(x.grad) and abs(x.grad) > 0

        def test_uniform_arrays_are_snapshotted_but_tensor_bindings_stay_live(self):
            center = np.array([2, 3])
            radius = Parameter(_RADIUS_TEXT, 2)
            node = sdf.circle(center, radius)
            plan = lower_graph({_FIELD_TEXT: node})
            center[:] = 100
            node.params = ()
            radius.assign(4)
            uniforms = plan.passes[0].uniforms(resolver())
            np.testing.assert_array_equal(uniforms[_OFFSET_TEXT], [2, 3])
            assert uniforms[_RADIUS_TEXT] is radius
            assert not uniforms[_OFFSET_TEXT].flags.writeable

        def test_unsupported_host_rasterizers_do_not_fall_back_to_cpu_images(self):
            with pytest.raises(ValueError, match="no CPU fallback"):
                lower_graph({_FIELD_TEXT: sdf.polygon(((1, 1), (6, 1), (3, 5)))})


class TestSlopLintCasesOneA:
            def test_api_is_immutable_and_shader_contracts_are_real(self):
                node = sdf.disc((3, 4), 2)
                plan = lower_graph({"sdf": node})
                assert isinstance(plan, GraphPlan)
                assert isinstance(plan.passes, tuple)
                assert plan.outputs["sdf"] == Port(0)
                p = plan.passes[0]
                assert isinstance(p, Pass)
                assert p.shader is registry.get("disc").shader
                assert p.contract is p.shader.differentiation
                assert p.uniforms(resolver()) == {_OFFSET_TEXT: (3, 4), _RADIUS_TEXT: 2}
                with pytest.raises(FrozenInstanceError):
                    p.shader = None
                with pytest.raises(FrozenInstanceError):
                    plan.outputs["sdf"].output_index = 2
                with pytest.raises(TypeError):
                    plan.outputs["other"] = Port(0)

            @pytest.mark.parametrize(
                "value,expected",
                [
                    ("#abc", (170 / 255, 187 / 255, 204 / 255, 1)),
                    ((128, 64, 32), (128 / 255, 64 / 255, 32 / 255, 1)),
                    ((0.2, 0.3, 0.4, 0.5), (0.2, 0.3, 0.4, 0.5)),
                ],
            )
            def test_uniform_color_normalization_matches_host(self, value, expected):
                plan = lower_graph({_IMAGE_TEXT: sdf.circle((3, 4), 2).fill(value)})
                u = plan.passes[-1].uniforms(resolver())
                np.testing.assert_allclose(u[_COLOR_TEXT].numpy(), expected)
                assert set(u) == {_COLOR_TEXT, "background", _INFLATE_TEXT, "first", _SECOND_TEXT}
                assert plan.outputs[_IMAGE_TEXT].mode == "LAB"

            @pytest.mark.parametrize(
                "operation,expected",
                [
                    ("blur_9", ("clear_color", "blur_ver_9", "blur_hor_9") * 1),
                    ("blur_13", ("clear_color", "blur_ver_13", "blur_hor_13")),
                    ("blur", ("clear_color", "to_rgb", "blur_ver_9", "blur_hor_9", "to_lab")),
                ],
            )
            def test_blur_initial_iteration_and_color_modes(self, operation, expected):
                plan = lower_graph({_IMAGE_TEXT: getattr(color.clear("#123456"), operation)(n=0)})
                assert names(plan) == expected
                assert plan.outputs[_IMAGE_TEXT].mode == "LAB"
                rgb = color.clear("#123456").to_rgb()
                plan = lower_graph({_IMAGE_TEXT: getattr(rgb, operation)(n=1)})
                assert names(plan).count("to_rgb") == 1
                assert names(plan).count("to_lab") == 0
                assert len(plan.passes) == 6
                assert plan.outputs[_IMAGE_TEXT].mode == "RGB"

            def test_linear_gradient_geometry_is_recomputed_with_live_tensor_math(self):
                x = Parameter("x", 1)
                scene = color.linear_gradient((param("x"), 2), (4, 6), "#abc", "#123")
                plan = lower_graph({_IMAGE_TEXT: scene})
                assert names(plan) == ("line", "fill")
                u = plan.passes[0].uniforms(resolver({"x": x}))
                np.testing.assert_allclose(u["a"].numpy(), [-2, 6])
                np.testing.assert_allclose(u["b"].numpy(), [4, -2])
                fill = plan.passes[1].uniforms(resolver({"x": x}))
                assert fill[_SECOND_TEXT].item() == 5
                fill[_SECOND_TEXT].backward()
                assert x.grad == pytest.approx(-3 / 5)
                x.assign(4)
                assert plan.passes[1].uniforms(resolver({"x": x}))[_SECOND_TEXT].item() == 4

            def test_cycle_and_materialized_texture_fail_with_actionable_errors(self):
                node = sdf.circle((3, 4), 2).translate((1, 2))
                node.inputs = (node,)
                with pytest.raises(ValueError, match="Cycle"):
                    lower_graph({_FIELD_TEXT: node})

                class FakeTexture:
                    def release(self):
                        pass

                materialized = SDFTexture(tex=FakeTexture(), context=object())
                with pytest.raises(ValueError, match="Materialized.*Canvas"):
                    lower_graph({_FIELD_TEXT: materialized})
                materialized.tex = None

            def test_every_raw_shader_and_alias_lowers_through_public_plugins(self):
                field = sdf.circle((3.2, 2.3), 1.7)
                image = field.fill("#abc")
                sample = {
                    _CENTER_TEXT: (3.2, 2.3),
                    _RADIUS_TEXT: 1.7,
                    "radii": (2.1, 1.3),
                    "a": (0.2, 1.1),
                    "b": (6.1, 4.3),
                    "c": (3.1, 0.4),
                    "start_angle": 0.2,
                    "end_angle": 4.3,
                    _OFFSET_TEXT: (1.2, 0.3),
                    "size": (3.7, 4.1),
                    "point": (2.2, 3.1),
                    _NORMAL_TEXT: (0.3, 1),
                    "point_1": (0.1, 0.2),
                    "point_2": (6.2, 0.3),
                    "point_3": (3.1, 5.2),
                    _CORNER_RADIUS_TEXT: 0.2,
                    _COLOR_TEXT: "#abc",
                    "fg_color": "#abc",
                    _ALPHA_TEXT: 0.7,
                    "columns": 3,
                    "rows": 2,
                    "spacing": (4, 4),
                    "linear_rgba": (0.8, 0.2, 0.1, 1),
                    "linear_background": (0.02, 0.01, 0.01, 1),
                    "inner": -0.5,
                    "outer": 0.5,
                    "stops": 0.2,
                    "grain_amount": 0.07,
                    "grain_size": 1.35,
                    "threshold": 0.5,
                    "amount": 0.2,
                    "tone": 0.3,
                    "dither": 0.5,
                    "origin": (0.1, 0.2),
                    "slope": (0.1, 0.2),
                    "sigma": (1.4, 2.8),
                    "amplitude": 1.7,
                }
                registry.ensure_loaded()
                plan = _lower_every_shader(field, image, sample)
                descriptors = {descriptor.name for descriptor in registry.shader_files()}
                assert {p.shader.name for p in plan.passes} == descriptors
                _assert_lowered_shader_contracts(plan)

            def test_polygon_topology_selectors_cannot_be_frozen_from_expressions_or_tensors(self):
                for sides in (param("sides", 6), Tensor(6)):
                    with pytest.raises(ValueError, match="fixed compile-time integer"):
                        lower_graph({_FIELD_TEXT: sdf.ngon((3, 4), 2, sides=sides)})
                with pytest.raises(ValueError, match="fixed compile-time vertex count"):
                    lower_graph({_FIELD_TEXT: sdf.convex_polygon(param("points"))})

            def test_wrong_multi_output_index_kind_and_input_types_are_rejected(self):
                field = sdf.circle((2, 3), 1)
                multi = field.masked_union(field)
                for index, kind in ((2, "sdf"), (1, "sdf")):
                    node = SDFTexture(
                        op="output", inputs=(multi,), params={"index": index, "kind": kind}
                    )
                    with pytest.raises(ValueError, match="output index|output kind"):
                        lower_graph({_FIELD_TEXT: node})
                node = SDFTexture(op="union", inputs=(field, color.clear("#abc")))
                with pytest.raises(ValueError, match="requires sdf inputs"):
                    lower_graph({_FIELD_TEXT: node})


class TestSlopLintCasesOneB:
            def test_uniforms_resolve_fresh_and_do_not_retain_a_previous_tape(self):
                radius = Parameter(_RADIUS_TEXT, 2)
                plan = lower_graph({_FIELD_TEXT: sdf.circle((3, 4), param(_RADIUS_TEXT) * 2)})
                p = plan.passes[0]
                first = p.uniforms(resolver({_RADIUS_TEXT: radius}))
                first[_RADIUS_TEXT].backward()
                assert radius.grad == 2
                radius.assign(5)
                radius.zero_grad()
                second = p.uniforms(resolver({_RADIUS_TEXT: radius}))
                assert second is not first
                assert second[_RADIUS_TEXT] is not first[_RADIUS_TEXT]
                assert second[_RADIUS_TEXT].item() == 10
                second[_RADIUS_TEXT].backward()
                assert radius.grad == 2
                assert first[_RADIUS_TEXT].item() == 4

            @pytest.mark.parametrize(
                "operation,parameter,expected",
                [
                    (
                        "translate",
                        {_OFFSET_TEXT: (param("p"), 1)},
                        {_OFFSET_TEXT: (3, 1), "transform_kind": 0},
                    ),
                    (
                        "scale",
                        {"factor": param("p")},
                        {_CENTER_TEXT: (0, 0), "factor": 3, "transform_kind": 1},
                    ),
                    (
                        "rotate",
                        {"angle": param("p")},
                        {_CENTER_TEXT: (0, 0), "angle": 3, "transform_kind": 2},
                    ),
                    (
                        "skew",
                        {"skew": (param("p"), 0.1)},
                        {_CENTER_TEXT: (0, 0), "skew": (3, 0.1), "transform_kind": 3},
                    ),
                ],
            )
            def test_transforms_use_symbolic_uniforms(self, operation, parameter, expected):
                field = sdf.circle((3, 4), 2)
                plan = lower_graph({_FIELD_TEXT: getattr(field, operation)(**parameter)})
                u = plan.passes[-1].uniforms(resolver({"p": 3}))
                assert u == expected
                assert names(plan) == (_CIRCLE_TEXT, "transform")

            def test_shadow_expands_to_mask_rgb_blur_and_transparency(self):
                plan = lower_graph(
                    {
                        "shadow": sdf.circle((3, 4), 2).shadow(
                            distance=1, inflate=param(_INFLATE_TEXT), transparency=param(_ALPHA_TEXT)
                        )
                    }
                )
                assert names(plan) == (
                    _CIRCLE_TEXT,
                    "fill",
                    "to_rgb",
                    "blur_ver_9",
                    "blur_hor_9",
                    "blur_ver_9",
                    "blur_hor_9",
                    "transparency",
                )
                assert (
                    plan.passes[1].uniforms(resolver({_INFLATE_TEXT: 2, _ALPHA_TEXT: 0.3}))[_INFLATE_TEXT] == 2
                )
                assert plan.passes[-1].uniforms(resolver({_INFLATE_TEXT: 2, _ALPHA_TEXT: 0.3})) == {
                    _ALPHA_TEXT: 0.3
                }
                assert plan.outputs["shadow"].mode == "RGB"

            def test_lowering_performs_no_shader_evaluation_or_host_rendering(self, monkeypatch):
                def forbidden(*args, **kwargs):
                    raise AssertionError("pixel evaluation during lowering")

                monkeypatch.setattr(ShaderDifferentiation, "evaluate", forbidden)
                monkeypatch.setattr(Plugin, "render", forbidden)
                scene = sdf.circle((3, 4), param(_RADIUS_TEXT)).shadow(distance=0)
                plan = lower_graph({"out": scene})
                assert plan.passes[0].uniforms(resolver({_RADIUS_TEXT: 2}))[_RADIUS_TEXT] == 2

            @pytest.mark.parametrize(
                "value",
                [
                    [],
                    {"out": 42},
                    {"out": MultiOutputResult("masked_union")},
                    {1: sdf.circle((1, 2), 3)},
                ],
            )
            def test_invalid_root_outputs_are_rejected(self, value):
                with pytest.raises(TypeError):
                    lower_graph(value)

            @pytest.mark.parametrize("operation", ["diamond", "rhombus", "parallelogram", "ngon"])
            def test_constructed_polygons_lower_to_native_planes_with_live_geometry(self, operation):
                radius = Parameter(_RADIUS_TEXT, 2)
                if operation == "ngon":
                    scene = sdf.ngon((3.1, 2.3), param(_RADIUS_TEXT), sides=5)
                    count = 5
                elif operation == "parallelogram":
                    scene = sdf.parallelogram((3.1, 2.3), (param(_RADIUS_TEXT) * 2, 3), skew=0.4)
                    count = 4
                else:
                    scene = getattr(sdf, operation)((3.1, 2.3), (param(_RADIUS_TEXT), 1.3))
                    count = 4
                plan = lower_graph({_FIELD_TEXT: scene})
                assert names(plan) == ("half_plane",) * count + ("intersection",) * (count - 1)
                uniform = plan.passes[0].uniforms(resolver({_RADIUS_TEXT: radius}))
                assert uniform["point"].shape == (2,) and uniform[_NORMAL_TEXT].shape == (2,)
                assert radius in uniform[_NORMAL_TEXT]._versions
                output = execute_reference(plan, {_RADIUS_TEXT: radius})[_FIELD_TEXT]
                output.mean().backward()
                assert np.isfinite(radius.grad) and abs(radius.grad) > 0
                radius.assign(3)
                updated = plan.passes[0].uniforms(resolver({_RADIUS_TEXT: radius}))
                assert not np.allclose(uniform[_NORMAL_TEXT].numpy(), updated[_NORMAL_TEXT].numpy())

            def test_resolved_uniform_tensors_keep_dependency_versions_for_the_executor(self):
                red = Parameter("red", 0.5)
                radius = Parameter(_RADIUS_TEXT, 2)
                derived = red * 0.5
                plan = lower_graph(
                    {
                        _IMAGE_TEXT: sdf.circle((2, 3), param(_RADIUS_TEXT)).fill(
                            (derived, 0.2, 0.3, param("red") * 2 / 5)
                        )
                    }
                )
                assert plan.parameter_names == (_RADIUS_TEXT, "red")
                geometry = plan.passes[0].uniforms(resolver({_RADIUS_TEXT: radius}))
                fill = plan.passes[1].uniforms(resolver({"red": red}))
                assert geometry[_RADIUS_TEXT]._versions == {radius: radius._version}
                assert fill[_COLOR_TEXT]._versions == {red: red._version}


def _lower_every_shader(field, image, sample):
    sources = {
        TextureKind.SDF: field,
        TextureKind.COLOR: image,
        TextureKind.FIELD: vector_field.constant((0.1, 0.2)),
        TextureKind.HEIGHT: surface.ramp((0, 0), (0.1, 0.2)),
    }
    nodes = {
        plugin.name: _build_required_plugin_node(plugin, sources, sample)
        for plugin in registry._plugins.values()
        if plugin.shader is not None and plugin.render_func is None
    }
    nodes.update(
        {
            "transform": field.rotate(0.2),
            "filled": field.fill_from_texture(image),
            "overlay": image.over(image),
            "masked": image.mask(image, image),
            "multiplied": image.multiply(image),
            "blur9": image.blur_9(),
            "blur13": image.blur_13(),
        }
    )
    distance, mask = field.masked_union(field)
    nodes.update(distance=distance, mask=mask)
    return lower_graph(nodes)


def _build_required_plugin_node(plugin, sources, sample):
    args = tuple(sources[kind] for kind in plugin.input_kinds)
    required = {
        key: sample[key]
        for key in plugin.params
        if key not in (plugin.defaults or {})
    }
    return registry.build(plugin.name, *args, **required)


def _assert_lowered_shader_contracts(plan):
    for operation in plan.passes:
        uniforms = operation.uniforms(resolver())
        allowed = {argument.name for argument in operation.contract.parameters}
        assert set(uniforms) <= allowed
        assert len(operation.inputs) == len(operation.contract.inputs)
        for argument, value in zip(operation.contract.inputs, operation.inputs):
            assert value.kind == argument.texture_kind
