"""Native graph plans contain descriptors and fresh uniform bindings, not pixels."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui import color, param, sdf
from sdf_ui.backends.opengl.nodes.differentiation import ShaderDifferentiation
from sdf_ui.core.expressions import evaluate_expr
from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, ShaderFile, TextureKind
from sdf_ui.backends.opengl.nodes.registry import registry
from sdf_ui.core.sdf import SDFTexture
from sdf_ui.core.texture import MultiOutputResult
from sdf_ui.graph.lowering import GraphPlan, Pass, Port, lower_graph
from sdf_ui.learning import Parameter
from sdf_ui.learning.autodiff import Tensor
from sdf_ui.util import hex_col


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


def test_api_is_immutable_and_shader_contracts_are_real():
    node = sdf.disc((3, 4), 2)
    plan = lower_graph({"sdf": node})
    assert isinstance(plan, GraphPlan)
    assert isinstance(plan.passes, tuple)
    assert plan.outputs["sdf"] == Port(0)
    p = plan.passes[0]
    assert isinstance(p, Pass)
    assert p.shader is registry.get("disc").shader
    assert p.contract is p.shader.differentiation
    assert p.uniforms(resolver()) == {"offset": (3, 4), "radius": 2}
    with pytest.raises(FrozenInstanceError):
        p.shader = None
    with pytest.raises(FrozenInstanceError):
        plan.outputs["sdf"].output_index = 2
    with pytest.raises(TypeError):
        plan.outputs["other"] = Port(0)


def test_shared_multi_output_dag_is_lowered_once_in_topological_order():
    base = sdf.circle((3.2, 4.1), param("radius"))
    field, mask = base.masked_union(base.translate((2, 0)))
    plan = lower_graph({"distance": field, "mask": mask, "base": base, "same": field})
    assert names(plan) == ("circle", "transform", "masked_union")
    assert plan.outputs["distance"] == Port(2, 0)
    assert plan.outputs["mask"] == Port(2, 1, "color", "RGB")
    assert plan.outputs["same"] is plan.outputs["distance"]
    assert plan.passes[1].inputs == (Port(0),)
    assert plan.passes[2].inputs == (Port(0), Port(1))
    for index, p in enumerate(plan.passes):
        assert all(port.pass_index < index for port in p.inputs)
    assert plan.parameter_names == ("radius",)


def test_uniforms_resolve_fresh_and_do_not_retain_a_previous_tape():
    radius = Parameter("radius", 2)
    plan = lower_graph({"field": sdf.circle((3, 4), param("radius") * 2)})
    p = plan.passes[0]
    first = p.uniforms(resolver({"radius": radius}))
    first["radius"].backward()
    assert radius.grad == 2
    radius.assign(5)
    radius.zero_grad()
    second = p.uniforms(resolver({"radius": radius}))
    assert second is not first
    assert second["radius"] is not first["radius"]
    assert second["radius"].item() == 10
    second["radius"].backward()
    assert radius.grad == 2
    assert first["radius"].item() == 4


@pytest.mark.parametrize("radius", [2, param("r"), (1, 2, 3, 4)])
def test_rect_scalar_corner_radius_and_vector_conventions(radius):
    r = Parameter("r", 2)
    plan = lower_graph({"field": sdf.rect((3, 4), (5, 6), corner_radius=radius)})
    u = plan.passes[0].uniforms(resolver({"r": r}))
    assert u["offset"] == (3, 4)
    assert u["size"] == (5, 6)
    if isinstance(u["corner_radius"], Tensor):
        np.testing.assert_array_equal(u["corner_radius"].numpy(), [2] * 4)
        u["corner_radius"].sum().backward()
        assert r.grad == 4
    else:
        assert u["corner_radius"] == (
            (1, 2, 3, 4) if isinstance(radius, tuple) else (2,) * 4
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("#abc", (170 / 255, 187 / 255, 204 / 255, 1)),
        ((128, 64, 32), (128 / 255, 64 / 255, 32 / 255, 1)),
        ((0.2, 0.3, 0.4, 0.5), (0.2, 0.3, 0.4, 0.5)),
    ],
)
def test_uniform_color_normalization_matches_host(value, expected):
    plan = lower_graph({"image": sdf.circle((3, 4), 2).fill(value)})
    u = plan.passes[-1].uniforms(resolver())
    np.testing.assert_allclose(u["color"].numpy(), expected)
    assert set(u) == {"color", "background", "inflate", "first", "second"}
    assert plan.outputs["image"].mode == "LAB"


def test_trainable_color_channels_and_light_float_cast_preserve_gradients():
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
    "operation,parameter,expected",
    [
        (
            "translate",
            {"offset": (param("p"), 1)},
            {"offset": (3, 1), "transform_kind": 0},
        ),
        (
            "scale",
            {"factor": param("p")},
            {"center": (0, 0), "factor": 3, "transform_kind": 1},
        ),
        (
            "rotate",
            {"angle": param("p")},
            {"center": (0, 0), "angle": 3, "transform_kind": 2},
        ),
        (
            "skew",
            {"skew": (param("p"), 0.1)},
            {"center": (0, 0), "skew": (3, 0.1), "transform_kind": 3},
        ),
    ],
)
def test_transforms_use_symbolic_uniforms(operation, parameter, expected):
    field = sdf.circle((3, 4), 2)
    plan = lower_graph({"field": getattr(field, operation)(**parameter)})
    u = plan.passes[-1].uniforms(resolver({"p": 3}))
    assert u == expected
    assert names(plan) == ("circle", "transform")


@pytest.mark.parametrize(
    "operation,kwargs,message",
    [
        ("scale", {"factor": param("p")}, "finite and non-zero"),
        ("skew", {"skew": (param("p"), 0.5)}, "singular"),
    ],
)
def test_transform_validation_uses_current_values(operation, kwargs, message):
    scene = getattr(sdf.circle((3, 4), 2), operation)(**kwargs)
    p = lower_graph({"field": scene}).passes[-1]
    p.uniforms(resolver({"p": 1}))
    with pytest.raises(ValueError, match=message):
        p.uniforms(resolver({"p": 0 if operation == "scale" else 2}))


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("blur_9", ("clear_color", "blur_ver_9", "blur_hor_9") * 1),
        ("blur_13", ("clear_color", "blur_ver_13", "blur_hor_13")),
        ("blur", ("clear_color", "to_rgb", "blur_ver_9", "blur_hor_9", "to_lab")),
    ],
)
def test_blur_initial_iteration_and_color_modes(operation, expected):
    plan = lower_graph({"image": getattr(color.clear("#123456"), operation)(n=0)})
    assert names(plan) == expected
    assert plan.outputs["image"].mode == "LAB"
    rgb = color.clear("#123456").to_rgb()
    plan = lower_graph({"image": getattr(rgb, operation)(n=1)})
    assert names(plan).count("to_rgb") == 1
    assert names(plan).count("to_lab") == 0
    assert len(plan.passes) == 6
    assert plan.outputs["image"].mode == "RGB"


@pytest.mark.parametrize(
    "value", [param("n", 0), Parameter("n", 0), Tensor(0), -1, 0.5, True]
)
@pytest.mark.parametrize("selector", ["n", "base", "distance", "index"])
def test_control_flow_selectors_are_compile_time_constants(value, selector):
    field = sdf.circle((3, 4), 2)
    if selector == "distance":
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


def test_shadow_expands_to_mask_rgb_blur_and_transparency():
    plan = lower_graph(
        {
            "shadow": sdf.circle((3, 4), 2).shadow(
                distance=1, inflate=param("inflate"), transparency=param("alpha")
            )
        }
    )
    assert names(plan) == (
        "circle",
        "fill",
        "to_rgb",
        "blur_ver_9",
        "blur_hor_9",
        "blur_ver_9",
        "blur_hor_9",
        "transparency",
    )
    assert (
        plan.passes[1].uniforms(resolver({"inflate": 2, "alpha": 0.3}))["inflate"] == 2
    )
    assert plan.passes[-1].uniforms(resolver({"inflate": 2, "alpha": 0.3})) == {
        "alpha": 0.3
    }
    assert plan.outputs["shadow"].mode == "RGB"


def test_layer_ports_follow_contract_order_and_share_inserted_conversions():
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
    assert tuple(p.kind for p in filled.inputs) == ("color", "sdf")
    assert all(p.mode == "RGB" for p in plan.outputs.values())


def test_linear_gradient_geometry_is_recomputed_with_live_tensor_math():
    x = Parameter("x", 1)
    scene = color.linear_gradient((param("x"), 2), (4, 6), "#abc", "#123")
    plan = lower_graph({"image": scene})
    assert names(plan) == ("line", "fill")
    u = plan.passes[0].uniforms(resolver({"x": x}))
    np.testing.assert_allclose(u["a"].numpy(), [-2, 6])
    np.testing.assert_allclose(u["b"].numpy(), [4, -2])
    fill = plan.passes[1].uniforms(resolver({"x": x}))
    assert fill["second"].item() == 5
    fill["second"].backward()
    assert x.grad == pytest.approx(-3 / 5)
    x.assign(4)
    assert plan.passes[1].uniforms(resolver({"x": x}))["second"].item() == 4


def test_radial_gradient_uses_circle_and_fill_with_fresh_colors():
    scene = color.radial_gradient(
        (2, 3), param("fg"), "#123", inner=param("inner"), outer=10
    )
    plan = lower_graph({"image": scene})
    assert names(plan) == ("circle", "fill")
    assert plan.passes[0].uniforms(resolver({"fg": "#abc", "inner": 2})) == {
        "offset": (2, 3),
        "radius": 0,
    }
    u = plan.passes[1].uniforms(resolver({"fg": "#abc", "inner": 2}))
    assert u["first"] == 2 and u["second"] == 10
    np.testing.assert_allclose(u["color"].numpy(), [170 / 255, 187 / 255, 204 / 255, 1])


def test_lowering_performs_no_shader_evaluation_or_host_rendering(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pixel evaluation during lowering")

    monkeypatch.setattr(ShaderDifferentiation, "evaluate", forbidden)
    monkeypatch.setattr(Plugin, "render", forbidden)
    scene = sdf.circle((3, 4), param("radius")).shadow(distance=0)
    plan = lower_graph({"out": scene})
    assert plan.passes[0].uniforms(resolver({"radius": 2}))["radius"] == 2


def test_plan_fingerprint_is_stable_and_records_expression_parameter_names():
    radius = Parameter("radius", 2)

    def scene():
        return sdf.circle((param("x", 3), 4), radius)

    node = scene()
    first = lower_graph({"z": node.fill("#abc"), "a": node})
    second = lower_graph({"a": node, "z": node.fill("#abc")})
    assert first.fingerprint == second.fingerprint
    assert first.parameter_names == ("radius", "x")
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


def test_cycle_and_materialized_texture_fail_with_actionable_errors():
    node = sdf.circle((3, 4), 2).translate((1, 2))
    node.inputs = (node,)
    with pytest.raises(ValueError, match="Cycle"):
        lower_graph({"field": node})

    class FakeTexture:
        def release(self):
            pass

    materialized = SDFTexture(tex=FakeTexture(), context=object())
    with pytest.raises(
        ValueError, match="Materialized.*Canvas"
    ):
        lower_graph({"field": materialized})
    materialized.tex = None


def test_custom_shader_plugins_are_allowed_but_opaque_callbacks_are_rejected(
    monkeypatch,
):
    registry.ensure_loaded()
    shader = registry.get("circle").shader
    custom = Plugin(
        "custom_circle",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        params=("center", "radius"),
        shader=shader,
        make_uniforms=lambda p: {"offset": p["center"], "radius": p["radius"]},
    )
    monkeypatch.setitem(registry._plugins, custom.name, custom)
    assert names(lower_graph({"field": registry.build(custom.name, (3, 4), 2)})) == (
        "circle",
    )
    for plugin in [
        replace(custom, render_func=lambda *args: None),
        replace(custom, shader=ShaderFile("circle", "custom.glsl")),
    ]:
        monkeypatch.setitem(registry._plugins, custom.name, plugin)
        with pytest.raises(
            ValueError, match="custom render function|known differentiation"
        ):
            lower_graph({"field": registry.build(custom.name, (3, 4), 2)})
    builtin = registry.get("translate")
    monkeypatch.setitem(
        registry._plugins, "translate", replace(builtin, render_func=lambda *args: None, lowering_func=None)
    )
    with pytest.raises(ValueError, match="custom render function"):
        lower_graph({"field": sdf.circle((3, 4), 2).translate((1, 2))})


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"out": 42},
        {"out": MultiOutputResult("masked_union")},
        {1: sdf.circle((1, 2), 3)},
    ],
)
def test_invalid_root_outputs_are_rejected(value):
    with pytest.raises(TypeError):
        lower_graph(value)


def test_empty_plan_is_valid():
    plan = lower_graph({})
    assert plan.passes == () and dict(plan.outputs) == {}


def test_every_raw_shader_and_alias_lowers_through_public_plugins():
    field = sdf.circle((3.2, 2.3), 1.7)
    image = field.fill("#abc")
    sample = {
        "center": (3.2, 2.3),
        "radius": 1.7,
        "radii": (2.1, 1.3),
        "a": (0.2, 1.1),
        "b": (6.1, 4.3),
        "c": (3.1, 0.4),
        "start_angle": 0.2,
        "end_angle": 4.3,
        "offset": (1.2, 0.3),
        "size": (3.7, 4.1),
        "point": (2.2, 3.1),
        "normal": (0.3, 1),
        "point_1": (0.1, 0.2),
        "point_2": (6.2, 0.3),
        "point_3": (3.1, 5.2),
        "corner_radius": 0.2,
        "color": "#abc",
        "fg_color": "#abc",
        "alpha": 0.7,
    }
    registry.ensure_loaded()
    nodes = {}
    for plugin in registry._plugins.values():
        if plugin.shader is None or plugin.render_func is not None:
            continue
        args = tuple(field if kind == "sdf" else image for kind in plugin.input_kinds)
        required = {
            key: sample[key]
            for key in plugin.params
            if key not in (plugin.defaults or {})
        }
        nodes[plugin.name] = registry.build(plugin.name, *args, **required)
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
    plan = lower_graph(nodes)
    descriptors = set(registry.shader_files())
    assert len(descriptors) == 43
    assert {p.shader for p in plan.passes} == descriptors
    for p in plan.passes:
        u = p.uniforms(resolver())
        assert set(u) <= {a.name for a in p.contract.parameters}
        assert len(p.inputs) == len(p.contract.inputs)
        for arg, value in zip(p.contract.inputs, p.inputs):
            assert value.kind == ("sdf" if arg.kind == "r32f" else "color")


def test_reference_execution_accumulates_both_outputs_and_reuses_plan_after_update():
    radius = Parameter("radius", 2)
    field = sdf.circle((2.1, 2.3), param("radius"))
    distance, mask = field.masked_union(field.translate((1.7, 0.3)))
    plan = lower_graph({"distance": distance, "mask": mask})
    outputs = execute_reference(plan, {"radius": radius})
    (outputs["distance"].mean() + outputs["mask"].mean()).backward()
    assert np.isfinite(radius.grad) and abs(radius.grad) > 0
    previous = outputs["distance"].numpy()
    radius.assign(3)
    radius.zero_grad()
    fresh = execute_reference(plan, {"radius": radius})
    assert not np.allclose(previous, fresh["distance"].numpy())
    fresh["distance"].mean().backward()
    assert np.isfinite(radius.grad)


@pytest.mark.parametrize("operation", ["diamond", "rhombus", "parallelogram", "ngon"])
def test_constructed_polygons_lower_to_native_planes_with_live_geometry(operation):
    radius = Parameter("radius", 2)
    if operation == "ngon":
        scene = sdf.ngon((3.1, 2.3), param("radius"), sides=5)
        count = 5
    elif operation == "parallelogram":
        scene = sdf.parallelogram((3.1, 2.3), (param("radius") * 2, 3), skew=0.4)
        count = 4
    else:
        scene = getattr(sdf, operation)((3.1, 2.3), (param("radius"), 1.3))
        count = 4
    plan = lower_graph({"field": scene})
    assert names(plan) == ("half_plane",) * count + ("intersection",) * (count - 1)
    uniform = plan.passes[0].uniforms(resolver({"radius": radius}))
    assert uniform["point"].shape == (2,) and uniform["normal"].shape == (2,)
    assert radius in uniform["normal"]._versions
    output = execute_reference(plan, {"radius": radius})["field"]
    output.mean().backward()
    assert np.isfinite(radius.grad) and abs(radius.grad) > 0
    radius.assign(3)
    updated = plan.passes[0].uniforms(resolver({"radius": radius}))
    assert not np.allclose(uniform["normal"].numpy(), updated["normal"].numpy())


@pytest.mark.parametrize("clockwise", [False, True])
def test_convex_polygon_winding_and_uniform_gradients(clockwise):
    x = Parameter("x", 1)
    points = ((param("x"), 1), (6, 1), (6, 5), (param("x"), 5))
    if clockwise:
        points = tuple(reversed(points))
    plan = lower_graph({"field": sdf.convex_polygon(points)})
    output = execute_reference(plan, {"x": x})["field"]
    assert output.numpy()[3, 3] < 0 and output.numpy()[0, 0] > 0
    output.mean().backward()
    assert np.isfinite(x.grad) and abs(x.grad) > 0


def test_polygon_topology_selectors_cannot_be_frozen_from_expressions_or_tensors():
    for sides in (param("sides", 6), Tensor(6)):
        with pytest.raises(ValueError, match="fixed compile-time integer"):
            lower_graph({"field": sdf.ngon((3, 4), 2, sides=sides)})
    with pytest.raises(ValueError, match="fixed compile-time vertex count"):
        lower_graph({"field": sdf.convex_polygon(param("points"))})


def test_uniform_arrays_are_snapshotted_but_tensor_bindings_stay_live():
    center = np.array([2, 3])
    radius = Parameter("radius", 2)
    node = sdf.circle(center, radius)
    plan = lower_graph({"field": node})
    center[:] = 100
    node.params = ()
    radius.assign(4)
    uniforms = plan.passes[0].uniforms(resolver())
    np.testing.assert_array_equal(uniforms["offset"], [2, 3])
    assert uniforms["radius"] is radius
    assert not uniforms["offset"].flags.writeable


def test_resolved_uniform_tensors_keep_dependency_versions_for_the_executor():
    red = Parameter("red", 0.5)
    radius = Parameter("radius", 2)
    derived = red * 0.5
    plan = lower_graph(
        {
            "image": sdf.circle((2, 3), param("radius")).fill(
                (derived, 0.2, 0.3, param("red") * 0.4)
            )
        }
    )
    assert plan.parameter_names == ("radius", "red")
    geometry = plan.passes[0].uniforms(resolver({"radius": radius}))
    fill = plan.passes[1].uniforms(resolver({"red": red}))
    assert geometry["radius"]._versions == {radius: radius._version}
    assert fill["color"]._versions == {red: red._version}


def test_unsupported_host_rasterizers_do_not_fall_back_to_cpu_images():
    with pytest.raises(ValueError, match="no CPU fallback"):
        lower_graph({"field": sdf.polygon(((1, 1), (6, 1), (3, 5)))})


def test_wrong_multi_output_index_kind_and_input_types_are_rejected():
    field = sdf.circle((2, 3), 1)
    multi = field.masked_union(field)
    for index, kind in ((2, "sdf"), (1, "sdf")):
        node = SDFTexture(
            op="output", inputs=(multi,), params={"index": index, "kind": kind}
        )
        with pytest.raises(ValueError, match="output index|output kind"):
            lower_graph({"field": node})
    node = SDFTexture(op="union", inputs=(field, color.clear("#abc")))
    with pytest.raises(ValueError, match="requires sdf inputs"):
        lower_graph({"field": node})
