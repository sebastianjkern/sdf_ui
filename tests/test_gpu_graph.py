"""Native compiled graph behavior across rendering, losses, gradients and updates."""

import json

import numpy as np
import pytest

from sdf_ui import Canvas, param, sdf
from sdf_ui.backends.opengl import GPUImage
from sdf_ui.backends.opengl.nodes.differentiation import ShaderDifferentiation
from sdf_ui.graph import GraphTrainer
from sdf_ui.learning import (
    Adam,
    DifferentiableRenderer,
    Objective,
    Parameter,
    ParameterStore,
)


@pytest.fixture
def canvas():
    with Canvas((8, 6)) as value:
        yield value


def setup(canvas):
    params = ParameterStore([Parameter("radius", 1.7)])
    field = sdf.circle((3.2, 2.3), param("radius"))
    return params, canvas.compile({"field": field}, params=params)


def target(radius=2.7):
    y, x = np.indices((6, 8))
    return np.hypot(x - 3.2, y - 2.3) - radius


def test_native_forward_and_gpu_gradients_match_reference_with_combined_loss(canvas):
    p = ParameterStore([Parameter("r", 1.7), Parameter("color", [0.6, 0.3, 0.4, 0.7])])
    field = sdf.circle((3.2, 2.3), param("r"))
    image = field.fill(param("color"), (0.1, 0.2, 0.3, 0.8)).to_rgb()
    goal = np.linspace(0.1, 0.8, 6 * 8 * 4).reshape(6, 8, 4).astype(np.float32)
    reference = DifferentiableRenderer((8, 6), p)
    cpu_field, cpu_image = reference.render(field), reference.render(image)
    loss = (
        ((cpu_field - target()) ** 2).mean() * 0.3
        + ((cpu_image - goal) ** 2).mean() * 0.7
        + (p["r"] - 2.3) ** 2 * 0.1
    )
    expected_loss = loss.item()
    loss.backward()
    expected = {key: value.grad.copy() for key, value in p.items()}
    for value in p.values():
        value.zero_grad()
    with canvas.compile({"field": field, "image": image}, params=p) as graph:
        objective = (
            Objective()
            .mse("field", target(), weight=0.3)
            .mse("image", goal, weight=0.7)
            .geometry(lambda: (p["r"] - 2.3) ** 2, weight=0.1)
        )
        with graph.run(objective) as run:
            assert run.loss == pytest.approx(expected_loss, rel=2e-5)
            gradients = run.backward()
            for key, value in expected.items():
                np.testing.assert_allclose(gradients[key], value, atol=2e-5, rtol=2e-4)
            assert run.stats["image_readbacks"] == 0
            np.testing.assert_allclose(run.read("field"), cpu_field.numpy(), atol=2e-6)
            np.testing.assert_allclose(run.read("image"), cpu_image.numpy(), atol=2e-5)


def test_training_cannot_use_cpu_pixel_fallback_or_readback(canvas, monkeypatch):
    p, graph = setup(canvas)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU pixel path called during native graph training")

    monkeypatch.setattr(DifferentiableRenderer, "render", forbidden)
    monkeypatch.setattr(ShaderDifferentiation, "evaluate", forbidden)
    monkeypatch.setattr(ShaderDifferentiation, "vjp", forbidden)
    monkeypatch.setattr(GPUImage, "read", forbidden)
    with (
        graph,
        GraphTrainer(graph, Objective().mse("field", target()), lr=0.1) as trainer,
    ):
        history = trainer.fit(15)
        assert history[-1]["loss"] < history[0]["loss"] * 0.1
        assert all(h["stats"]["image_readbacks"] == 0 for h in history)
        assert all(h["stats"]["backward_dispatches"] == 1 for h in history)
        assert p["radius"].item() > 2.5


@pytest.mark.parametrize("kind", ["mse", "mae", "huber"])
def test_gpu_loss_values_and_seeds_are_normalized_consistently(canvas, kind):
    p, graph = setup(canvas)
    error = target(1.7) - 0.13
    if kind == "mse":
        expected, derivative = error**2, 2 * error
    elif kind == "mae":
        expected, derivative = np.abs(error), np.sign(error)
    else:
        expected = np.where(
            np.abs(error) <= 0.7, 0.5 * error**2, 0.7 * (np.abs(error) - 0.35)
        )
        derivative = np.clip(error, -0.7, 0.7)
    objective = Objective().pixel("field", 0.13, kind=kind, weight=2.3, delta=0.7)
    with graph, graph.run(objective) as run:
        assert run.loss == pytest.approx(expected.mean() * 2.3, rel=1e-5)
        run.backward()
        assert p["radius"].grad == pytest.approx(
            -derivative.mean() * 2.3, rel=2e-5, abs=1e-6
        )
        np.testing.assert_allclose(
            run.gradient("field"), derivative * 2.3 / error.size, atol=2e-7
        )


def test_alias_roots_sum_seeds_and_shared_pass_runs_once(canvas):
    p = ParameterStore([Parameter("r", 1.7)])
    field = sdf.circle((3.2, 2.3), param("r"))
    combined = field.interpolate(field, t=0.3)
    with canvas.compile({"a": combined, "b": combined}, p) as graph:
        assert len(graph.plan.passes) == 2
        objective = Objective().mse("a", target(2.7)).mse("b", target(3.7), weight=0.2)
        with graph.run(objective) as run:
            run.backward()
            assert run.stats["forward_dispatches"] == 2
            assert run.stats["backward_dispatches"] == 2
            assert p["r"].grad == pytest.approx(-2.8, abs=1e-5)


def test_masked_union_multi_output_backwards_matches_reference(canvas):
    p = ParameterStore([Parameter("r", 1.7)])
    field, mask = sdf.circle((2.3, 2.7), param("r")).masked_union(
        sdf.circle((5.2, 2.1), 1.8)
    )
    reference = DifferentiableRenderer((8, 6), p)
    loss = (reference.render(field) ** 2).mean() + (
        (reference.render(mask) - 0.3) ** 2
    ).mean() * 0.2
    loss.backward()
    expected = p["r"].grad.copy()
    p["r"].zero_grad()
    with canvas.compile({"field": field, "mask": mask}, p) as graph:
        assert len(graph.plan.passes) == 3
        with graph.run(Objective().mse("field", 0).mse("mask", 0.3, weight=0.2)) as run:
            run.backward()
            assert run.stats["backward_dispatches"] == 2
            assert p["r"].grad == pytest.approx(expected, rel=2e-4, abs=1e-5)


def test_cached_values_are_bounded_and_parameter_versions_invalidate(canvas):
    p, graph = setup(canvas)
    with graph:
        with graph.run() as run:
            first = run.outputs["field"].generation
        with graph.run() as run:
            assert run.outputs["field"].generation == first
            assert run.stats["forward_dispatches"] == 0
            assert run.stats["cache_hits"] == 1
        for i in range(8):
            p["radius"].assign(1.8 + i * 0.01)
            with graph.run() as run:
                assert run.outputs["field"].generation != first
                assert run.stats["forward_dispatches"] == 1
            assert graph.cache_info()["images"] == 1
        graph.clear_cache()
        assert graph.cache_info()["images"] == 0


def test_stale_parameters_and_mutated_objective_reject_backward(canvas):
    p, graph = setup(canvas)
    objective = Objective().mse("field", target(), name="fit")
    with graph:
        with graph.run(objective) as run:
            p["radius"].assign(1.8)
            with pytest.raises(RuntimeError, match="Stale"):
                run.backward()
        with graph.run(objective) as run:
            objective.set_weight("fit", 2)
            with pytest.raises(RuntimeError, match="Objective changed"):
                run.backward()


def test_execution_lifetime_and_double_backward_are_explicit(canvas):
    p, graph = setup(canvas)
    with graph:
        run = graph.run(Objective().mse("field", target()))
        with pytest.raises(RuntimeError, match="previous"):
            graph.run()
        run.backward()
        with pytest.raises(RuntimeError, match="consumed"):
            run.backward()
        run.close()
        with pytest.raises(RuntimeError, match="closed"):
            run.read("field")
        with graph.run() as display:
            with pytest.raises(ValueError, match="objective"):
                display.backward()


def test_target_changes_invalidate_only_values_not_compiled_programs(canvas):
    p, graph = setup(canvas)
    objective = Objective().mse("field", target(), name="fit")
    with graph:
        with graph.run(objective) as run:
            first = run.loss
        programs = graph.backend.stats["program_compilations"]
        objective.set_target("fit", target(3.7))
        with graph.run(objective) as run:
            assert run.loss > first
            assert run.stats["forward_dispatches"] == 0
            assert run.stats["image_uploads"] == 1
        assert graph.backend.stats["program_compilations"] == programs
        assert len(graph._targets) == 1


def test_checkpoint_restores_weights_moments_and_next_update(canvas, tmp_path):
    p, graph = setup(canvas)
    objective = Objective().mse("field", target())
    with graph, GraphTrainer(graph, objective, lr=0.1) as trainer:
        trainer.fit(4)
        checkpoint = tmp_path / "trainer.json"
        trainer.save(checkpoint)
        expected_metrics = trainer.step()
        expected = p["radius"].numpy()
        trainer.load(checkpoint)
        assert trainer.steps == 4
        metrics = trainer.step()
        np.testing.assert_array_equal(p["radius"].numpy(), expected)
        assert metrics["loss"] == expected_metrics["loss"]
        invalid = json.loads(checkpoint.read_text())
        invalid["parameters"]["radius"] = [1.0]
        with pytest.raises(ValueError, match="Invalid checkpoint"):
            trainer.load_state_dict(invalid)
        np.testing.assert_array_equal(p["radius"].numpy(), expected)


def test_gpu_snapshot_targets_and_checkpoint_signature_are_portable(canvas):
    with canvas.compile({"field": sdf.circle((3.2, 2.3), 2.7)}) as teacher:
        with teacher.snapshot("field") as target_image:
            p, graph = setup(canvas)
            with (
                graph,
                GraphTrainer(graph, Objective().mse("field", target_image)) as trainer,
            ):
                trainer.step()
                state = trainer.state_dict()
            with teacher.snapshot("field") as second_image:
                p2, graph2 = setup(canvas)
                with (
                    graph2,
                    GraphTrainer(
                        graph2, Objective().mse("field", second_image)
                    ) as resumed,
                ):
                    resumed.load_state_dict(state)
                    assert resumed.steps == 1
                    assert p2["radius"].item() == p["radius"].item()


def test_scalar_geometry_only_objective_updates_bound_parameters(canvas):
    p, graph = setup(canvas)
    objective = Objective().geometry(lambda: (p["radius"] - 3) ** 2)
    with graph, GraphTrainer(graph, objective, lr=0.1) as trainer:
        result = trainer.step()
        assert p["radius"].item() > 1.7
        assert result["stats"]["backward_dispatches"] == 0


def test_checkpoint_after_changing_gpu_target_uses_new_target(canvas):
    with canvas.compile(
        {"old": sdf.circle((3, 2), 2), "new": sdf.circle((3, 2), 3)}
    ) as teacher:
        with teacher.snapshot("old") as old, teacher.snapshot("new") as new:
            _, graph = setup(canvas)
            objective = Objective().mse("field", old, name="fit")
            with graph, GraphTrainer(graph, objective) as trainer:
                trainer.step()
                objective.set_target("fit", new)
                state = trainer.state_dict()
            _, graph2 = setup(canvas)
            with (
                graph2,
                GraphTrainer(
                    graph2, Objective().mse("field", new, name="fit")
                ) as resumed,
            ):
                resumed.load_state_dict(state)
                assert resumed.steps == 1


def test_parameter_identity_and_optimizer_identity_are_checked(canvas):
    p, graph = setup(canvas)
    with graph:
        with pytest.raises(ValueError, match="exactly"):
            GraphTrainer(
                graph,
                Objective().mse("field", target()),
                Adam([Parameter("other", 1.0)]),
            )
        p._parameters["radius"] = Parameter("radius", 2.0)
        with pytest.raises(ValueError, match="identities"):
            graph.run()


def test_closed_canvas_closes_registered_graph_and_default_optimizer():
    canvas = Canvas((8, 6))
    p, graph = setup(canvas)
    trainer = GraphTrainer(graph, Objective().mse("field", target()))
    canvas.close()
    assert graph.closed and trainer.closed
    with pytest.raises(RuntimeError, match="closed"):
        graph.run()


@pytest.mark.parametrize("shared_program", [False, True])
def test_explicit_custom_shader_contract_runs_through_the_full_native_stack(
    canvas, tmp_path, monkeypatch, shared_program
):
    from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
    from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, ShaderFile, TextureKind
    from sdf_ui.backends.opengl.nodes.registry import registry
    from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram

    name = "test_native_graph_gain"
    path = tmp_path / "gain.glsl"
    path.write_text(
        """#version 430
layout(local_size_x=4,local_size_y=2) in;
layout(r32f,binding=0) writeonly uniform image2D destTex;
layout(r32f,binding=1) readonly uniform image2D sourceField;
uniform float gain;
uniform int mode;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,imageSize(destTex))))return;
float value=imageLoad(sourceField,p).r;imageStore(destTex,p,vec4(mode==1?value*gain:value,0,0,0));}
""",
        encoding="utf-8",
    )
    contract = ShaderDifferentiation(
        name,
        str(path),
        (ShaderArgument("sourceField", (), "r32f"),),
        (ShaderArgument("gain", (), "float"), ShaderArgument("mode", (), "int")),
        (ShaderArgument("destTex", (), "r32f"),),
        zero_gradient_parameters=("mode",),
        reference=lambda ctx, inputs, params: (
            inputs["sourceField"] * params["gain"]
            if params["mode"].item() == 1
            else inputs["sourceField"]
        ),
    )
    derivative = """void evaluate(ivec2 pos,out D o[8]){
o[0]=U_mode.v==1.0?mul(ld(0,pos,0),U_gain):ld(0,pos,0);
for(int i=1;i<8;i++)o[i]=c(0.0);}
"""
    descriptor = (
        ShaderFile(name, str(path), contract, program=ShaderProgram(name, derivative))
        if shared_program
        else ShaderFile(name, str(path), contract, derivative)
    )
    plugin = Plugin(
        name,
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        (TextureKind.SDF,),
        params=("gain", "mode"),
        defaults={"mode": 1},
        shader=descriptor,
        input_uniforms=("sourceField",),
        make_uniforms=lambda p: p,
    )
    registry.ensure_loaded()
    monkeypatch.setitem(registry._plugins, name, plugin)
    gain = Parameter("gain", 2.0)
    node = registry.build(name, sdf.circle((3.2, 2.3), 1.7), gain)
    from sdf_ui.graph import lower_graph

    plan = lower_graph({"field": node})
    display = canvas.render_plan(plan)["field"]
    np.testing.assert_allclose(
        np.frombuffer(display.tex.read(), dtype=np.float32).reshape(6, 8),
        target(1.7) * 2,
        atol=2e-6,
    )
    objective = Objective().mse("field", target(1.7) * 3)
    with canvas.compile({"field": node}) as graph:
        assert graph.parameters == (gain,)
        with graph.run(objective) as execution:
            np.testing.assert_allclose(
                execution.read("field"), target(1.7) * 2, atol=2e-6
            )
            execution.backward()
            assert gain.grad == pytest.approx(-2 * np.mean(target(1.7) ** 2), rel=2e-5)
        with GraphTrainer(graph, objective, lr=0.1) as trainer:
            history = trainer.fit(30)
            assert history[-1]["loss"] < history[0]["loss"] * 0.03
    reference = descriptor.backward(
        (8, 6), {"sourceField": target(1.7)}, {"gain": 2.0, "mode": 1}, np.ones((6, 8))
    )
    assert reference.parameters["gain"] == pytest.approx(target(1.7).sum())
    assert reference.parameters["mode"] == 0


def test_invalid_forward_results_are_never_reused_as_valid_cache_hits(canvas):
    p = ParameterStore([Parameter("color", [1e38, 1e38, 1e38, 1.0])])
    from sdf_ui import color

    with canvas.compile({"image": color.clear(param("color"))}, p) as graph:
        for _ in range(2):
            with pytest.raises(FloatingPointError):
                graph.run()
            assert graph.cache_info()["images"] == 0
        p["color"].assign([0.2, 0.3, 0.4, 1.0])
        with graph.run() as execution:
            assert np.isfinite(execution.read("image")).all()


def test_existing_gpu_texture_is_a_native_input_with_pixel_adjoints_and_refresh(canvas):
    from sdf_ui.core.color import ColorTexture

    pixels = np.empty((6, 8, 4), dtype=np.uint8)
    pixels[:] = [51, 102, 153, 255]
    texture = canvas._mgl_ctx.texture((8, 6), 4, pixels.tobytes())
    source = ColorTexture(tex=texture, context=canvas, mode="RGB")
    params = ParameterStore([Parameter("alpha", 0.3)])
    image = source.transparency(param("alpha"))
    expected = pixels.astype(np.float32) / 255
    target_image = expected.copy()
    target_image[..., 3] = 0.8
    with canvas.compile({"source": source, "image": image}, params) as graph:
        assert len(graph.plan.sources) == 1
        assert len(graph.plan.passes) == 1
        with graph.run(Objective().mse("image", target_image)) as execution:
            np.testing.assert_allclose(execution.read("source"), expected)
            execution.backward()
            assert params["alpha"].grad == pytest.approx(-0.25, abs=1e-6)
            input_gradient = execution.input_gradient(0)
            np.testing.assert_allclose(input_gradient[..., :3], 0, atol=1e-7)
            np.testing.assert_allclose(input_gradient[..., 3], -0.075 / 48, atol=1e-7)
            with pytest.raises(RuntimeError, match="active execution"):
                graph.refresh_inputs()
        pixels[:] = [153, 102, 51, 255]
        texture.write(pixels.tobytes())
        with graph.run() as execution:
            np.testing.assert_allclose(execution.read("source"), expected)
        graph.refresh_inputs()
        with graph.run() as execution:
            np.testing.assert_allclose(
                execution.read("source"), pixels.astype(np.float32) / 255
            )
            assert execution.stats["forward_dispatches"] == 1
        assert graph.cache_info()["input_images"] == 1
        assert texture.glo != 0  # graph closes only its owned FP32 snapshot
    source.tex = None
    texture.release()


@pytest.mark.parametrize("trainable", [False, True])
def test_fixed_pass_propagates_input_and_intermediate_adjoints(canvas, trainable):
    from sdf_ui.core.color import ColorTexture

    pixels = np.full((6, 8, 4), 64, dtype=np.uint8)
    texture = canvas._mgl_ctx.texture((8, 6), 4, pixels.tobytes())
    source = ColorTexture(tex=texture, context=canvas, mode="RGB")
    fixed = source.post.invert()
    params = (
        ParameterStore([Parameter("alpha", 0.5)]) if trainable else ParameterStore()
    )
    output = fixed.transparency(param("alpha")) if trainable else fixed
    try:
        with canvas.compile({"fixed": fixed, "image": output}, params) as graph:
            with graph.run(Objective().mse("image", 0)) as execution:
                execution.backward()
                expected = np.full((6, 8, 4), -2 * (1 - 64 / 255) / (6 * 8 * 4))
                alpha = 0.5 if trainable else 1.0
                expected[..., 3] = 2 * (64 / 255) * alpha**2 / (6 * 8 * 4)
                np.testing.assert_allclose(
                    execution.input_gradient(0), expected, atol=1e-7
                )
                intermediate = expected.copy()
                intermediate[..., :3] *= -1
                np.testing.assert_allclose(
                    execution.gradient("fixed"), intermediate, atol=1e-7
                )
                if trainable:
                    assert params["alpha"].grad == pytest.approx(
                        (64 / 255) ** 2 * 0.5 / 2, abs=1e-7
                    )
    finally:
        source.tex = None
        texture.release()


def test_materialized_sdf_input_can_be_transformed_and_optimized(canvas):
    prepared = sdf.circle((3.2, 2.3), 1.7).render(canvas)
    p = ParameterStore([Parameter("scale", 1.1)])
    with canvas.compile(
        {"field": prepared.scale(param("scale"), center=(3.2, 2.3))}, p
    ) as graph:
        with graph.run(Objective().mse("field", 0)) as execution:
            execution.backward()
            assert np.isfinite(p["scale"].grad)
            assert execution.stats["image_readbacks"] == 0


def test_external_texture_context_and_shape_are_validated(canvas):
    from sdf_ui.core.color import ColorTexture

    with Canvas((8, 6)) as other:
        source = sdf.circle((3, 2), 1).render(other)
        with pytest.raises(ValueError, match="this open Canvas"):
            canvas.compile({"field": source})
    texture = canvas._mgl_ctx.texture((4, 3), 4)
    source = ColorTexture(tex=texture, context=canvas, mode="RGB")
    with pytest.raises(ValueError, match="canvas size"):
        canvas.compile({"image": source})
    source.tex = None
    texture.release()
