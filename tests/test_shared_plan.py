import moderngl
import numpy as np
import pytest

from sdf_ui import Canvas, param, sdf
from sdf_ui.core.color import ColorTexture
from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.registry import registry
from sdf_ui.graph import GraphTrainer, lower_graph
from sdf_ui.learning import (
    DifferentiableRenderer,
    Objective,
    ParameterStore,
)


def test_display_layer_shaders_match_reference_without_readback(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("layer attempted CPU texture transfer")

    rng = np.random.default_rng(13)
    arrays = [rng.integers(0, 256, (6, 8, 4), dtype=np.uint8) for _ in range(3)]
    arrays[0][0, 0, 3] = arrays[1][0, 0, 3] = 0
    with Canvas((8, 6)) as canvas:
        textures = []
        for array in arrays:
            texture = canvas.rgba8()
            texture.write(array.tobytes())
            textures.append(ColorTexture(tex=texture, context=canvas, mode="RGB"))
        a, b, mask = [v.astype(np.float32) / 255 for v in arrays]
        alpha = b[..., 3:4] + (1 - b[..., 3:4]) * a[..., 3:4]
        numerator = (
            b[..., :3] * b[..., 3:4] + a[..., :3] * (1 - b[..., 3:4]) * a[..., 3:4]
        )
        rgb = np.divide(numerator, alpha, out=np.zeros_like(numerator), where=alpha > 0)
        expected = (
            a * b,
            a * mask[..., :1] + b * (1 - mask[..., :1]),
            np.concatenate((rgb, alpha), axis=2),
        )
        scenes = (
            textures[0].multiply(textures[1]),
            textures[0].mask(textures[1], textures[2]),
            textures[0].over(textures[1]),
        )
        for scene, goal in zip(scenes, expected):
            with monkeypatch.context() as guarded:
                guarded.setattr(moderngl.Texture, "read", fail)
                result = canvas.render(scene)
            pixels = np.frombuffer(result.tex.read(), np.uint8).reshape(6, 8, 4)
            np.testing.assert_allclose(pixels, np.rint(goal * 255), atol=1)


def test_shared_plan_executes_display_cpu_and_training():
    params = ParameterStore()
    params.define("r", 2.0)
    scene = sdf.circle((4, 3), param("r"))
    image = scene.fill("#ff0000", "#000000").to_rgb()
    plan = lower_graph({"field": scene, "image": image})
    reference = DifferentiableRenderer((8, 6), params).render_plan(plan)
    with Canvas((8, 6)) as canvas:
        display = canvas.render_plan(plan, params)
        np.testing.assert_allclose(
            np.frombuffer(display["field"].tex.read(), np.float32).reshape(6, 8),
            reference["field"].numpy(),
            atol=1e-5,
        )
        with canvas.compile(plan, params) as graph:
            with graph.run() as execution:
                np.testing.assert_allclose(
                    execution.read("field"), reference["field"].numpy(), atol=1e-5
                )
                np.testing.assert_allclose(
                    execution.read("image"), reference["image"].numpy(), atol=2e-4
                )
        legacy = canvas.render(image, params)
        a = np.frombuffer(display["image"].tex.read(), np.uint8).astype(int)
        b = np.frombuffer(legacy.tex.read(), np.uint8).astype(int)
        assert np.max(np.abs(a - b)) <= 1


def test_custom_plugin_lowering_is_used_by_all_executors(monkeypatch):
    registry.ensure_loaded()

    def lower(builder, plugin, inputs, raw, values):
        return builder.emit(
            "circle",
            uniforms=lambda resolve: {
                "offset": values(resolve)["center"],
                "radius": values(resolve)["radius"],
            },
        )[0]

    plugin = Plugin(
        "custom_lowering_test",
        PluginFamily.PRIMITIVE,
        TextureKind.SDF,
        params=("center", "radius"),
        render_func=lambda *args: None,
        lowering_func=lower,
    )
    monkeypatch.setitem(registry._plugins, plugin.name, plugin)
    scene = registry.build(plugin.name, (4, 3), 2)
    plan = lower_graph({"field": scene})
    cpu = DifferentiableRenderer((8, 6)).render_plan(plan)["field"].numpy()
    with Canvas((8, 6)) as canvas:
        image = canvas.render_plan(plan)["field"]
        np.testing.assert_allclose(
            np.frombuffer(image.tex.read(), np.float32).reshape(6, 8), cpu, atol=1e-5
        )
        with canvas.compile({"field": scene}) as graph, graph.run() as run:
            np.testing.assert_allclose(run.read("field"), cpu, atol=1e-5)


def test_geometry_checkpoint_requires_and_checks_identity():
    with Canvas((8, 6)) as canvas:
        params = ParameterStore()
        params.define("r", 2.0)
        scene = sdf.circle((4, 3), param("r"))
        with canvas.compile({"field": scene}, params) as graph:
            with GraphTrainer(
                graph, Objective().geometry(lambda: params["r"] ** 2)
            ) as trainer:
                with pytest.raises(ValueError, match="identity"):
                    trainer.state_dict()
            with GraphTrainer(
                graph,
                Objective().geometry(lambda: params["r"] ** 2, identity={"target": 1}),
            ) as trainer:
                state = trainer.state_dict()
            with GraphTrainer(
                graph,
                Objective().geometry(lambda: params["r"] ** 2, identity={"target": 2}),
            ) as trainer:
                with pytest.raises(ValueError, match="objective"):
                    trainer.load_state_dict(state)
            # The explicit identity is portable even if a callback is recreated
            # at another source location.
            with GraphTrainer(
                graph,
                Objective().geometry(lambda: params["r"] ** 2, identity={"target": 1}),
            ) as trainer:
                trainer.load_state_dict(state)
            with pytest.raises(ValueError, match="JSON-serializable"):
                Objective().geometry(lambda: 0, identity={"target": float("nan")})
