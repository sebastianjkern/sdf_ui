"""Compare all shared shader programs with their original GLSL forwards."""

import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from sdf_ui import Canvas
from sdf_ui.backends.opengl.backend import GPUShaderBackend
from sdf_ui.backends.opengl.nodes.registry import registry
from tests.test_gpu_effects import DEFAULTS
from tests.test_gpu_effects import NAMES as EFFECTS
from tests.test_gpu_effects import _case as effects_case
from tests.test_gpu_shading import SOURCES as SHADING
from tests.test_gpu_shading import _case as shading_case


@pytest.mark.parametrize("name", sorted((*EFFECTS, *SHADING)))
def test_shared_forward_matches_original_shader(name):
    size = (5, 3)
    inputs, params = (
        shading_case(name, size) if name in SHADING else effects_case(name, size)[:2]
    )
    descriptor = next(item for item in registry.shader_files() if item.name == name)
    contract = descriptor.differentiation
    assert descriptor.operation.expression() is not None
    assert not re.search(r"\.d\b", descriptor.operation.forward_source())
    source = (
        (Path(__file__).resolve().parents[1] / "src/sdf_ui" / descriptor.path)
        .read_text()
        .replace("rgba8", "rgba32f")
    )
    with (
        Canvas(size) as canvas,
        GPUShaderBackend(canvas) as runtime,
        ExitStack() as stack,
    ):
        images = {
            key: stack.enter_context(runtime.upload(value))
            for key, value in inputs.items()
        }
        actual = [
            stack.enter_context(image)
            for image in runtime.forward(name, images, params, size)
        ]
        expected = [
            stack.enter_context(runtime.allocate(size, 1 if arg.kind == "r32f" else 4))
            for arg in contract.outputs
        ]
        program = runtime.ctx.compute_shader(source)
        try:
            for index, (arg, image) in enumerate(zip(contract.outputs, expected)):
                image.texture.bind_to_image(index, read=False, write=True)
                if arg.name in program:
                    program[arg.name].value = index
            for index, arg in enumerate(contract.inputs, start=len(expected)):
                images[arg.name].texture.bind_to_image(index, read=True, write=False)
                if arg.name in program:
                    program[arg.name].value = index
            uniforms = runtime.uniforms(name, params)
            for arg in contract.parameters:
                if arg.name in program:
                    program[arg.name].write(uniforms[arg.name].tobytes())
            program.run(1, 1, 1)
            runtime.ctx.memory_barrier()
            for generated, original in zip(actual, expected):
                np.testing.assert_allclose(
                    generated.read(),
                    original.read(),
                    atol=4e-5,
                    rtol=5e-5,
                    err_msg=name,
                )
        finally:
            program.release()


@pytest.mark.parametrize(
    "name", ["blur_hor_9", "blur_ver_9", "blur_hor_13", "blur_ver_13"]
)
def test_display_shader_preserves_builtin_blur_uniform_initializers(name):
    with Canvas((5, 3)) as canvas:
        shader = canvas.get_shader(name)
        for uniform, expected in DEFAULTS[name].items():
            np.testing.assert_allclose(
                np.frombuffer(shader[uniform].read(), dtype=np.float32),
                expected,
                atol=1e-7,
            )
