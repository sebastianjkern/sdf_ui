"""Recompute approximate signed distance from a sampled zero contour on GPU."""

import struct

import wgpu

from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def _jump_flood(ctx, seed_a, seed_b, jump_buffer, width, height):
    jump = 1 << (max(width, height).bit_length() - 1)
    current, destination = seed_a, seed_b
    while jump >= 1:
        ctx.write_buffer(jump_buffer, 0, struct.pack("<I", jump))
        ctx.run_shader(
            "redistance_jump", {0: current, 1: destination, 2: jump_buffer},
            color_format="rgba32f",
        )
        current, destination = destination, current
        jump >>= 1
    ctx.write_buffer(jump_buffer, 0, struct.pack("<I", 1))
    ctx.run_shader(
        "redistance_jump", {0: current, 1: destination, 2: jump_buffer},
        color_format="rgba32f",
    )
    return destination

def render_redistance(renderer, inputs, params):
    ctx = renderer.ctx
    source = inputs[0].tex
    width, height = ctx.size
    seed_a = seed_b = output = jump_buffer = None

    try:
        seed_a = ctx.rgba32f()
        seed_b = ctx.rgba32f()
        output = ctx.r32f()
        jump_buffer = ctx.create_buffer(
            struct.pack("<I", 1) + bytes(12),
            usage=wgpu.BufferUsage.UNIFORM,
            label="redistance jump length",
        )

        ctx.run_shader(
            "redistance_seed",
            {0: source, 1: seed_a},
            color_format="rgba32f",
        )

        current = _jump_flood(ctx, seed_a, seed_b, jump_buffer, width, height)

        ctx.run_shader(
            "redistance_finalize",
            {0: source, 1: current, 2: output},
        )
    except BaseException:
        if output is not None:
            output.close()
        raise
    finally:
        if seed_a is not None:
            seed_a.close()
        if seed_b is not None:
            seed_b.close()
        if jump_buffer is not None:
            if not ctx._defer_resource_destroy(jump_buffer):
                jump_buffer.destroy()

    return SDFTexture(tex=output, context=ctx)


def register_plugins(registry):
    registry.register(
        Plugin(
            "redistance",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            render_func=render_redistance,
            method_of=(TextureKind.SDF,),
        )
    )
