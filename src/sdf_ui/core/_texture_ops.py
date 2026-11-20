"""Operations shared by concrete texture wrappers."""


def warp_texture(operation, texture, field, strength, steps):
    from ..gpu.nodes.common import build

    return build(operation, texture, field, strength=strength, steps=steps)
