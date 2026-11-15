import numpy as np

from sdf_ui import Canvas, color
from sdf_ui.gpu.nodes.layer.alpha_overlay.plugin import (
    render_alpha_overlay_batch,
    render_alpha_overlay_pair,
)


def test_chunked_overlay_outputs_match_sequential_rgba8_composites():
    base_node = color.clear((0.18, 0.62, 0.34, 0.73))
    top_nodes = [
        color.clear((0.91, 0.23, 0.14, 0.39)),
        color.clear((0.12, 0.77, 0.56, 0.62)).to_rgb(),
        color.clear((0.34, 0.27, 0.88, 0.51)),
    ]

    with Canvas((20, 16)) as canvas, canvas.session() as renderer:
        base = renderer.render(base_node)
        tops = tuple(renderer.render(node) for node in top_nodes)
        batched = render_alpha_overlay_batch(renderer, base, tops)

        sequential = []
        current = base
        for top in tops:
            current = render_alpha_overlay_pair(renderer, (current, top), {})
            sequential.append(current)

        for index, expected in enumerate(sequential):
            np.testing.assert_array_equal(batched[index].tex.read(), expected.tex.read())

        for image in (*batched.values(), *sequential):
            image.close()


def test_long_overlay_chain_batches_passes_and_keeps_intermediate_cache_entries():
    layers = [
        color.clear((0.1 + index * 0.1, 0.7 - index * 0.06, 0.3, 0.25 + index * 0.07))
        for index in range(7)
    ]
    chain = [layers[0]]
    for index, layer in enumerate(layers[1:], start=1):
        previous = chain[-1]
        if index == 4:
            previous = previous.cache("checkpoint")
        chain.append(previous.alpha_overlay(layer))

    with Canvas((24, 18)) as canvas, canvas.session() as renderer:
        output = renderer.render(chain[-1])
        assert canvas.last_render_stats.shader_dispatches_by_name["overlay"] == 2
        checkpoint = chain[4].inputs[0]
        assert renderer.render(checkpoint) is renderer.cache[renderer._node_key(checkpoint)]
        assert renderer.stats.cache_hits > 0
        output.close()
