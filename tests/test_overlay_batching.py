import numpy as np

from examples.lagrange_intuition_short import _frame_scene
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


def test_repeated_long_overlay_chains_recycle_batch_intermediates():
    with Canvas((24, 18)) as canvas, canvas.session() as renderer:
        # Cover both sides of every three-overlay batch boundary, with and
        # without a reusable cache checkpoint inside the chain.
        for overlay_count, checkpoint_at in (
            (2, None),
            (3, None),
            (4, None),
            (6, None),
            (7, None),
            (8, None),
            (7, 3),
            (9, 4),
        ):
            layers = [
                color.clear(
                    (0.12 + index * 0.04, 0.68 - index * 0.03, 0.3, 0.45)
                ).uncached()
                for index in range(overlay_count + 1)
            ]
            chain = layers[0]
            for index, layer in enumerate(layers[1:], start=1):
                if index == checkpoint_at:
                    chain = chain.cache(f"checkpoint_{overlay_count}_{index}")
                chain = chain.alpha_overlay(layer).uncached()

            for _ in range(2):
                renderer.render(chain).close()
            allocations_after_warmup = renderer.stats.texture_allocations

            for _ in range(3):
                renderer.render(chain).close()

            assert renderer.stats.texture_allocations == allocations_after_warmup, (
                f"allocations grew for {overlay_count} overlays with "
                f"checkpoint at {checkpoint_at}"
            )

        assert renderer.stats.texture_reuses > 0


def test_repeated_lagrange_frames_do_not_grow_render_cache_or_image_allocations():
    static_scenes = tuple(
        color.clear("#10151f").cache(f"test_section_{index}")
        for index in range(7)
    )
    gradient_labels = color.clear((0.0, 0.0, 0.0, 0.0)).cache("test_gradients")
    starts = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    durations = (2.0,) * len(starts)

    with Canvas((32, 24)) as canvas, canvas.session() as renderer:
        def render_frame(time):
            frame = renderer.render(
                _frame_scene(time, starts, durations, static_scenes, gradient_labels)
            )
            frame.close()

        for start in starts:
            render_frame(start)
        for start in starts:
            render_frame(start + 0.1)  # Warm all sections and the image pool.
        cache_entries_after_warmup = renderer.cache_info().entries
        allocations_after_warmup = renderer.stats.texture_allocations

        for offset in (0.35, 0.75, 1.15, 1.55):
            for start in starts:
                render_frame(start + offset)

        assert renderer.cache_info().entries == cache_entries_after_warmup
        assert renderer.stats.texture_allocations == allocations_after_warmup
