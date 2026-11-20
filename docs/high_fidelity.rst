High-fidelity stills
====================

Linear color and finishing
--------------------------

Start with ``color.linear_rgba()``, ``shape.ink_linear()``,
or ``shape.fill_linear(foreground, background, inner=..., outer=...)``.
Supply four numeric channels in linear sRGB. Color channels may exceed
one and alpha is straight and must lie between zero and one.
``fill_linear()`` interpolates between colors over the given signed
distance interval. Existing display layers can join the same composition
with ``to_linear()``. This decodes an already rendered color and retains
the precision of that source.

``over()`` and ``alpha_overlay()`` use linear light if a layer is
linear. Use ``expose(stops)`` to multiply RGB by a power of two.
``bloom(threshold=..., amount=..., passes=...)`` extracts light above a
linear luminance threshold, blurs it, and adds it back to the source.
The pass count is a fixed integer and the blur stencil is measured in
output pixels.

Use ``scene.film_grain(amount=0.075, size=1.35)`` before finishing to add a
subtle texture. Grain modulates brightness while preserving hue, alpha, and
black. The effect tapers in bright areas. Use ``amount=0`` for an unchanged
linear layer. Amount ranges from 0 to 1, and size controls the grain spacing
in output pixels, starting at 1. Increase size for coarser grain. The
standalone ``color.film_grain()`` factory provides a full-range noise texture
for use as a map.

``finish(exposure=..., tone=..., dither=...)`` applies exposure, a
luminance-based Reinhard operator, and the sRGB transfer function.
Use ``tone=0`` to bypass tone compression or ``dither=0`` to turn off
the deterministic ordered dither. Dither strength is measured in 8-bit
code values and preserves alpha. Saving a raw linear layer directly uses
sRGB encoding with clipping. Call ``finish()`` to select the look.
For raw floating point pixels, use ``Canvas.render()`` and read the
rendered texture.

Linear colors, fill distances, exposure, bloom threshold and amount, grain
amount and size, tone compression, and dither strength accept the engine's
lazy parameters. The learning graph differentiates the underlying
continuous math before export quantization. Keep positive controls in
range with bounded parameters or smooth constraints.

Tiled export
------------

``canvas.save_tiled(scene, path, tile_size=(512, 512))`` writes a color
still. The tile size is in physical pixels. Canvas coordinates, the dither
pattern, and procedural noise keep their full-frame origin. The renderer
adds a sampling margin for blur, bloom, and vector warps from built-in
bounded fields and discards this margin before assembling the still.
Transparent pixels are preserved in PNG output.

Compatible graphs use pointwise shaders, the supported bounded filters,
or supported composites such as polygon fields and gradients. Unsupported
operations, including repeating and resampling transforms with nonlocal
dependencies and materialized texture inputs, raise an error before the
output is written. Reduce tile size if the tile plus margin exceeds the
adapter's texture limit. The GPU working set depends on the graph and the
expanded tile size. The output image itself is assembled in CPU memory.

Examples
--------

Render two versions of Starry Current and a tile-compatible still::

    uv run python -m examples.high_fidelity_art

Render a 3840 by 2560 still without regenerating the comparison::

    uv run python -m examples.high_fidelity_art --scale 4 --skip-reference

Outputs are written to ``out/generative_art/high_fidelity/``.
