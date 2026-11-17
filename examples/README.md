# Examples

Run examples from the repository root after `uv sync`:

```bash
uv run python -m examples --help
uv run python -m examples render_api
uv run python -m examples text_layout
uv run python -m examples text_opentype
uv run python -m examples vortices
uv run python -m examples procedural_fields
uv run python -m examples tidal_bloom
uv run python -m examples blue_hour
uv run python -m examples learned_score
uv run python -m examples distance_corrections
uv run python -m examples.learning_ui --mode all --steps 250
uv run python -m examples.learned_dashboard --steps 80
uv run python -m examples.gpu_graph --steps 150
uv run python -m examples.center_div --steps 60
```

The first command lists the original rendering examples. The learning and GPU
examples have their own command-line options; append `--help` for details.
Generated images and checkpoints go under `out/`.

``tidal_bloom`` renders a looping 640×640 generative animation from animated
SDF petals, contour threads, orbiting constellations, and layered orbital
tracks. It writes ``out/generative_art/tidal_bloom.mp4`` and a still preview
to ``out/generative_art/tidal_bloom.png``. The scene is intentionally built
from a modest number of GPU passes to keep frame renders comfortably below
ten seconds on typical adapters.

``blue_hour`` is a landscape-format neon city animation: layered SDF skylines,
lit windows, diagonal rain, a gliding tram, and moving water reflections. It
writes ``out/generative_art/blue_hour_relay.mp4`` and a PNG preview alongside
it.

``learned_score`` uses the learning API to arrange a kinetic graphic score.
Its marks begin with small deterministic offsets; a layout trainer discovers
their shared spacing, baseline, page coverage, and visual balance. The final
learned layout breathes into a 72-frame animation and is saved with its learned
layout report under ``out/generative_art/``.

``vortices`` shows an SDF warped by a pair of counter-rotating fields, with
streamlines drawn over the distorted contours.

``distance_corrections`` compares contour spacing after smooth union, local
gradient correction, and redistancing.

``text_layout`` demonstrates line wrapping, horizontal alignment, and vertical
alignment in fixed lower-left-origin text boxes. It uses ``measure_text`` to
size boxes from the actual line layout; ``sdf.text`` also accepts ordered
``fallback_fonts`` and can raise on unsupported characters with
``missing_glyph="error"``. Text boxes can clip explicitly with ``clip=True``;
``measure_text(..., width=..., height=...)`` reports whether the layout overflows.
Use ``align="justify"`` to expand word spaces on non-final lines, or choose
``overflow="visible"``, ``"clip"``, or ``"error"`` for a text box. Set
``trim_trailing_spaces=False`` to retain line-ending space advances.

``text_opentype`` shows standard ligature and kerning behavior and how to
override OpenType feature tags with ``features={"liga": False}``. Text shaping
also accepts explicit ``direction``, ``script``, ``language``, and variable
font ``variations``; measurement and rendering share the same shaping options.

Text can add SDF-native ``underline``, ``overline``, and ``line-through`` rules.
For inline styling, build runs from ``span("Large", size=38)`` and
``span(" regular", size=22)`` and pass them to ``text([...])``. Inline runs
can wrap and align in text boxes and select font paths, sizes, OpenType features,
and decorations.
``measure_text`` reports line widths, ascent/descent, baselines, and glyph ink
bounds; ``measure_glyph`` reports a grapheme's advance and ink bounds. To use
the host's fonts, resolve one with ``from sdf_ui.text import system_font`` and
pass ``path=system_font("Noto Sans")``. The bundled Roboto remains the default
for consistent output between systems.

``center_div`` trains an offset at progress 0.5, then sweeps the learned div
across its animation curve. It exports ``center_div.gif`` for the learning
curve and ``center_div_motion.gif`` for the timeline sweep. PNG frames are kept
in ``out/center_div/frames`` and ``out/center_div/motion``.

``procedural_fields`` renders a finite repeated SDF array and a composed
vector-field scene. The field scene adds radial, vortex, and fractal turbulence
fields, warps an SDF on the GPU, and overlays direction and magnitude
visualizations. Both images are written to ``out/examples/``.
