# Examples

Run examples from the repository root after `uv sync`. The catalog command
(`python -m examples`) accepts the names in the first section below. Scripts
in the second section are standalone modules and accept their own options.

## Rendering examples

```bash
uv run python -m examples --help
uv run python -m examples render_api
uv run python -m examples text_layout
uv run python -m examples text_opentype
uv run python -m examples vortices
uv run python -m examples procedural_fields
uv run python -m examples field_flow
uv run python -m examples starry_current
uv run python -m examples preset_gallery
uv run python -m examples.preset_gallery --preset ember
uv run python -m examples tidal_bloom
uv run python -m examples blue_hour
uv run python -m examples learned_score
uv run python -m examples distance_corrections
```

The catalog also supports `all` (the default) and `--no-stats`. Run
`uv run python -m examples --help` for the complete list of catalog names.

### Catalog index

| Topic | Names | What to explore |
| --- | --- | --- |
| Shapes and operations | `arc`, `capsule`, `complex`, `convex_polygon`, `diamond`, `ellipse`, `half_plane`, `masked_union`, `ngon`, `parallelogram`, `polygon`, `ring`, `sector` | Shape primitives, composition, and boolean operations |
| Rendering and effects | `distance_corrections`, `isolines`, `light`, `neuromorphism`, `render_api`, `transparency`, `vortices`, `voronoi` | Render controls, fields, lighting, and color effects |
| Text | `font`, `text`, `text_layout`, `text_opentype` | Font selection, layout, shaping, and rendering |
| Animation and artwork | `anim`, `art`, `blue_hour`, `github`, `striped_circles`, `tidal_bloom`, `field_flow`, `starry_current`, `ic` | Animated scenes and finished compositions |
| Learning | `learned_score`, `pd` | Learned layout and partial derivative visualization |
| Vector fields | `procedural_fields` | Finite repetition and composable fields |
| Geometry transforms | `rotation` | Shape rotation |
| Presets | `preset_gallery` | Palette and field presets |

## Learning and standalone examples

These scripts are not entries in the catalog above. Run them as modules; use
`--help` for each script's options:

```bash
uv run python -m examples.learning_ui --mode all --steps 250
uv run python -m examples.learned_dashboard --steps 80
uv run python -m examples.gpu_graph --steps 150
uv run --extra math python -m examples.plot_gallery
uv run python -m examples.high_fidelity_art --scale 4
uv run python -m examples.spatial_parameters
uv run python -m examples.quadtree_mosaic
uv run python -m examples.series_plot
uv run python -m examples.mark_gallery
uv run python -m examples.chart_grammar
uv run python -m examples.plot_distributions
uv run python -m examples.center_div --steps 60
uv run python -m examples.benchmark_graph --help
uv run python -m examples.benchmark_dirty_animation --help
uv run python -m examples.geometric_forest_animation --help
uv run python -m examples.geometric_forest_wallpaper --help
uv run python -m examples.optimize_circle --help
uv run python -m examples.shader_optimization --help
uv run python -m examples.high_level_learning
uv run python -m examples.contextual_graph_learning
uv run python -m examples.dawn_example
uv run python -m examples.distance_corrections_example
uv run python -m examples.field_flow_animation
uv run python -m examples.generative_art_animation
uv run python -m examples.github_banner_example
uv run python -m examples.kinetic_city_animation
uv run python -m examples.learned_rule_art
uv run python -m examples.procedural_fields_example
uv run python -m examples.starry_current_animation
uv run python -m examples.text_layout_example
uv run python -m examples.text_opentype_example
uv run python -m examples.transparency_example
uv run python -m examples.vortex_field_example
```

The narrated optimization video also requires ffmpeg and espeak-ng. Generated
images, videos, and checkpoints are generally written under `out/`.

## Catalog-only examples

These examples are available through `python -m examples NAME`; they do not
define a standalone module entry point:

`arc`, `capsule`, `complex`, `convex_polygon`, `diamond`, `ellipse`, `font`,
`half_plane`, `ic`, `isolines`, `light`, `masked_union`, `neuromorphism`,
`ngon`, `parallelogram`, `pd`, `polygon`, `ring`, `rotation`, `sector`,
`striped_circles`, `text`, and `voronoi`.

The standalone aurora illustration can be rendered with
`uv run python -m examples.aurora_landscape`.

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

``field_flow`` animates two counter-rotating vortex centers, their strengths,
and the frequency and amplitude of a turbulence field from normalized render
progress. The combined field warps a finite repeated SDF pattern and the color
backdrop. It writes ``out/generative_art/field_flow_animation.gif`` and a PNG
preview.

``starry_current`` is an original looping night landscape inspired by the
swirling skies and saturated blues of Van Gogh's *The Starry Night*. A moving
vortex field bends repeated SDF brush marks into a broad rotating sky whirl
above a village and cypress silhouette. It writes
``out/generative_art/starry_current.gif`` and a PNG preview.

``lagrange_intuition_short`` renders a one-minute narrated visual explanation
of constrained optimization. It follows the contours of ``xy`` to their
tangency with the unit circle, then shows why the gradients align and how the
multiplier measures the value of relaxing the constraint. SDF UI renders the
plot, equations, annotations, and animation frames. The ``Plot`` widget draws
the axes and samples the hyperbolas directly from ``y = c/x``. Matplotlib
supplies the MathText parser used by ``sdf.math``. Render it with
``uv run --extra math python -m examples.lagrange_intuition_short``; ffmpeg and
espeak-ng are also required. The MP4, poster, and subtitle file are written
under ``out/learning/``.

``preset_gallery`` demonstrates named and custom palettes with four
parameterized vector-field scenes. Running it without options writes each
render and a labeled side-by-side comparison to
``out/examples/preset_gallery/``. Select one or more variants with repeated
``--preset`` options, such as ``--preset ember --preset moss``.

``aurora_landscape`` is a standalone illustration example combining a warped
SDF ribbon, radial glow, a crescent moon, layered hills, and small star fields.
It writes ``out/examples/aurora_landscape.png``.

``plot_gallery`` renders three math graphics under ``out/examples/plot_gallery/``:
sampled sine and cosine functions with an annotated maximum, rose and Lissajous
parametric curves, and objective contours meeting a circular constraint. It uses
``Plot.axes``, ``curve``, ``parametric``, and ``annotate``, with equation labels
from ``sdf.math``. Install the optional MathText dependency with ``uv sync
--extra math`` before running it. Use ``--only functions``, ``--only parametric``,
or ``--only optimization`` to render one panel.

``series_plot`` combines signed bars, a line with circular markers, formatted
axis ticks, and a data annotation tied to a render parameter. It writes
``out/examples/series_plot.png``.

``mark_gallery`` renders a scatter plot, a grouped bar chart, and a
diverging bar chart in ``out/examples/mark_gallery/``.

``chart_grammar`` demonstrates shared scales, facets, and layered intervals
as well as grouped counting bars. It writes two images to
``out/examples/chart_grammar/``.

``plot_distributions`` renders a faceted latency histogram, a forecast with
a ribbon and capped error bars, a schedule made from rectangle layers, and a
sensor envelope drawn with direct ribbon and rectangle methods. Outputs are
written to ``out/examples/plot_distributions/``.
