<img src="docs/assets/banner.png">

# SDF UI

SDF UI is a GPU-backed Python framework for high-fidelity generative art.
Build stills and animations from signed distance fields, procedural color,
vector fields, and composable effects. Use the same scene at preview or export
resolution, and drive geometry and shading with parameters that can be animated
or optimized.

Start with the [art and animation examples](examples/README.md), including
[Starry Current](#starry-current), [animated vector fields](#animated-fields),
and the [preset gallery](examples/preset_gallery.py). See
[What can I do with this engine?](#what-can-i-do-with-this-engine) for more renders.

The project grew out of a [metaballs experiment](https://github.com/sebastianjkern/metaballs).
Many of the distance functions are based on [Inigo Quilez’s SDF reference](https://iquilezles.org/articles/distfunctions2d/).

### Installation:

The project requires Python 3.11 or newer and can be installed with pip:

```bash
pip install git+https://github.com/sebastianjkern/sdf_ui
```

The GPU renderer targets desktop Windows 10+, macOS 10.13+, and recent Linux
distributions with Vulkan support. Linux users may need Mesa or vendor Vulkan
drivers; WSL is not currently supported by `wgpu-py`.

For interactive work, prefer `Canvas.render(...)`, the built-in `.save(...)` method on render nodes, or a cached `Canvas.session(...)`.

Text and glyph rendering use the bundled Roboto font by default. The glyph SDF
cache renders at 2.5x the effective text size by default to leave a small quality
margin; increase ``oversample`` for unusually large resizes, at additional cache
cost. Text accepts TrueType or OpenType fonts, ordered fallback fonts, line
wrapping and clipping, and OpenType shaping controls:

```python
from sdf_ui import sdf

sdf.text("office AVATAR", features={"liga": False, "kern": True})
```

Filled SVG geometry can also be rendered directly as an SDF. This accepts
MathJax-style SVG with a `viewBox`, path definitions, local `<use>` references,
affine transforms, nonzero or even-odd fill rules, and one-line `<text>` nodes:
It evaluates SVG geometry at 2× subpixel resolution by default; use
`oversample=1` for faster rendering or increase it up to `4` for finer details.

```python
from pathlib import Path
from sdf_ui import sdf

math_svg = Path("formula.svg").read_text(encoding="utf-8")
formula = sdf.svg(math_svg, center=(320, 180), height=72)
scene = formula.fill("#f8fafc", "#00000000")
```

SDFs can be distorted by a superposition of vortices. Each vortex is an
`(x, y, strength)` tuple; positive and negative strengths rotate in opposite
directions. `color.vortex_streamlines(...)` draws the same field as an overlay:

```python
from sdf_ui import color, sdf

vortices = ((300, 360, 1.15), (660, 360, -1.15))
warped = sdf.circle((480, 360), 220).vortex_warp(
    vortices, radius=175, strength=0.9
)
flow = color.vortex_streamlines(vortices, radius=175)
scene = warped.fill("#d2ddd4").alpha_overlay(flow)
```

Run `uv run python -m examples vortices` to render the distorted contour and
streamline example. The existing ``vortex_warp`` helper remains available for
its streamlines-oriented API. The general vector-field warp runs on the GPU.

Vector fields are floating-point textures with XY displacement in their red
and green channels. Build them from constant, radial, vortex, or turbulence
fields, then add or scale them and use the result to warp SDF or color textures.
Finite repetition samples a bounded grid of copies from a source SDF. Set
``center`` to the source shape center and ``spacing`` to the distance between
copies:

```python
from sdf_ui import field, sdf

flow = field.vortex((320, 180), radius=180, strength=0.8).add(
    field.turbulence(frequency=0.015, amplitude=12, octaves=3)
)
warped = sdf.circle((320, 180), 96).warp(flow, strength=1.0, steps=2)
repeated = sdf.circle((320, 180), 18).repeat_grid(
    columns=9, rows=5, spacing=(58, 58), center=(320, 180)
)
flow_preview = flow.visualize().alpha_overlay(flow.contours(spacing=8))
```

Run ``uv run python -m examples procedural_fields`` for rendered examples of
finite repetition, field composition, GPU warping, and field inspection. Field
parameters also accept progress expressions, so animated centers, strengths,
and turbulence settings work with ``Canvas.save_animation``. Run
``uv run python -m examples field_flow`` for a looping example.

Use ``color.palette(name)`` to get a built-in palette (``starry_night``,
``ember``, ``sea_glass``, or ``moss``), or pass a custom list of hex colors.
``Palette.sample(t)`` interpolates a normalized position and returns an RGBA
color tuple that can be used by fills and gradients:

```python
from sdf_ui import color

night = color.palette("starry_night")
sky_color = night.sample(0.58)
custom = color.palette(("#142b58", "#397cb5", "#f1d17a"))
swatches = custom.steps(5)
```

Use ``ease_in_cubic``, ``ease_out_cubic``, or ``ease_in_out_sine`` to shape a
normalized progress expression. ``Canvas.save_frame_sequence`` writes numbered
PNG frames for tools that need individual images:

```python
from sdf_ui import Canvas, ease_in_out_sine, param, sdf
from sdf_ui.learning import Parameter

phase = param("phase")
x = 40 + 560 * ease_in_out_sine(phase)
scene = sdf.circle((x, 180), 24).fill("#62bb47")
phase_value = Parameter("phase", 0.0, animate=lambda progress: progress)

with Canvas((640, 360)) as canvas:
    paths = canvas.save_frame_sequence(
        scene, "out/frames", frames=48, params={"phase": phase_value}
    )
```

Frames are named ``frame_000000.png`` onward, and the method returns their
paths. ``prefix``, ``start_index``, and ``dirty_regions`` can be set when a
different sequence layout or full-frame rendering is needed.

Smooth booleans and spatial warps can change an SDF's gradient magnitude.
`normalize_distance()` applies the inexpensive local correction
`field / |gradient(field)|` near the zero contour and fades it out across a
finite band (32 canvas units by default). It improves nearby distance values
but is not a global reconstruction. `redistance()` instead rebuilds
approximate signed distances to the sampled zero contour, preserving the
contour while restoring more uniform offsets at canvas resolution. Both run on
the GPU; redistancing uses multiple jump-flooding passes and does not preserve
gradients for learning:

```python
from sdf_ui import sdf

soft_shape = sdf.circle((280, 180), 90).smooth_union(
    sdf.circle((390, 180), 90), k=24
)
locally_corrected = soft_shape.normalize_distance()
reconstructed = soft_shape.redistance()
```

SDF contours can be outlined with `.stroke(width, fg_color, bg_color=...)`;
the width is measured in canvas units. Join paths by chaining `smooth_union`
calls, choosing a `k` for each join. Color layers can use
`.mask(top, mask, mode=...)` with `mix`, `multiply`, `screen`, `darken`,
`lighten`, or `over`;
the mask's red channel controls how much of the receiver (`self`) is used.
Masked unions return both the union SDF and a smoothly antialiased ownership
mask, with `edge_width` controlling the transition:

```python
from sdf_ui import color, sdf

joined = sdf.circle((180, 180), 70).smooth_union(
    sdf.circle((270, 180), 70), k=18
).smooth_union(sdf.circle((360, 180), 70), k=5)
outlined = joined.stroke(8, "#315b72")
mask_sdf, ownership = sdf.circle((240, 180), 130).masked_union(
    joined, edge_width=1.0
)
layered = outlined.mask(color.clear("#f4b942"), ownership, mode="over")
```

Voronoi cells are available as signed fields. Select a site to get its
unbounded cell; compose cells or use `debug_distance()` to inspect their
distance values and boundaries:

```python
from sdf_ui import sdf

sites = [(160, 140), (420, 180), (280, 420)]
cell = sdf.voronoi(sites, cell=1)
debug = cell.debug_distance(range=80)
```

`debug_seams()` emphasizes regularly spaced contours, while
`debug_boolean(other, operation="union")` shows signed values and the result
boundary for union, intersection, or subtraction. The Voronoi operation returns
one selected cell at a time; cells are unbounded.
Its field is the maximum of the signed bisector distances, which is exact
inside the cell and a useful signed distance approximation outside it. The
distance heatmap reads the field back to the CPU for color mapping, so it is
intended for inspection rather than a performance-sensitive render path.

For TeX-like equations, `sdf.math(...)` uses Matplotlib's built-in MathText
parser and emits vector geometry for the same SVG renderer. It needs no Node.js,
npm, or system TeX installation. MathText supports a useful subset of TeX math;
install the optional dependency with `pip install 'sdf_ui[math]'`. It defaults
to the STIX font set, 0.75 canvas unit SDF inflation, and 2× subpixel
oversampling; set `inflate=0` or `oversample=1` to change those defaults:

```python
from sdf_ui import sdf

equation = sdf.math(
    r"\nabla \cdot \mathbf{u} = 0, \qquad"
    r"\frac{\partial \mathbf{u}}{\partial t}"
    r"+ (\mathbf{u} \cdot \nabla) \mathbf{u}"
    r"= -\nabla p + \nu \nabla^2 \mathbf{u}",
    center=(320, 180),
    height=120,
)
scene = equation.fill("#f8fafc", "#00000000")
```

Pass `height=None` to size equations by `font_size` in canvas units. This keeps
ordinary symbols consistent across equations with superscripts or fractions.

The SVG renderer handles filled paths and simple one-line text; text uses the
matching system font when available, then the bundled Roboto fallback. Nested
`<tspan>` layout, strokes, gradients, filters, clipping paths, and external
references are not supported.

For denser output without changing scene coordinates, set a device scale on the
canvas. Geometry, text, and pixel-sized effects are scaled together; percentage
helpers continue to refer to the logical canvas dimensions:

```python
with Canvas((640, 360), device_scale=2) as canvas:
    scene.save("out@2x.png", ctx=canvas)
```

Use `direction`, `script`, and `language` to override inferred shaping runs, and
`variations={"wght": 650}` to set variable font axes. See
[`text_opentype`](examples/README.md) for a rendered example.

For fixed boxes, `align="justify"` expands word spacing on non-final lines.
`overflow="visible"`, `"clip"`, and `"error"` select box overflow behavior;
`trim_trailing_spaces=False` preserves the final space advance.

### Plotting

`sdf_ui.plot` provides basic plotting layers for annotations, explanatory
graphics, and data overlays in a scene.

| Workflow | Available features |
| --- | --- |
| `Plot(box, x_range, y_range)` | Axes and annotations, sampled function and parametric curves, lines, points, bars, ribbons, and rectangles |
| `Plot.from_data(rows, aes(...))` | Layered marks, grouped bars, histograms, ribbons and error bars, categorical legends, and facets with shared scales |

```python
from math import sin
from sdf_ui import Canvas, Plot, color

plot = Plot((72, 60, 504, 252), (-4, 4), (-1.5, 1.5))
scene = color.clear("#10151f").over(
    plot.axes(x_ticks=None, y_ticks=None),
    plot.curve(sin).stroke(3, "#62dfcf"),
)
with Canvas((640, 360)) as canvas:
    canvas.save(scene, "wave.png")
```

<table>
    <tr>
        <th>Parametric curves</th>
        <th>Forecast ribbon and error bars</th>
    </tr>
    <tr>
        <td><img src="docs/assets/examples/plot_parametric_curves.png" width="420" alt="A five-petal rose and a Lissajous curve on separate axes"></td>
        <td><img src="docs/assets/examples/plot_solar_forecast.png" width="420" alt="A solar generation forecast with a blue uncertainty band and gold sensor readings with capped error bars"></td>
    </tr>
</table>

Regenerate these with `uv run --extra math python -m examples.plot_gallery`
and `uv run python -m examples.plot_distributions`. See the
[plotting guide](docs/plotting.rst) for layered examples and live parameters.

### Widgets and annotations

`Panel` takes a full canvas box and exposes a padded `content_box`.
Use `shape()` for an SDF or `layer()` to paint the surface and optionally
composite a child. `progress_bar()` paints a track and a fill for a
normalized value. Both accept lazy expressions for their geometry.

```python
from sdf_ui import param
from sdf_ui.widgets import Panel, progress_bar

panel = Panel((32, 40, 240, 104), padding=(14, 12), radius=12)
x, y, width, _ = panel.content_box
bar = progress_bar((x, y, width, 8), param("fraction", 0.6))
scene = panel.layer("#171e2b", bar, clip=True)
```

`sdf.arrow(start, end)` returns an SDF that can be filled, stroked, or
combined with other shapes. `plot.annotate((1, sin(1)), "sample",
offset=(16, 12), leader_width=1.5)` adds a label and leader to the sine
plot above. Annotation offsets are measured in canvas units. The label
can be text or an existing SDF.

For development from a checkout, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
uv sync
uv run pytest
uv run python -m examples render_api
uv run python -m examples.learned_dashboard --steps 80
uv run python -m examples.center_div --steps 60
uv build
```

The lockfile keeps development and CI dependencies consistent. See
[the example index](examples/README.md) for more examples.

### Usage:

The public API is built around render nodes. `sdf` creates signed-distance-field textures,
`color` creates color textures, and `Canvas` materializes the same nodes on the GPU:

```python
from sdf_ui import Canvas, color, sdf

scene = (
    color.clear("#ffffff")
    .over(
        sdf.circle(("40%x", "50%y"), "18%min")
        .smooth_union(sdf.rect(("58%x", "50%y"), ("22%x", "18%y"), ("3%min", "3%min", "3%min", "3%min")), k="2%min")
        .fill("#62bb47", (0.0, 0.0, 0.0, 0.0))
    )
    .post.blur(1)
    .to_rgb()
)

cache = {}
with Canvas((640, 360)) as canvas:
    canvas.save(scene, "out.png", cache=cache)
```

`canvas.save(scene, path)` reuses the canvas and returns the output path. The
node shortcut accepts either `canvas=` or the backward-compatible `ctx=`:
`scene.save(path, canvas=canvas)`. A path is required so an accidental save
cannot silently overwrite a generic `image.png` file.

`render()` returns a materialized texture that owns a GPU image. Close it when
you are done, or use it as a context manager. Call `.read()` for a NumPy copy of
the rendered pixels:

```python
with Canvas((640, 360)) as canvas:
    with scene.render(canvas, cache=cache) as image:
        pixels = image.read()
        image.save("out.png")
```

`Canvas.render_many({...})` evaluates several named scenes through one renderer
and cache, which is useful when outputs share cached subgraphs. Each returned
image owns a GPU resource and should be closed when finished:

```python
with Canvas((640, 360)) as canvas:
    images = canvas.render_many({"base": base_scene, "framed": framed_scene})
    try:
        images["base"].save("base.png")
        images["framed"].save("framed.png")
    finally:
        for image in images.values():
            image.close()
```

Calling `scene.save(..., size=(640, 360))` creates and closes a temporary canvas
for one-off output. Pass `canvas=canvas` (or the backward-compatible
`ctx=canvas`) when you want to reuse an existing canvas.

For repeated evaluations, `Canvas.session()` reuses one renderer and cache and
exposes render counters:

```python
with Canvas((640, 360)) as canvas:
    with canvas.session(cache={}) as renderer:
        image = renderer.render(scene)
        assert renderer.render(scene) is image
        with image:
            image.save("out.png")
        print(renderer.cache_info().hits)
        print(renderer.stats.shader_dispatches)
```

For a known region of interest, `evaluation_rect` limits shader evaluation and
returns a cropped image. Rectangles use `(x, y, width, height)` integer pixel
coordinates from the top-left of the canvas:

```python
with Canvas((640, 360)) as canvas:
    crop_scene = sdf.circle((300, 180), 90).fill("#62bb47").to_rgb()
    crop = crop_scene.render(canvas, evaluation_rect=(180, 80, 240, 200))
    print(crop.size, crop.origin)  # (240, 200), (180, 80)
```

The rectangle applies to every shader in that render. Pointwise operations are
supported; filters and shaders that sample neighboring pixels need a halo and
currently reject cropped evaluation. This explicit region is the first step
toward automatic bounds propagation.

For repeated updates, a renderer session can retain each shader layer's full
output and patch only a dirty rectangle into layers whose inputs changed:

```python
with Canvas((640, 360)) as canvas:
    with canvas.session(dirty_regions=True) as renderer:
        frame = renderer.render(scene)  # full render seeds per-layer outputs
        frame = renderer.render(scene, progress=0.1, dirty_rect=(180, 80, 240, 200))
```

The rectangle must cover every changed pixel, including both old and new
positions of moving shapes. Pointwise shader layers can be patched; filters,
host-rendered nodes, and other unsupported paths fall back to full-layer
rendering. `renderer.stats` reports layer hits, updates, patched pixels, and
fallbacks. Set `dirty_regions=True` on a session to derive a conservative region
from changed primitive parameters automatically. Common finite SDF primitives
and translations, rotations, and scales are bounded; unknown geometry and
neighborhood effects use full rendering. Animation exports enable this mode by
default; pass `dirty_regions=False` to `Canvas.save_animation(...)` for a full
render baseline. The exporter uses dirty rendering for scenes with at most 96
graph nodes; larger scenes stay on the compiled renderer because current
per-layer bookkeeping costs more than the crop saves. During an animation, it
also returns to compiled rendering when the estimated crop exceeds 35% of the
canvas or more than ten layers have fallen back to full rendering.

### Architecture Note

Internally, `sdf_ui` is organized around immutable render nodes instead of immediate drawing calls.
Factories such as `sdf.circle(...)` and `color.clear(...)` do not render right away. They build a
small expression graph of `TextureNode` objects that can be composed, cached, and rendered later
against a `Canvas`.

At render time, the node graph is resolved by the renderer and dispatched through an internal
registry of built-in shader operations. Those operations are not a user extension API.

Tracked `.pyi` stubs provide editor completion and type checking. Dynamic core factory and
texture-operation signatures are generated from built-in operation metadata; GPU, graph, and
learning module stubs are maintained alongside their runtime APIs.

After changing built-in operation metadata, run `uv run python build_tools/generate_api_stubs.py`
and commit the generated core stubs. Run
`uv run python build_tools/generate_api_stubs.py --check` to check those stubs for drift
without modifying files. For development, install
[uv](https://docs.astral.sh/uv/getting-started/installation/), run `uv sync`,
then `uv run pre-commit install`; `uv run pre-commit run --all-files` checks tracked
files. If an older checkout configured `core.hooksPath` to `.githooks`, remove
that local setting with `git config --local --unset core.hooksPath` before
installing the hook.

Built-in primitives, shading passes, post effects, and composition operators are shader-backed
graph nodes implemented by the WebGPU backend. The public `Canvas` uses `wgpu-py` and targets
Windows, macOS, and Linux desktop systems.

### Learning layouts and graphics

`sdf_ui.learning` adds trainable parameters, reverse-mode autodiff, SGD/Adam,
semantic layout losses, and policies trained from reward callbacks. The CPU
reference backend requires no additional machine-learning dependency. A native
GPU render graph also supports combined image, field, and geometry objectives,
with GPU shader gradients and optimizer updates.

```python
from sdf_ui import Canvas, sdf
from sdf_ui import param
from sdf_ui.learning import Parameter
from math import pi, sin

offset = Parameter("offset", 0.0)
x = Parameter(
    "x",
    0.0,
    animate=lambda progress: 120 + 80 * sin(progress * 2 * pi) + offset,
)
scene = sdf.circle((param("x"), 32), 12)

def loss(graph):
    # x is evaluated at the selected frame; gradients flow to `offset`.
    return (graph.parameters["x"] - 120) ** 2

with Canvas((240, 64)) as canvas:
    canvas.fit({"shape": scene}, loss, params={"x": x, "offset": offset},
               progress=0.25, steps=50)
    scene.save("frame.png", ctx=canvas, params={"x": x}, progress=0.25)

print(offset.item())
```

An animated `Parameter` evaluates its callback at the normalized `progress`
for each render or training step. Its callback value is fixed for that frame
and is not itself trainable. Captured trainable parameters such as `offset`
remain differentiable. Inside a custom loss, `graph.parameters` exposes the
resolved values for the current frame. Set a frame per update with
`session.step(progress=...)` or `graph.run(..., progress=...)`; a fit call
uses one selected progress value for all of its steps.

The loss callback also receives differentiable layer outputs. It can combine
image or field losses with arbitrary scalar terms; targets are optional.
``graph[scene]`` can reference any layer in the compiled scene, including
intermediate layers. Accessing ``.output`` reads that layer back to a CPU Tensor
for Python loss evaluation. For built-in losses that keep image data on the
GPU, use ``canvas.training`` with an ``Objective``.

The CPU reference backend evaluates a documented subset of the existing render
operations as floating-point tensors. It has its own per-execution tape and
does not backpropagate through WebGPU or quantized display textures. Unsupported
operations raise an explicit error. Reward-based policy learning can use the
full GPU renderer because its reward callback does not need pixel gradients.

See [the learning guide](docs/learning.rst) for geometry optimization, continuous
and discrete policies, graph training, checkpoints, and runnable
examples.

The center div example uses an animated `Parameter` for its horizontal sweep
and trains a separate offset at progress `0.5`. It exports both the learning
curve at that frame and the final sweep:

![A rounded rectangle learning to center on the canvas](docs/assets/examples/center_div.gif)

![The learned div sweeping across the canvas](docs/assets/examples/center_div_motion.gif)

Run it with `uv run python -m examples.center_div --steps 60`; the script saves
both GIFs and individual PNG frames under `out/center_div/`. See
[the example source](examples/center_div.py).

For a larger procedural layout example, run
`uv run python -m examples.learned_dashboard --steps 80`. It learns how to distribute
space between three content cards across viewport and content scenarios, and
produces before/after/reference images plus held-out metrics. See the
[learning guide](docs/learning.rst#procedural-dashboard-example) for how to
load the checkpoint and evaluate new measurements.

All 43 registered shader passes also expose executable differentiation contracts
through `shader.differentiation` and `shader.backward(...)`. These analytic CPU
references report pixel-input gradients and reduced uniform gradients, including
multi-output passes and explicit zero-gradient cases. See the
[shader differentiation guide](docs/shader_differentiation.rst) for the API,
coverage and numerical conventions. The [native render graph guide](docs/gpu_render_graph.rst)
describes GPU execution, custom shaders, checkpoints, and limitations. Run its
combined field, image, and geometry optimization with
`uv run python -m examples.gpu_graph --steps 150`.

### How it works:

<img src="./docs/assets/sdf_ui_diagram.png" width="500">

### What can i do with this engine?

Take a look below, or at the banner image, that was also created using this engine ;)
___

#### Primitives

<table>
    <tr>
        <th>Capsule</th>
        <th>Ring</th>
        <th>Diamond</th>
        <th>Polygon</th>
    </tr>
    <tr>
        <td><img src="./docs/assets/examples/capsule_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/ring_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/diamond_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/polygon_square.png" width="180"></td>
    </tr>
</table>

<table>
    <tr>
        <th>Arc</th>
        <th>Sector</th>
        <th>Ellipse</th>
    </tr>
    <tr>
        <td><img src="./docs/assets/examples/arc_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/sector_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/ellipse_square.png" width="180"></td>
    </tr>
</table>

#### Composition

<table>
    <tr>
        <th>Transparency</th>
        <th>Voronoi Field</th>
        <th>Cached Text</th>
        <th>Complex Shapes</th>
    </tr>
    <tr>
        <td><img src="./docs/assets/examples/transparency_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/voronoi_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/text_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/complex_square.png" width="180"></td>
    </tr>
</table>

<table>
    <tr>
        <th>Lighting</th>
        <th>Isolines</th>
        <th>Complex Shapes</th>
    </tr>
    <tr>
        <td><img src="./docs/assets/examples/light_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/isolines_square.png" width="180"></td>
        <td><img src="./docs/assets/examples/complex_square.png" width="180"></td>
    </tr>
</table>

#### Procedural Fields

<table>
    <tr>
        <th>Finite SDF Repetition</th>
        <th>Composed Vector Fields</th>
    </tr>
    <tr>
        <td><img src="./docs/assets/examples/finite_field_grid.png" width="420"></td>
        <td><img src="./docs/assets/examples/composed_vector_fields.png" width="420"></td>
    </tr>
</table>

#### Animated Fields

<img src="./docs/assets/examples/field_flow_animation.gif" width="480" alt="Two moving vortices animate a turbulence field that bends a repeated SDF pattern">

#### Starry Current

<img src="./docs/assets/examples/starry_current.gif" width="640" alt="An animated swirling blue night sky over a village, crescent moon, and cypress tree">

#### Linear light and large stills

Use linear RGBA to keep bright light and subtle gradients in 32-bit floating
point through compositing. RGB values can exceed 1, and alpha stays in
`[0, 1]`. Finish the scene with exposure, tone mapping, and optional
dithering before saving a display image.

```python
from sdf_ui import Canvas, color, sdf

light = sdf.circle(("50%x", "50%y"), "12%min").ink_linear((3.5, 1.3, 0.24, 1))
scene = color.linear_rgba((0.015, 0.025, 0.055, 1)).over(light)
finished = scene.bloom(threshold=0.9, amount=0.55).finish(exposure=0.25)
with Canvas((960, 640), device_scale=4) as canvas:
    canvas.save_tiled(finished, "glow.png", tile_size=(512, 512))
```

`save_tiled()` allocates GPU textures for one tile and its sampling margin.
The finished image is assembled on the CPU. See the
[high-fidelity rendering guide](docs/high_fidelity.rst) for compatible
operations and parameter behavior. Render the example with
`uv run python -m examples.high_fidelity_art --scale 4`.

#### Larger Scenes

<table>
    <tr>
        <th>Impossible City</th>
    </tr>
    <tr>
        <td><img src="./docs/assets/examples/impossible_city_square.png" width="240"></td>
    </tr>
</table>

### Development Checklist

The renderer already supports progress-driven parameters and GIF/MP4 export.
The completed feature history is in [Tasks.md](Tasks.md). Open work is sorted
below by rough implementation effort and expected project value. “Easy” means
a focused API or shader change; “hard” means broader engine or pipeline work.
These are planning estimates, not promises.

The easy, important quadrant has been implemented: named/custom palette
sampling, parameterized preset comparisons, numbered PNG frame export,
reusable progress easing, and darken/lighten masked blends. Existing `mix`,
`multiply`, and `screen` mask modes remain available.

| Effort | Important | Less important |
| --- | --- | --- |
| **Easy** | Reusable visual button/control widget with content-aware text fitting; annotated learning-layout preview export. | First-class XOR composition; icon, badge, and glyph convenience helpers; naming cleanup across shaders and public methods; more showcase art examples. |
| **Hard** | Shader/loss adapter boundaries and learning-graph organization; bounds-aware shaders and subframe rendering; variable-radius capsules and mask-to-SDF conversion; glyph SDF atlas; native timelines and keyframes; soft-light blending; explicit sRGB management; reusable effects; improved image-to-SDF conversion and image controls; common image formats and color profiles; safer texture cleanup. | Lazy-load example catalog entries so an unrelated example's import or optional dependency failure does not prevent other examples from running; custom SDF font pipeline; orthographic 2.5D rendering, depth composition, camera and world transforms; script hot reload; additional noise families; boid/swarm showcase. |

Use `.uncached()` for one-shot intermediate layers when preserving a cross-render
cache has no value. Their output images can be recycled once every graph branch
has consumed them; the final output remains live. `RenderStats.texture_reuses`
reports allocations reused from the context's bounded GPU image pool.
