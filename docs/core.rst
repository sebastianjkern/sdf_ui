Core Modules
============

The :mod:`sdf_ui.core` package contains render nodes, expression resolution,
and the public texture types used to describe scenes. WebGPU shader execution
helpers live under :mod:`sdf_ui.gpu`.

Package Overview
----------------

.. automodule:: sdf_ui.core
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Modules
-------

Context and rendering
~~~~~~~~~~~~~~~~~~~~~~

``Canvas.render(..., evaluation_rect=(x, y, width, height))`` evaluates a
pointwise shader graph in the requested canvas-pixel region and returns cropped
images. Filters and other neighborhood-sampling operations currently require a
full-canvas render.

.. automodule:: sdf_ui.gpu.context
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Texture nodes
~~~~~~~~~~~~~

.. automodule:: sdf_ui.core.texture
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Signed-distance textures
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sdf_ui.core.sdf
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Color textures
~~~~~~~~~~~~~~

.. automodule:: sdf_ui.core.color
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Expression system
~~~~~~~~~~~~~~~~~

.. automodule:: sdf_ui.core.expressions
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Canvas-relative values
~~~~~~~~~~~~~~~~~~~~~~

Percentage expressions resolve against the canvas when a render executes.
``percent`` and ``percent_x`` use the canvas width, ``percent_y`` uses its
height, and ``percent_of_min`` uses the smaller dimension. Numeric strings
offer the same forms in texture parameters: ``"40%"`` or ``"40%x"`` for width,
``"40%y"`` for height, and ``"40%min"`` for the smaller dimension. For
example, ``sdf.circle(("50%x", "50%y"), "12%min")`` remains centered and
proportional when rendered at different canvas sizes.

The function forms return lazy expressions and can be composed with other
expressions or trainable parameters::

    from sdf_ui import percent_x, percent_y, sdf

    scene = sdf.circle((percent_x(50), percent_y(50)), percent_x(12))

Dynamic API namespaces
~~~~~~~~~~~~~~~~~~~~~~

``sdf.rect`` and ``sdf.rounded_rect`` take half-width and half-height values.
Use ``sdf.rect_from_size`` or ``sdf.rounded_rect_from_size`` when the inputs
are full dimensions. These helpers keep lazy expressions live for training.
``sdf.union_all`` combines a nonempty iterable of SDF nodes. ``sdf.star`` and
``sdf.fir_tree`` build reusable geometry from parameterized triangle nodes;
their ray and tier counts are fixed topology controls. ``sdf.isometric_cube``
builds a learnable wireframe from segment nodes. ``color.concentric_rings``
composes a fixed number of outlines while keeping the geometry and tint live.
``ColorTexture.over`` composites layers in order and leaves temporary
intermediate composites uncached::

    from sdf_ui import color, param, sdf

    panel = sdf.rounded_rect_from_size(
        (param("x"), 80), (param("width"), 48), corner_radius=8
    )
    field = sdf.union_all((sdf.circle((40, 80), 12), panel))
    star = sdf.star((param("star_x"), 80), 18, rays=6)
    tree = sdf.fir_tree(100, 20, 42, 30, tiers=3)
    cube = sdf.isometric_cube((180, 20), 10, 10, stroke_width=2)
    rings = color.concentric_rings(
        (140, 80), 20, count=3, tint="#ffc978", line_width=2, gap=4
    )
    scene = color.clear("#10151f").over(
        field.fill("#78c7a4", "#00000000"),
        star.fill("#ffc978", "#00000000"),
        tree.fill("#78c7a4", "#00000000"),
        cube.fill("#ffc978", "#00000000"),
        rings,
        color.radial_gradient((40, 80), "#ffffff", "#00000000", 0, 24),
    )

Pass ``cache=False`` to ``ColorTexture.over`` when the final composite is
unique to one frame. Intermediate results remain disposable in either mode.

Use ``.uncached()`` for one-shot intermediate layers that do not need to be
kept across renders. Their output images can be recycled after every graph
branch has consumed them. ``RenderStats.texture_reuses`` counts allocations
reused from the context's bounded GPU image pool.

.. automodule:: sdf_ui.core.api
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

WebGPU shader execution
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sdf_ui.gpu.context
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

WebGPU texture utilities
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sdf_ui.gpu.image
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Paint scenes for PNG, SVG, and G-code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A paint scene reuses the same SDF fields, layer order, and hatch recipe for
pixels and pen paths. Hatch angles are in radians. Spacing, stroke width,
phase, and origin use logical canvas units. The plotter profile maps those
units to millimetres. For example::

    import math
    from sdf_ui import Canvas, paint, sdf

    silhouette = sdf.circle((64, 64), 48).subtract(sdf.circle((64, 64), 12))
    hatch = paint.HatchStyle((
        paint.HatchPass(math.pi / 4, 6, width=1.25),
        paint.HatchPass(-math.pi / 4, 9, width=1.25),
    ))
    drawing = paint.scene(
        paint.fill(silhouette, "#346b84", hatch=hatch),
        paint.outline(silhouette, "#172d42", width=1.5),
    )
    with Canvas((128, 128)) as canvas:
        drawing.save_png("drawing.png", canvas=canvas)
        drawing.save_svg("drawing.svg", canvas=canvas)

``drawing.save_gcode`` takes the same scene and a ``PlotterProfile`` from
``sdf_ui.export``. Set the paper dimensions, unit scale, travel and draw feed
rates, and the pen lift commands for the machine. A layer can name a pen
with ``pen=...``. Named pens require corresponding change commands in the
profile, and layers without a pen name continue with the current pen.
The physical nib determines the stroke width on paper.

SVG and G-code both use ``drawing.to_vector(...)`` to trace the SDF sampled at
the canvas resolution. Features smaller than the sampling pitch may be
missed. Plotting keeps the authored stroke order and accumulates ink where
layers overlap. A fill with no hatch recipe renders solid in PNG and uses
parallel lines spaced eight canvas units apart in a plot.

Hand strokes and hatch lighting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A hand stroke can drift gradually as the wrist turns, then start at a
slightly different position and angle when the pen is picked up. Pass
``hand=paint.HandStroke(bow=3, jitter=1, seed=41)`` to ``HatchStyle`` to
give each stroke one gentle arc and a small placement variation. Bow and
jitter use logical canvas units. The seed makes the pattern repeatable
across exports.

``HatchLight`` treats the shape as relief by turning inward SDF distance
into a rounded height profile. Its direction points towards the light in
right, up, and forward coordinates. The depth controls how far the
rounded profile extends into the shape. Each pass can specify a minimum
shadow amount between 0 and 1. For example, use one angle for midtones
and add a different angle in deeper shadow::

    shaded = paint.HatchStyle(
        (
            paint.HatchPass(0.6, 8, width=1.15, min_shadow=0.25),
            paint.HatchPass(-1.0, 8, width=1.1, min_shadow=0.6),
        ),
        hand=paint.HandStroke(bow=3, jitter=1, seed=41),
        light=paint.HatchLight(direction=(-1, 1, 0.8), depth=48),
    )
    drawing = paint.scene(paint.fill(silhouette, "#346b84", hatch=shaded))

Use ``save_png``, ``save_svg``, and ``save_gcode`` with the same scene.
The three exporters use the same clipped hatch centerlines. In PNG, the
centerlines are rasterized with antialiasing. A plotter uses the physical
nib width. Lights, stroke dimensions, and placement accept render
parameters and resolve anew for each export. These sampled path styles
use the export methods. The differentiable color graph from
``to_color()`` supports ordinary straight hatching.
