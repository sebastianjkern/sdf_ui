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
