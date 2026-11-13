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
