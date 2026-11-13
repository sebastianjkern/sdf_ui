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
