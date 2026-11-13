API Reference
=============

.. automodule:: sdf_ui
   :members:
   :undoc-members:
   :show-inheritance:

Render nodes and sessions
-------------------------

``TextureNode`` is the base type returned by ``sdf`` and ``color`` factories.
``Renderer`` is returned by ``Canvas.session()`` and exposes cache and render
statistics for repeated evaluations.

.. automodule:: sdf_ui.core.texture
   :members: TextureNode, Renderer, RenderStats, RenderCacheInfo, MultiOutputResult
   :undoc-members:
   :show-inheritance:

Learning API
------------

The :mod:`sdf_ui.learning` namespace contains the CPU autodiff, optimizers,
layout tools, policies, and reference renderer. See :doc:`learning` for
end-to-end workflows.

.. automodule:: sdf_ui.learning
   :members:
   :undoc-members:
   :show-inheritance:

Graph API
---------

The :mod:`sdf_ui.graph` namespace exposes compiled graphs, executions, training
sessions, datasets, and render-backed policy environments. See
:doc:`gpu_render_graph` for native execution and training examples.

.. automodule:: sdf_ui.graph
   :members:
   :undoc-members:
   :show-inheritance:

GPU resources
-------------

.. automodule:: sdf_ui.gpu.image
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sdf_ui.gpu.resources
   :members:
   :undoc-members:
   :show-inheritance:

GPU optimizers
--------------

.. automodule:: sdf_ui.gpu.optim
   :members:
   :undoc-members:
   :show-inheritance:
