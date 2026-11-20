Spatial parameters
==================

A texture can supply a numeric shader parameter. Select a channel with
``channel("red")`` or use vector aliases such as ``channel("x")``. An SDF
has one channel, which contains its signed distance in canvas pixels.
Color channels are read as stored. Call ``to_rgb()`` or ``to_linear()``
before selecting a channel if the encoding matters.

``at_pixel()`` samples the map at the output pixel. ``at_cell()`` samples
at the center of the innermost repetition cell, so the sampled parameter
stays constant within one copy. Both sample bilinearly. Outside repetition,
``at_cell()`` uses the output pixel. Spatial expressions support arithmetic,
ordinary ``param()`` values, ``clamp()``, and ``remap()``. Group scalar
expressions in a tuple to drive a vector or color.

Procedural stamps
-----------------

This stamp varies radius per copy and varies its color within each copy::

    from sdf_ui import Canvas, color, field, sdf

    noise = field.turbulence(frequency=0.007, amplitude=1, seed=21)
    radius = noise.channel("x").at_cell().remap((-0.4, 0.6), (0, 5.5))
    tint = noise.channel("y").at_pixel().remap((-1, 1), (0.15, 0.55))
    stamp = sdf.circle((0, 0), radius).ink_linear((0.015, tint, 0.9, 1))
    dots = stamp.repeat_grid(60, 60, (12, 12), (360, 360))
    scene = color.clear("#ffffff").over(dots).finish(tone=0)
    with Canvas((720, 720)) as canvas:
        canvas.save_tiled(scene, "dots.png", tile_size=(256, 256))

``repeat_grid()`` supports finite grids and ``repeat()`` can fill the canvas.
Procedural templates are authored around ``(0, 0)`` by default. Use
``stamp_center`` for a template authored elsewhere. Each procedural copy
is clipped to its cell. The existing SDF repetition calls retain their
original behavior when used with a static stamp and no additional options.

Rotation and scale can be driven by maps on the stamp itself or supplied as
``rotation=`` and ``scale=`` to the repetition. The latter also works for
already rendered color, field, or SDF stamps. Use ``baked=True`` to materialize a
stamp once. Baked stamps keep their internal shapes and colors. Compose
opacity and tint as separate layers, or keep the original procedural template
to control its internal parameters.

Dependencies and execution
--------------------------

Maps can come from procedural textures, uploaded images, or other
texture-driven nodes. Map outputs remain graph dependencies and can be shared
across parameters and layers. Load an image with ``canvas.load_image(path)``
and keep the returned texture open while using it. Image samples use canvas
pixel coordinates and clamp at the image's edges. The shader compiler calls existing shader
functions with the parameters resolved at the current sample. It generates one
program for the scene structure. Replacing image pixels or changing numeric
uniforms uses the same GPU pipeline.

Spatial values can drive numeric shader uniforms and components of vectors
or arrays. Use fixed values for graph structure such as polygon vertex count,
finite grid dimensions, and host filter pass counts. Host-only artwork can
serve as a map or baked stamp. Changing a host operation's internal parameters
per cell requires a callable GPU evaluator for that operation.
Plugins can supply a ``spatial_uniforms`` adapter when their normal uniform
adapter coerces values to Python numbers. The adapter maps public parameters
to shader uniforms while preserving expressions. A composed plugin can use
``spatial_lowering`` to supply its equivalent subgraph.

Compiled GPU graphs can train continuous spatial controls and parameters
upstream of the maps. Generated passes use local centered differences on the
GPU and accumulate shared texel gradients. This is approximate and can cost
more than the hand-authored derivative kernels for individual operations.
Discrete control boundaries can give unstable gradients. These fused passes
do not have a separate CPU reference evaluator. As with other compiled graph
inputs, uploaded maps must match the canvas size.

Tiled stills support procedural stamps with bounded sampling reach and
complete, already rendered maps and baked stamps. Clamp sampled spacing or
nested scale when it affects the sampling reach. Render a procedural stamp
once with ``canvas.render(stamp)`` before using it as a baked stamp in
``save_tiled()``. Animation sessions render spatial scenes as full frames
and discard temporary image caches between frames.

Examples
--------

Run ``uv run python -m examples.spatial_parameters`` to render a halftone,
composite stamps with dependent maps, a repeated baked design, and a grid
of outlined squares that drift and rotate more toward the bottom. The
squares use four interleaved grids so neighboring outlines can overlap
across cell boundaries. The images are saved in
``out/generative_art/spatial_parameters/``.

Run ``uv run python -m examples.quadtree_mosaic`` for an adaptive mosaic
of rounded squares. The example splits regions with high luminance contrast
into four quadrants and stores each node's tone and split state in a parameter map.
Four repeated SDF layers sample the map to select which squares to draw and
what tone to use. Pass ``--image path/to/photo.jpg`` to use your own
guide or lower ``--contrast`` to split more regions. Set ``--max-depth``
from 0 (64-pixel tiles) to 3 (tiles as small as 8 pixels). The scene reads
``max_depth`` at render time, so you can reuse the scene and parameter map
at different depths. Depth makes a discrete choice at each split.
The default output is
``out/generative_art/quadtree_mosaic.png``.
