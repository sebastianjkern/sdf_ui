Plotting
========

Use :class:`sdf_ui.plot.Plot` to put functions, geometry, and data on a
canvas. Both workflows produce ordinary scene layers that can be saved,
overlaid, or combined with other SDF UI rendering.

Plot a function or draw it directly
-----------------------------------

Create a plot with a canvas box and explicit data ranges. In canvas
coordinates the box starts at the lower left and uses full dimensions.
Use ``curve()`` or ``parametric()`` for sampled functions, or use
``line()``, ``scatter()``, and ``bars()`` for explicit data points::

    from math import sin
    from sdf_ui import Canvas, color
    from sdf_ui.plot import Plot

    plot = Plot((60, 55, 520, 245), (-4, 4), (-1.5, 1.5))
    scene = color.clear("#111c2b").over(
        plot.axes(x_ticks=None, y_ticks=None),
        plot.curve(sin).stroke(3, "#66e0c5"),
    )
    with Canvas((640, 360)) as canvas:
        canvas.save(scene, "wave.png")

Plot data with layers
---------------------

When your data has named columns, start with ``Plot.from_data()``. Add
marks with ``+`` and map columns to axes, color, or grouping with ``aes()``.
The plot infers shared scales from its layers and can arrange panels with
``facet_wrap()``::

    from sdf_ui import Canvas, color
    from sdf_ui.plot import Plot, aes, geom_bar, labs

    rows = [{"day": day} for day in ("Mon", "Tue", "Tue", "Wed", "Wed", "Wed")]
    plot = Plot.from_data(rows, aes(x="day")) + geom_bar() + labs(title="Orders by day")
    with Canvas((640, 420)) as canvas:
        scene = color.clear("#111c2b").over(plot.scene((20, 20, 600, 380)))
        canvas.save(scene, "orders.png")

``geom_bar()`` counts rows. Use ``geom_col()`` if the heights are already
in your data. Other layers include points, lines, and vertical intervals.
All are available from ``sdf_ui.plot``.

Distributions and intervals
---------------------------

Use ``geom_histogram(bins=16)`` with a numeric x mapping to count observations.
Each layer computes equal-width bins once across all its groups and facets.
The bins include their left edge and exclude their right edge, except for the
final bin, which also includes its right edge. Groups in one panel overlay
one another, so set ``alpha`` or facet by group when the bars overlap.

For measurements with lower and upper values, map ``ymin`` and ``ymax``.
Use ``geom_ribbon()`` to fill between successive rows in x order and
``geom_errorbar()`` for individual readings with caps. Use ``geom_rect()``
to supply both x and y bounds, such as for custom bins or shaded data windows.

For direct drawing, ``Plot.ribbon()`` and ``Plot.rectangles()`` return
SDFs with lazy geometry support. Ribbons interpolate between samples and
slightly overlap internal spans to hide antialiasing seams. Pass ``None``
to break a ribbon into separate runs. Histogram binning uses fixed numeric
observations at build time.

Learning parameters
-------------------

Explicit data marks can use lazy coordinates. Adaptive function curves
are sampled during scene construction. With ``Plot.from_data()``, give
explicit limits for axes that contain lazy values. Use ``build(box)`` to
get separate ``chrome`` and ``marks`` layers. Render the axes and text
in ``chrome`` once and keep the resulting texture open while the compiled
graph renders and trains ``marks``. See :doc:`learning` for graph training.

Examples
--------

Run these modules from the repository root::

    uv run --extra math python -m examples.plot_gallery
    uv run python -m examples.mark_gallery
    uv run python -m examples.chart_grammar
    uv run python -m examples.plot_distributions
