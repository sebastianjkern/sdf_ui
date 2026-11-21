"""Finite rectangular repetition of an SDF image."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation


def _repeat_anchor(context, center, spacing, columns, rows):
    import numpy as np

    x, y = context.x.data, context.y.data
    anchor_x = np.floor((x - center[0]) / spacing[0] + (columns - 1) * 0.5 + 0.5)
    anchor_y = np.floor((y - center[1]) / spacing[1] + (rows - 1) * 0.5 + 0.5)
    return x, y, anchor_x, anchor_y


def _sample_cell(source, x, y, cell_x, cell_y, spacing, columns, rows, maximum):
    import numpy as np

    from sdf_ui.learning.autodiff import where

    height, width = source.shape
    valid = (cell_x >= 0) & (cell_x < columns) & (cell_y >= 0) & (cell_y < rows)
    offset_x = (cell_x - (columns - 1) * 0.5) * spacing[0]
    offset_y = (cell_y - (rows - 1) * 0.5) * spacing[1]
    sample_x = np.rint(x - offset_x).astype(np.int64)
    sample_y = np.rint(y - offset_y).astype(np.int64)
    valid &= (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    if not np.any(valid):
        return None
    values = source[np.clip(sample_y, 0, height - 1), np.clip(sample_x, 0, width - 1)]
    return where(valid, values, maximum)


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import Tensor, minimum

    source = inputs["sdf0"]
    columns, rows = int(params["columns"].item()), int(params["rows"].item())
    spacing = params["spacing"].data.astype(np.float32, copy=False)
    center = params["center"].data.astype(np.float32, copy=False)
    x, y, anchor_x, anchor_y = _repeat_anchor(context, center, spacing, columns, rows)
    height, width = source.shape
    maximum = max(width, height)
    distance = Tensor(np.full(context.x.shape, maximum, dtype=np.float32))
    for row_offset in range(-1, 2):
        for column_offset in range(-1, 2):
            candidate = _sample_cell(
                source, x, y, anchor_x + column_offset, anchor_y + row_offset,
                spacing, columns, rows, maximum,
            )
            if candidate is not None:
                distance = minimum(distance, candidate)
    return distance


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    (ImageBinding("sdf0", 1, read=True),),
    (
        Uniform("columns", "float"),
        Uniform("rows", "float"),
        Uniform("spacing", "vec2"),
        Uniform("center", "vec2"),
    ),
)
CONTRACT = ShaderDifferentiation(
    "repeat_grid",
    "gpu/nodes/primitives/repeat_grid/repeat_grid.wgsl",
    (ShaderArgument("sdf0", (), "r32f"),),
    (
        ShaderArgument("columns", (), "float"),
        ShaderArgument("rows", (), "float"),
        ShaderArgument("spacing", (2,), "vec2"),
        ShaderArgument("center", (2,), "vec2"),
    ),
    (ShaderArgument("destTex", (), "r32f"),),
    zero_gradient_parameters=("columns", "rows", "spacing", "center"),
    sampling="integer_zero_outside",
    reference=reference,
    notes="Finite 3x3 neighboring-cell sampling; each cell uses integer source coordinates.",
)
