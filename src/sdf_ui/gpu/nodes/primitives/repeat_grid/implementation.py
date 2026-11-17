"""Finite rectangular repetition of an SDF image."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import Tensor, minimum, where

    source = inputs["sdf0"]
    columns = int(params["columns"].item())
    rows = int(params["rows"].item())
    spacing = params["spacing"].data.astype(np.float32, copy=False)
    center = params["center"].data.astype(np.float32, copy=False)
    x, y = context.x.data, context.y.data
    anchor_x = np.floor((x - center[0]) / spacing[0] + (columns - 1) * 0.5 + 0.5)
    anchor_y = np.floor((y - center[1]) / spacing[1] + (rows - 1) * 0.5 + 0.5)
    height, width = source.shape
    distance = Tensor(np.full(context.x.shape, max(width, height), dtype=np.float32))
    for oy in range(-1, 2):
        for ox in range(-1, 2):
            cell_x, cell_y = anchor_x + ox, anchor_y + oy
            valid = (cell_x >= 0) & (cell_x < columns) & (cell_y >= 0) & (cell_y < rows)
            offset_x = (cell_x - (columns - 1) * 0.5) * spacing[0]
            offset_y = (cell_y - (rows - 1) * 0.5) * spacing[1]
            sample_x = np.rint(x - offset_x).astype(np.int64)
            sample_y = np.rint(y - offset_y).astype(np.int64)
            valid &= (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
            if np.any(valid):
                values = source[np.clip(sample_y, 0, height - 1), np.clip(sample_x, 0, width - 1)]
                distance = minimum(distance, where(valid, values, max(width, height)))
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
    "gpu/nodes/primitives/repeat_grid/shader.glsl",
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
BODY = r"""
void evaluate(ivec2 pos,out D o[8]) {
    vec2 counts=vec2(U_columns.v,U_rows.v);
    vec2 spacing=vec2(U_spacing.x.v,U_spacing.y.v);
    vec2 center=vec2(U_center.x.v,U_center.y.v);
    vec2 anchor=floor((vec2(pos)-center)/spacing+(counts-1.0)*0.5+0.5);
    D distance=c(float(max(imageSize(destTex).x,imageSize(destTex).y)));
    ivec2 sourceSize=imageSize(sdf0);
    for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
        vec2 cell=anchor+vec2(x,y);
        if(cell.x<0.0||cell.y<0.0||cell.x>=U_columns.v||cell.y>=U_rows.v) continue;
        vec2 offset=(cell-(counts-1.0)*0.5)*spacing;
        ivec2 source=ivec2(round(vec2(pos)-offset));
        if(any(lessThan(source,ivec2(0)))||any(greaterThanEqual(source,sourceSize))) continue;
        distance=minD(distance,ld(0,source,0));
    }
    o[0]=distance;
}
"""
PROGRAM = ShaderProgram("repeat_grid", BODY)
