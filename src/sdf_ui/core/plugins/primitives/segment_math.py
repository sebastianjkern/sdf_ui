"""Segment distance shared by line, capsule, and related primitives."""


def minimum_candidates(*values):
    """Split the gradient evenly among equal minimum candidates."""
    from sdf_ui.learning.autodiff import stack, where

    values = stack(values, axis=-1)
    data = values.data
    active = data == data.min(axis=-1, keepdims=True)
    return where(active, values, 0.0).sum(axis=-1) / active.sum(axis=-1)


def segment(x, y, ax, ay, bx, by, epsilon=None):
    import numpy as np
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _clip, _length

    px, py = x - ax, y - ay
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if epsilon is not None:
        denominator = maximum(denominator, epsilon)
    elif np.any(denominator.data == 0):
        raise ValueError("line/triangle requires distinct segment endpoints")
    t = _clip((px * dx + py * dy) / denominator, 0.0, 1.0)
    return _length(px - t * dx, py - t * dy)


GLSL = r"""
D primitiveSegment(D2 p,D2 a,D2 b,float epsilon) {
    D2 delta=sub2(p,a), edge=sub2(b,a);
    D denominator=dot2(edge,edge);
    if(epsilon>0.0) denominator=maxD(denominator,c(epsilon));
    D t=clampD(divD(dot2(delta,edge),denominator),c(0.0),c(1.0));
    return len2(sub2(delta,mul2(edge,t)));
}
"""

MIN3_GLSL = r"""
D primitiveMin3(D a,D b,D q) {
    float value=min(a.v,min(b.v,q.v)), derivative=0.0, count=0.0;
    if(a.v==value) { derivative+=a.d; count+=1.0; }
    if(b.v==value) { derivative+=b.d; count+=1.0; }
    if(q.v==value) { derivative+=q.d; count+=1.0; }
    return D(value,derivative/count);
}
"""
