"""Rotation math shared by grid and rectangle shader extensions."""


def rotate(x, y, angle):
    """Apply the row-vector rotation used by the native shader."""
    return x * angle.cos() - y * angle.sin(), x * angle.sin() + y * angle.cos()


GLSL = r"""
D2 primitiveRotate(D2 p,D angle) {
    D co=cosD(angle), si=sinD(angle);
    return D2(sub(mul(p.x,co),mul(p.y,si)),
              add(mul(p.x,si),mul(p.y,co)));
}
"""
