"""GLSL helpers shared by shading extensions."""

RGBA = """
void shadingOutput(D4 color, out D o[8]) {
    o[0] = color.x; o[1] = color.y; o[2] = color.z; o[3] = color.w;
    for (int i = 4; i < 8; ++i) o[i] = c(0.0);
}
D4 shadingMix(D4 a, D4 b, D t) {
    return D4(mixD(a.x, b.x, t), mixD(a.y, b.y, t),
              mixD(a.z, b.z, t), mixD(a.w, b.w, t));
}
"""
