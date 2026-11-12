"""Ordered Bayer thresholds shared by dithering shader extensions."""


def threshold(context):
    import numpy as np

    bayer = np.array((
        (0, 32, 8, 40, 2, 34, 10, 42),
        (48, 16, 56, 24, 50, 18, 58, 26),
        (12, 44, 4, 36, 14, 46, 6, 38),
        (60, 28, 52, 20, 62, 30, 54, 22),
        (3, 35, 11, 43, 1, 33, 9, 41),
        (51, 19, 59, 27, 49, 17, 57, 25),
        (15, 47, 7, 39, 13, 45, 5, 37),
        (63, 31, 55, 23, 61, 29, 53, 21),
    ))
    x = np.trunc(context.x.data).astype(np.intp) % 8
    y = np.trunc(context.y.data).astype(np.intp) % 8
    return (bayer[x, y] + 1.0) / 64.0


GLSL = r"""
float effectsThreshold(ivec2 pos) {
    // The original shader indexes dither[x][y], not image rows [y][x].
    const int bayer[64] = int[](
         0,32, 8,40, 2,34,10,42,
        48,16,56,24,50,18,58,26,
        12,44, 4,36,14,46, 6,38,
        60,28,52,20,62,30,54,22,
         3,35,11,43, 1,33, 9,41,
        51,19,59,27,49,17,57,25,
        15,47, 7,39,13,45, 5,37,
        63,31,55,23,61,29,53,21
    );
    return float(bayer[(pos.x % 8)*8 + pos.y % 8] + 1)/64.0;
}
"""
