"""Common color-output helper for layer shader programs."""


def color_program(body):
    return (
        "void evaluate(ivec2 pos, out D o[8]) {\n"
        + body
        + "\no[0]=col.x; o[1]=col.y; o[2]=col.z; o[3]=col.w;\n}\n"
    )
