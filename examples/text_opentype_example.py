"""Show OpenType ligatures and positioning feature controls."""

from pathlib import Path

from sdf_ui import Canvas, color, sdf
from sdf_ui.text import measure_text


def text_opentype_example(output=Path("out/examples/text_opentype.png")):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Canvas((720, 360)) as canvas:
        scene = color.clear("#111827")
        rows = (
            ("Default OpenType", "office AVATAR", {}),
            ("Ligatures disabled", "office AVATAR", {"liga": False}),
            ("Kerning disabled", "AVATAR To Wa", {"kern": False}),
            ("Combining mark", "cafe\u0301", {}),
        )
        for index, (label, value, features) in enumerate(rows):
            center_y = 300 - index * 78
            scene = scene.over(
                sdf.text(
                    label,
                    size=17,
                    box=(36, center_y + 7, 648, 28),
                    valign="middle",
                ).fill("#94a3b8", "#00000000")
            )
            layout = measure_text(value, size=34, features=features)
            scene = scene.over(
                sdf.text(
                    value,
                    size=34,
                    features=features,
                    box=(36, center_y - 34, max(layout["width"] + 24, 648), 48),
                    valign="middle",
                ).fill("#f8fafc", "#00000000")
            )

        scene.to_rgb().save(output, canvas=canvas)
    return output


if __name__ == "__main__":
    print(text_opentype_example())
