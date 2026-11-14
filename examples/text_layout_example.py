"""Show text wrapping and alignment inside reusable button boxes."""

from pathlib import Path

from sdf_ui import Canvas, color, sdf
from sdf_ui.text import measure_text


def text_layout_example(output=Path("out/examples/text_layout.png")):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Canvas((640, 360)) as canvas:
        scene = color.clear("#111827")
        buttons = (
            (320, 238, 360, "Create project", 26),
            (320, 126, 260, "Continue without saving changes", 23),
        )
        for center_x, center_y, width, label, font_size in buttons:
            layout = measure_text(label, size=font_size, width=width - 32)
            height = max(72, layout["height"] + 20)
            left = center_x - width / 2
            bottom = center_y - height / 2
            button = sdf.rounded_rect((center_x, center_y), (width, height), 12).fill(
                "#263449", "#00000000"
            )
            text = sdf.text(
                label,
                size=font_size,
                box=(left + 16, bottom, width - 32, height),
                align="center",
                valign="middle",
            ).fill("#f8fafc", "#00000000")
            scene = scene.over(button).over(text)

        scene.to_rgb().save(output, ctx=canvas)
    return output


if __name__ == "__main__":
    print(text_layout_example())
