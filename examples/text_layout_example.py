"""Show text wrapping and alignment inside reusable button boxes."""

from pathlib import Path

from sdf_ui import Canvas, color, sdf
from sdf_ui.text import measure_glyph, measure_text, span
from sdf_ui.widgets import Panel


def _button_scene(scene):
    buttons = (
        (320, 238, 360, "Create project", 26),
        (320, 126, 260, "Continue without saving changes", 23),
    )
    for center_x, center_y, width, label, font_size in buttons:
        layout = measure_text(label, size=font_size, width=width - 32)
        height = max(48, layout["height"] - 16) if label.startswith("Continue") else max(72, layout["height"] + 20)
        clipped = measure_text(label, size=font_size, width=width - 32, height=height)["overflow"]
        panel = Panel((center_x - width / 2, center_y - height / 2, width, height), padding=(16, 0), radius=12)
        button = panel.layer("#263449")
        text = sdf.text(
            label, size=font_size, box=panel.content_box, align="center",
            valign="middle", overflow="clip" if clipped else "visible",
        ).fill("#f8fafc", "#00000000")
        scene = scene.over(button).over(text)
    return scene


def _paragraph_scene():
    paragraph = "Justified text fills the line by expanding word spaces while keeping the final paragraph line natural.   "
    background = sdf.rounded_rect_from_size((320, 29), (560, 46), 10).fill(
        "#263449", "#00000000"
    )
    text = sdf.text(
        paragraph, size=16, box=(40, 6, 560, 46), align="justify",
        valign="middle", overflow="clip", trim_trailing_spaces=False,
        decoration="underline", first_line_indent=18,
    ).fill("#f8fafc", "#00000000")
    return background.over(text)


def text_layout_example(output=Path("out/examples/text_layout.png")):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas((640, 360)) as canvas:
        measure_glyph("A", size=24)
        rich_label = sdf.text(
            [span("Styled ", size=25), span("SDF", size=34, decoration="underline")],
            box=(270, 230, 100, 100), align="center", valign="middle",
        ).fill("#f8fafc", "#00000000")
        scene = _button_scene(color.clear("#111827").over(rich_label))
        scene = scene.over(_paragraph_scene())
        scene.to_rgb().save(output, canvas=canvas)
    return output


if __name__ == "__main__":
    print(text_layout_example())
