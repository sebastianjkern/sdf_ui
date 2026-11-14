"""Show text wrapping and alignment inside reusable button boxes."""

from pathlib import Path

from sdf_ui import Canvas, color, sdf
from sdf_ui.text import measure_glyph, measure_text, span


def text_layout_example(output=Path("out/examples/text_layout.png")):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Canvas((640, 360)) as canvas:
        scene = color.clear("#111827")
        # Measure a single glyph outline as well as complete paragraphs.
        measure_glyph("A", size=24)
        rich_label = sdf.text(
            [span("Styled ", size=25), span("SDF", size=34, decoration="underline")],
            box=(270, 230, 100, 100),
            align="center",
            valign="middle",
        ).fill("#f8fafc", "#00000000")
        scene = scene.over(rich_label)
        buttons = (
            (320, 238, 360, "Create project", 26),
            (320, 126, 260, "Continue without saving changes", 23),
        )
        for center_x, center_y, width, label, font_size in buttons:
            layout = measure_text(label, size=font_size, width=width - 32)
            if label.startswith("Continue"):
                height = max(48, layout["height"] - 16)
            else:
                height = max(72, layout["height"] + 20)
            has_overflow = measure_text(
                label, size=font_size, width=width - 32, height=height
            )["overflow"]
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
                overflow="clip" if has_overflow else "visible",
            ).fill("#f8fafc", "#00000000")
            scene = scene.over(button).over(text)

        paragraph = (
            "Justified text fills the line by expanding word spaces while keeping "
            "the final paragraph line natural.   "
        )
        paragraph_box = (40, 6, 560, 46)
        paragraph_background = sdf.rounded_rect((320, 29), (560, 46), 10).fill(
            "#263449", "#00000000"
        )
        paragraph_text = sdf.text(
            paragraph,
            size=16,
            box=paragraph_box,
            align="justify",
            valign="middle",
            overflow="clip",
            trim_trailing_spaces=False,
            decoration="underline",
            first_line_indent=18,
        ).fill("#f8fafc", "#00000000")
        scene = scene.over(paragraph_background).over(paragraph_text)

        scene.to_rgb().save(output, ctx=canvas)
    return output


if __name__ == "__main__":
    print(text_layout_example())
