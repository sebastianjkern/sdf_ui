"""Animate a learned rounded rectangle moving to the center of a canvas.

Run with ``uv run python -m examples.center_div --steps 60``. Every optimizer
step is exported as a PNG before the frames are assembled into a GIF.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sdf_ui import Canvas, color, sdf
from sdf_ui.learning import Adam, Parameter


def create_animation(steps=60, output=Path("out/center_div")):
    width, height = 640, 360
    div_half_size = (90, 56)
    div_size = (div_half_size[0] * 2, div_half_size[1] * 2)
    center_x = Parameter("center_x", 135.0)
    center_y = Parameter("center_y", 260.0)

    div = sdf.rounded_rect((center_x, center_y), div_half_size, 18).fill(
        "#62bb47", "#14251b"
    )
    scene = color.clear("#111827").over(div).to_rgb()

    output = Path(output)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    font_path = Path(__file__).parents[1] / "src/sdf_ui/assets/Roboto-Regular.ttf"
    title_font = ImageFont.truetype(str(font_path), 20)
    label_font = ImageFont.truetype(str(font_path), 15)
    frame_paths = []
    losses = []

    def loss(graph):
        del graph  # This objective depends on the layout parameters alone.
        dx = (center_x - width / 2) / width
        dy = (center_y - height / 2) / height
        return dx**2 + dy**2

    with Canvas((width, height)) as canvas:
        with canvas.compile(
            {"div": div, "image": scene},
            {"center_x": center_x, "center_y": center_y},
        ) as graph:
            optimizer = Adam(graph.trainable_parameters, lr=4.0)
            for step in range(steps):
                optimizer.zero_grad()
                with graph.run(loss=loss) as execution:
                    execution.backward()
                    optimizer.step()

                loss_value = ((center_x.item() - width / 2) / width) ** 2 + (
                    (center_y.item() - height / 2) / height
                ) ** 2
                frame_path = frames_dir / f"step_{step:04d}.png"
                with graph.run() as execution:
                    execution.save("image", frame_path)
                _annotate_frame(
                    frame_path,
                    step=step + 1,
                    x=center_x.item(),
                    y=center_y.item(),
                    loss=loss_value,
                    size=(width, height),
                    div_size=div_size,
                    title_font=title_font,
                    label_font=label_font,
                )
                frame_paths.append(frame_path)
                losses.append(loss_value)

    gif_path = output / "center_div.gif"
    frames = [
        Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE)
        for path in frame_paths
    ]
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()

    print(
        f"Saved {len(frame_paths)} frames and {gif_path}; "
        f"center=({center_x.item():.1f}, {center_y.item():.1f}), "
        f"loss={losses[-1]:.6f}"
    )
    return gif_path


def _annotate_frame(
    path,
    *,
    step,
    x,
    y,
    loss,
    size,
    div_size,
    title_font,
    label_font,
):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = size
    div_width, div_height = div_size
    target_x, target_y = width // 2, height // 2

    # The dashed box marks the final div bounds; the crosshair marks its center.
    box = (
        target_x - div_width // 2,
        target_y - div_height // 2,
        target_x + div_width // 2,
        target_y + div_height // 2,
    )
    _dashed_rectangle(draw, box, fill=(225, 235, 255, 125), width=2)
    draw.line(
        (target_x - 12, target_y, target_x + 12, target_y),
        fill=(255, 255, 255, 220),
        width=2,
    )
    draw.line(
        (target_x, target_y - 12, target_x, target_y + 12),
        fill=(255, 255, 255, 220),
        width=2,
    )

    draw.rounded_rectangle((20, 18, 284, 78), radius=12, fill=(10, 16, 28, 220))
    draw.text((36, 27), "CENTER A DIV", font=title_font, fill=(246, 248, 252))
    draw.text(
        (36, 54),
        f"step {step + 1:03d}   loss {loss:.4f}",
        font=label_font,
        fill=(176, 191, 214),
    )

    footer = f"center = ({x:5.1f}, {y:5.1f})    target = ({target_x}, {target_y})"
    draw.rounded_rectangle(
        (20, height - 48, 355, height - 18), radius=8, fill=(10, 16, 28, 220)
    )
    draw.text((32, height - 42), footer, font=label_font, fill=(230, 236, 246))

    image.save(path)
    image.close()


def _dashed_rectangle(draw, box, *, fill, width, dash=8, gap=6):
    left, top, right, bottom = box
    for start in range(left, right, dash + gap):
        draw.line((start, top, min(start + dash, right), top), fill=fill, width=width)
        draw.line(
            (start, bottom, min(start + dash, right), bottom), fill=fill, width=width
        )
    for start in range(top, bottom, dash + gap):
        draw.line(
            (left, start, left, min(start + dash, bottom)), fill=fill, width=width
        )
        draw.line(
            (right, start, right, min(start + dash, bottom)), fill=fill, width=width
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("out/center_div"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    create_animation(steps=args.steps, output=args.output)


if __name__ == "__main__":
    main()
