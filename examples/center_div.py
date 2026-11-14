"""Learn a div's offset at one animated frame, then sweep the full motion.

Run with ``uv run python -m examples.center_div --steps 60``. Every optimizer
step and every animation frame are exported as PNGs before GIF assembly.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sdf_ui import Canvas, color, param, sdf
from sdf_ui.learning import Adam, Parameter


def create_animation(steps=60, output=Path("out/center_div")):
    width, height = 640, 360
    div_half_size = (90, 56)
    div_size = (div_half_size[0] * 2, div_half_size[1] * 2)
    training_progress = 0.5
    offset = Parameter("offset", -180.0)
    center_x = Parameter(
        "center_x",
        0.0,
        animate=lambda progress: 120 + 400 * progress + offset,
    )

    div = sdf.rounded_rect((param("center_x"), height / 2), div_half_size, 18).fill(
        "#62bb47", "#14251b"
    )
    scene = color.clear("#111827").over(div).to_rgb()

    output = Path(output)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    font_path = Path(__file__).parents[1] / "src/sdf_ui/assets/Roboto-Regular.ttf"
    title_font = ImageFont.truetype(str(font_path), 20)
    label_font = ImageFont.truetype(str(font_path), 15)
    learning_frame_paths = []
    motion_frame_paths = []
    losses = []

    def loss(graph):
        # The animation curve is fixed at this frame; only its learned offset moves.
        dx = (graph.parameters["center_x"] - width / 2) / width
        return dx**2

    with Canvas((width, height)) as canvas:
        with canvas.compile(
            {"div": div, "image": scene},
            {"center_x": center_x, "offset": offset},
        ) as graph:
            optimizer = Adam(graph.trainable_parameters, lr=4.0)
            for step in range(steps):
                optimizer.zero_grad()
                with graph.run(loss=loss, progress=training_progress) as execution:
                    execution.backward()
                    optimizer.step()

                x_value = 120 + 400 * training_progress + offset.item()
                loss_value = ((x_value - width / 2) / width) ** 2
                frame_path = frames_dir / f"step_{step:04d}.png"
                with graph.run(progress=training_progress) as execution:
                    execution.save("image", frame_path)
                _annotate_frame(
                    frame_path,
                    label=f"step {step + 1:03d}   fit progress {training_progress:.2f}",
                    detail=f"loss {loss_value:.5f}",
                    x=x_value,
                    progress=training_progress,
                    size=(width, height),
                    div_size=div_size,
                    title_font=title_font,
                    label_font=label_font,
                )
                learning_frame_paths.append(frame_path)
                losses.append(loss_value)

    with Canvas((width, height)) as canvas:
        with canvas.compile(
            {"div": div, "image": scene},
            {"center_x": center_x, "offset": offset},
        ) as graph:
            motion_dir = output / "motion"
            motion_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in range(steps):
                progress = frame_index / max(steps - 1, 1)
                frame_path = motion_dir / f"frame_{frame_index:04d}.png"
                with graph.run(progress=progress) as execution:
                    execution.save("image", frame_path)
                _annotate_frame(
                    frame_path,
                    label=f"animation frame {frame_index + 1:03d}/{steps:03d}",
                    detail=f"progress {progress:.2f}",
                    x=120 + 400 * progress + offset.item(),
                    progress=progress,
                    size=(width, height),
                    div_size=div_size,
                    title_font=title_font,
                    label_font=label_font,
                )
                motion_frame_paths.append(frame_path)

    gif_path = output / "center_div.gif"
    motion_gif_path = output / "center_div_motion.gif"
    _assemble_gif(learning_frame_paths, gif_path)
    _assemble_gif(motion_frame_paths, motion_gif_path)

    print(
        f"Saved {len(learning_frame_paths)} learning frames and "
        f"{len(motion_frame_paths)} motion frames; "
        f"GIFs: {gif_path}, {motion_gif_path}; "
        f"center at progress {training_progress:.2f}="
        f"({120 + 400 * training_progress + offset.item():.1f}, {height / 2:.1f}), "
        f"loss={losses[-1]:.6f}"
    )
    return gif_path


def _assemble_gif(frame_paths, path):
    frames = [
        Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE)
        for frame_path in frame_paths
    ]
    try:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()


def _annotate_frame(
    path,
    *,
    label,
    detail,
    x,
    progress,
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

    draw.rounded_rectangle((20, 18, 390, 104), radius=12, fill=(10, 16, 28, 220))
    draw.text((36, 24), "CENTER A DIV", font=title_font, fill=(246, 248, 252))
    draw.text((36, 49), label, font=label_font, fill=(208, 218, 234))
    draw.text((36, 74), detail, font=label_font, fill=(176, 191, 214))

    footer = (
        f"center = ({x:5.1f}, {target_y:5.1f})    "
        f"target = ({target_x}, {target_y})    p={progress:.2f}"
    )
    draw.rounded_rectangle(
        (20, height - 48, 435, height - 18), radius=8, fill=(10, 16, 28, 220)
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
