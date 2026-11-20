"""Learn a div's offset at one animated frame, then sweep the full motion.

Run with ``uv run python -m examples.center_div --steps 60``. Every optimizer
step and every animation frame are exported as PNGs before GIF assembly.
"""



import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sdf_ui import Canvas, color, param, sdf
from sdf_ui.anim import write_gif
from sdf_ui.learning import Adam, Parameter

_CENTER_X_TEXT = 'center_x'

def create_animation(steps=60, output=Path("out/center_div")):
    width, height = 640, 360
    div_half_size = (90, 56)
    div_size = (div_half_size[0] * 2, div_half_size[1] * 2)
    training_progress = 0.5
    offset = Parameter("offset", -180.0)
    center_x = Parameter(
        _CENTER_X_TEXT,
        0.0,
        animate=lambda progress: 120 + 400 * progress + offset,
    )

    div, scene = _center_scene(height, div_half_size)

    output = Path(output)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    title_font, label_font = _center_fonts()
    learning_frame_paths = []
    motion_frame_paths = []
    losses = []

    with Canvas((width, height)) as canvas:
        with canvas.compile(
            {"div": div, "image": scene},
            {_CENTER_X_TEXT: center_x, "offset": offset},
        ) as graph:
            learning_frame_paths, losses = _render_learning_frames(
                graph, steps, width, height, training_progress, offset,
                frames_dir, div_size, title_font, label_font,
            )

    with Canvas((width, height)) as canvas:
        with canvas.session(
            params={_CENTER_X_TEXT: center_x, "offset": offset}, dirty_regions=True
        ) as renderer:
            motion_frame_paths = _render_motion_frames(
                renderer, scene, steps, width, height, offset,
                output / "motion", div_size, title_font, label_font,
            )

    gif_path = _save_animation_gifs(output, learning_frame_paths, motion_frame_paths)
    _print_animation_summary(
        learning_frame_paths, motion_frame_paths, gif_path, output,
        training_progress, offset, height, losses,
    )
    return gif_path


def _center_scene(height, div_half_size):
    div = sdf.rounded_rect((param(_CENTER_X_TEXT), height / 2), div_half_size, 18).fill(
        "#62bb47", "#14251b"
    )
    return div, color.clear("#111827").over(div).to_rgb()


def _center_fonts():
    font_path = Path(__file__).parents[1] / "src/sdf_ui/assets/Roboto-Regular.ttf"
    return ImageFont.truetype(str(font_path), 20), ImageFont.truetype(str(font_path), 15)


def _save_animation_gifs(output, learning_paths, motion_paths):
    gif_path = output / "center_div.gif"
    write_gif(learning_paths, gif_path, duration=80)
    write_gif(motion_paths, output / "center_div_motion.gif", duration=80)
    return gif_path


def _print_animation_summary(learning_paths, motion_paths, gif_path, output,
                             progress, offset, height, losses):
    center = 120 + 400 * progress + offset.item()
    print(
        f"Saved {len(learning_paths)} learning frames and {len(motion_paths)} motion frames; "
        f"GIFs: {gif_path}, {output / 'center_div_motion.gif'}; "
        f"center at progress {progress:.2f}=({center:.1f}, {height / 2:.1f}), "
        f"loss={losses[-1]:.6f}"
    )


def _center_loss(graph, width):
    delta = (graph.parameters[_CENTER_X_TEXT] - width / 2) / width
    return delta**2


def _render_learning_frames(
    graph, steps, width, height, progress, offset, frames_dir,
    div_size, title_font, label_font,
):
    paths, losses = [], []
    optimizer = Adam(graph.trainable_parameters, lr=4.0)
    for step in range(steps):
        optimizer.zero_grad()
        with graph.run(
            loss=lambda current: _center_loss(current, width), progress=progress
        ) as execution:
            execution.backward()
            optimizer.step()
        x_value = 120 + 400 * progress + offset.item()
        loss_value = ((x_value - width / 2) / width) ** 2
        frame_path = frames_dir / f"step_{step:04d}.png"
        with graph.run(progress=progress) as execution:
            execution.save("image", frame_path)
        _annotate_frame(
            frame_path, label=f"step {step + 1:03d}   fit progress {progress:.2f}",
            detail=f"loss {loss_value:.5f}", x=x_value, progress=progress,
            size=(width, height), div_size=div_size,
            title_font=title_font, label_font=label_font,
        )
        paths.append(frame_path)
        losses.append(loss_value)
    return paths, losses


def _render_motion_frames(
    renderer, scene, steps, width, height, offset, motion_dir,
    div_size, title_font, label_font,
):
    motion_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame_index in range(steps):
        progress = frame_index / max(steps - 1, 1)
        frame_path = motion_dir / f"frame_{frame_index:04d}.png"
        renderer.render(scene, progress=progress).save(frame_path)
        x_value = 120 + 400 * progress + offset.item()
        _annotate_frame(
            frame_path, label=f"animation frame {frame_index + 1:03d}/{steps:03d}",
            detail=f"progress {progress:.2f}", x=x_value, progress=progress,
            size=(width, height), div_size=div_size,
            title_font=title_font, label_font=label_font,
        )
        paths.append(frame_path)
    return paths


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
    _draw_target_guides(draw, target_x, target_y, div_width, div_height)
    _draw_frame_labels(
        draw, width, height, target_x, target_y, x, progress,
        label, detail, title_font, label_font,
    )

    image.save(path)
    image.close()


def _draw_target_guides(draw, target_x, target_y, div_width, div_height):
    box = (
        target_x - div_width // 2, target_y - div_height // 2,
        target_x + div_width // 2, target_y + div_height // 2,
    )
    _dashed_rectangle(draw, box, fill=(225, 235, 255, 125), width=2)
    draw.line((target_x - 12, target_y, target_x + 12, target_y),
              fill=(255, 255, 255, 220), width=2)
    draw.line((target_x, target_y - 12, target_x, target_y + 12),
              fill=(255, 255, 255, 220), width=2)


def _draw_frame_labels(draw, width, height, target_x, target_y, x, progress,
                       label, detail, title_font, label_font):
    draw.rounded_rectangle((20, 18, 390, 104), radius=12, fill=(10, 16, 28, 220))
    draw.text((36, 24), "CENTER A DIV", font=title_font, fill=(246, 248, 252))
    draw.text((36, 49), label, font=label_font, fill=(208, 218, 234))
    draw.text((36, 74), detail, font=label_font, fill=(176, 191, 214))
    footer = f"center = ({x:5.1f}, {target_y:5.1f})    target = ({target_x}, {target_y})    p={progress:.2f}"
    draw.rounded_rectangle((20, height - 48, 435, height - 18), radius=8,
                           fill=(10, 16, 28, 220))
    draw.text((32, height - 42), footer, font=label_font, fill=(230, 236, 246))


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
