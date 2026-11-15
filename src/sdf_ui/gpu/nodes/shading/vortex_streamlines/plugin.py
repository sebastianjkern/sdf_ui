"""Draw antialiased streamlines for a smooth vortex field."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from sdf_ui.core.color import ColorSpaceMode, ColorTexture
from sdf_ui.gpu.nodes._vortex import normalize_vortices, validate_radius, vortex_velocity
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def _rgba8(value):
    channels = tuple(value)
    if len(channels) == 3:
        channels += (1.0,)
    if len(channels) != 4:
        raise ValueError("streamline colors must have three or four channels")
    if any(abs(channel) > 1 for channel in channels):
        channels = tuple(channel / 255 for channel in channels)
    return tuple(int(np.clip(channel, 0.0, 1.0) * 255) for channel in channels)


def _trace(seed_x, seed_y, direction, vortices, radius, step_size, steps, bounds):
    x, y = float(seed_x), float(seed_y)
    points = [(x, y)]
    width, height = bounds
    for _ in range(steps):
        vx, vy = vortex_velocity(
            np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32),
            vortices, radius,
        )
        speed = float(np.hypot(vx, vy))
        if speed < 0.018:
            break
        x += direction * float(vx) / speed * step_size
        y += direction * float(vy) / speed * step_size
        if x < 2 or y < 2 or x >= width - 2 or y >= height - 2:
            break
        points.append((x, y))
    return points


def render_streamlines(renderer, inputs, params):
    vortices = normalize_vortices(params["vortices"])
    radius = validate_radius(params["radius"])
    spacing = float(params["spacing"])
    step_size = float(params["step_size"])
    steps = int(params["steps"])
    if not np.isfinite(spacing) or spacing < 12:
        raise ValueError("streamline spacing must be finite and at least 12")
    if not np.isfinite(step_size) or step_size <= 0 or not 1 <= steps <= 2000:
        raise ValueError("step_size must be positive and steps must be from 1 to 2000")

    width, height = renderer.ctx.size
    scale = 2
    size = (width * scale, height * scale)
    glow = Image.new("RGBA", size)
    glow_draw = ImageDraw.Draw(glow)
    crisp = Image.new("RGBA", size)
    crisp_draw = ImageDraw.Draw(crisp)
    line_color = _rgba8(params["color"])
    glow_color = _rgba8(params["glow_color"])
    seed_step = max(16.0, spacing)

    for seed_y in np.arange(seed_step / 2, height, seed_step):
        for seed_x in np.arange(seed_step / 2, width, seed_step):
            forward = _trace(seed_x, seed_y, 1, vortices, radius, step_size, steps, (width, height))
            backward = _trace(seed_x, seed_y, -1, vortices, radius, step_size, steps, (width, height))
            points = list(reversed(backward[1:])) + forward
            if len(points) < 4:
                continue
            scaled = [(round(x * scale), round(y * scale)) for x, y in points]
            glow_draw.line(scaled, fill=glow_color, width=5 * scale, joint="curve")
            crisp_draw.line(scaled, fill=line_color, width=scale, joint="curve")

    glow = glow.filter(ImageFilter.GaussianBlur(radius=3 * scale))
    image = Image.alpha_composite(glow, crisp).resize((width, height), Image.Resampling.LANCZOS)
    texture = renderer.ctx.rgba8()
    texture.write(np.asarray(image, dtype=np.uint8).tobytes())
    return ColorTexture(tex=texture, context=renderer.ctx, mode=ColorSpaceMode.RGB)


def register_plugins(registry):
    registry.register(
        Plugin(
            "vortex_streamlines",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            params=(
                "vortices", "radius", "spacing", "step_size", "steps", "color",
                "glow_color",
            ),
            defaults={
                "spacing": 56.0,
                "step_size": 2.0,
                "steps": 220,
                "color": (0.44, 0.94, 0.89, 0.82),
                "glow_color": (0.18, 0.78, 0.76, 0.48),
            },
            # Additional controls are optional graph parameters too.
            render_func=render_streamlines,
            public=True,
        )
    )
