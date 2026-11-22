from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping

from .vector import VectorDrawing, VectorPath


@dataclass(frozen=True)
class PlotterProfile:
    """Machine settings for converting logical canvas units to paper millimetres.

    ``size_mm`` is the usable physical paper size. ``mm_per_unit`` converts
    drawing coordinates to millimetres. Pen commands are complete machine
    commands supplied by the profile owner; optional change commands are
    selected by each path's pen name. A path with ``pen=None`` keeps the active
    pen without issuing a pen change.
    """

    size_mm: tuple[float, float]
    mm_per_unit: float
    pen_up_command: str
    pen_down_command: str
    travel_speed_mm_min: float
    draw_speed_mm_min: float
    pen_change_commands: Mapping[str, str] | None = None


def write_svg(drawing: VectorDrawing, path: str | Path) -> Path:
    paths = _normalized_paths(drawing)
    width, height = drawing.size
    elements = [_svg_path(vector_path, points) for vector_path, points in paths]
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(element for element in elements if element)
        + "\n</svg>\n"
    )
    return _write_text(path, content)


def write_gcode(
    drawing: VectorDrawing,
    path: str | Path,
    *,
    plotter: PlotterProfile,
) -> Path:
    paths = _normalized_paths(drawing)
    _validate_profile(drawing, plotter)
    commands = ["G21", "G90", plotter.pen_up_command]
    active_pen = None
    for vector_path, points in paths:
        if points:
            active_pen = _append_path(
                commands, vector_path, points, plotter, active_pen
            )
    if commands[-1] != plotter.pen_up_command:
        commands.append(plotter.pen_up_command)
    return _write_text(path, "\n".join(commands) + "\n")


def _svg_path(vector_path: VectorPath, points: tuple[tuple[float, float], ...]) -> str:
    if not points:
        return ""
    point_data = " ".join(f"{_number(x)},{_number(y)}" for x, y in points)
    return (
        f'<polyline points="{point_data}" fill="none" '
        f'stroke="{escape(vector_path.color, quote=True)}" '
        f'stroke-width="{_number(vector_path.width)}" '
        'stroke-linecap="round" stroke-linejoin="round" />'
    )


def _append_path(
    commands: list[str],
    vector_path: VectorPath,
    points: tuple[tuple[float, float], ...],
    profile: PlotterProfile,
    active_pen: str | None,
) -> str | None:
    if vector_path.pen is not None and vector_path.pen != active_pen:
        active_pen = _change_pen(commands, vector_path.pen, profile)
    _append_lift(commands, profile)
    first_x, first_y = points[0]
    commands.append(_travel_command(first_x, first_y, profile))
    commands.append(profile.pen_down_command)
    commands.extend(_draw_command(x, y, profile) for x, y in points[1:])
    commands.append(profile.pen_up_command)
    return active_pen


def _change_pen(commands: list[str], pen: str, profile: PlotterProfile) -> str:
    _append_lift(commands, profile)
    pen_commands = profile.pen_change_commands
    if pen_commands is None or pen not in pen_commands:
        raise ValueError(f"No pen change command configured for {pen!r}")
    command = pen_commands[pen]
    _validate_command("pen change command", command)
    commands.append(command)
    return pen


def _append_lift(commands: list[str], profile: PlotterProfile) -> None:
    if commands[-1] != profile.pen_up_command:
        commands.append(profile.pen_up_command)


def _travel_command(x: float, y: float, profile: PlotterProfile) -> str:
    return (
        f"G0 X{_number(x * profile.mm_per_unit)} "
        f"Y{_number(profile.size_mm[1] - y * profile.mm_per_unit)} "
        f"F{_number(profile.travel_speed_mm_min)}"
    )


def _draw_command(x: float, y: float, profile: PlotterProfile) -> str:
    return (
        f"G1 X{_number(x * profile.mm_per_unit)} "
        f"Y{_number(profile.size_mm[1] - y * profile.mm_per_unit)} "
        f"F{_number(profile.draw_speed_mm_min)}"
    )


def _validate_profile(drawing: VectorDrawing, profile: PlotterProfile) -> None:
    _validate_physical_size(profile)
    _validate_scale_and_fit(drawing, profile)
    _validate_command("pen_up_command", profile.pen_up_command)
    _validate_command("pen_down_command", profile.pen_down_command)
    _validate_speeds(profile)


def _validate_physical_size(profile: PlotterProfile) -> None:
    if len(profile.size_mm) != 2 or not all(
        math.isfinite(value) and value > 0 for value in profile.size_mm
    ):
        raise ValueError(
            "Plotter physical size must contain two positive finite values"
        )


def _validate_scale_and_fit(drawing: VectorDrawing, profile: PlotterProfile) -> None:
    if not math.isfinite(profile.mm_per_unit) or profile.mm_per_unit <= 0:
        raise ValueError("Plotter scale must be a positive finite value")
    width, height = drawing.size
    if (
        width * profile.mm_per_unit > profile.size_mm[0]
        or height * profile.mm_per_unit > profile.size_mm[1]
    ):
        raise ValueError("Drawing does not fit within the plotter workspace")


def _validate_command(name: str, command: str) -> None:
    if not command.strip() or "\n" in command or "\r" in command:
        raise ValueError(f"{name} must be a single non-empty command")


def _validate_speeds(profile: PlotterProfile) -> None:
    speeds = (profile.travel_speed_mm_min, profile.draw_speed_mm_min)
    if not all(math.isfinite(speed) and speed > 0 for speed in speeds):
        raise ValueError("Plotter speeds must be positive and finite")


def _normalized_paths(
    drawing: VectorDrawing,
) -> tuple[tuple[VectorPath, tuple[tuple[float, float], ...]], ...]:
    _validate_size(drawing)
    width, height = drawing.size
    return tuple(
        (vector_path, _normalized_points(vector_path, width, height))
        for vector_path in drawing.paths
    )


def _validate_size(drawing: VectorDrawing) -> None:
    width, height = drawing.size
    if width <= 0 or height <= 0:
        raise ValueError("Drawing size must be positive")


def _normalized_points(
    vector_path: VectorPath, width: int, height: int
) -> tuple[tuple[float, float], ...]:
    if not math.isfinite(vector_path.width) or vector_path.width < 0:
        raise ValueError("Path width must be finite and non-negative")
    return tuple(
        (_normalize_coordinate(x, width), _normalize_coordinate(y, height))
        for x, y in vector_path.points
    )


def _normalize_coordinate(value: float, upper: int) -> float:
    if not math.isfinite(value):
        raise ValueError("Path coordinates must be finite")
    tolerance = 8 * math.ulp(float(max(upper, 1)))
    if value < -tolerance or value > upper + tolerance:
        raise ValueError("Path coordinate lies outside the drawing workspace")
    return min(max(value, 0.0), float(upper))


def _write_text(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.write_text(content, encoding="utf-8")
    return destination


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
