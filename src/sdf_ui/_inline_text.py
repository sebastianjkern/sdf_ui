"""Layout and render mixed-style inline text spans."""

from __future__ import annotations

import re
from math import isfinite
from numbers import Real

from uniseg.linebreak import line_break_units

_FALLBACK_FONTS = "fallback_fonts"
_OVERFLOW = "overflow"
_ALIGN = "align"
_SEGMENTS = "segments"
_WIDTH = "width"
_WHITE_SPACE = "white_space"
_ASCENT_RATIO = 4 / 5
_DESCENT_RATIO = 1 / 5


class InlineTextLayout:
    def __init__(self, value, common, text_factory):
        from ._font_graphemes import graphemes
        from .fonts import measure_text

        self.value = value
        self.common = common
        self.text_factory = text_factory
        self.graphemes = graphemes
        self.measure_shaped_text = measure_text
        self.x, self.y, self.box_width, self.box_height = self._validate_box()
        self._validate_options()
        self.base_options = self._make_base_options()
        self.default_style = {
            "size": common["size"],
            "path": common["path"],
            _FALLBACK_FONTS: common[_FALLBACK_FONTS],
            **self.base_options,
        }
        self.lines = [self._new_line(first=True)]

    def _validate_box(self):
        common = self.common
        box = common["box"]
        if box is None:
            return common["ox"], common["oy"], None, None
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise TypeError("text box must be (x, y, width, height)")
        if any(
            isinstance(item, bool) or not isinstance(item, Real) or not isfinite(item)
            for item in box
        ):
            raise ValueError("text box values must be finite numbers")
        x, y, width, height = map(float, box)
        if width <= 0 or height <= 0:
            raise ValueError("text box width and height must be positive")
        return x, y, width, height

    def _validate_options(self):
        common = self.common
        if not isinstance(self.value, (list, tuple)):
            raise TypeError("inline text value must be a sequence of span() values")
        if self.box_width is None and common[_OVERFLOW] != "visible":
            raise ValueError("inline text overflow modes require a box")
        if common[_OVERFLOW] not in {"visible", "clip", "error", "ellipsis"}:
            raise ValueError("unsupported inline text overflow mode")
        if common[_OVERFLOW] == "ellipsis" and common["max_lines"] is None:
            raise ValueError("ellipsis overflow requires max_lines")
        self._validate_alignment()
        self._validate_line_limit()
        self._validate_indent()

    def _validate_alignment(self):
        common = self.common
        if self.box_width is None and common[_ALIGN] != "left":
            raise ValueError("inline text alignment requires a box")
        if common[_ALIGN] not in {"left", "center", "right", "justify"}:
            raise ValueError("unsupported inline text alignment")
        if common["valign"] not in {"top", "middle", "bottom"}:
            raise ValueError("unsupported inline text vertical alignment")

    def _validate_line_limit(self):
        limit = self.common["max_lines"]
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("max_lines must be a positive integer")

    def _validate_indent(self):
        indent = self.common["first_line_indent"]
        if isinstance(indent, bool) or not isinstance(indent, Real) or not isfinite(indent):
            raise ValueError("first_line_indent must be a finite number")

    def _make_base_options(self):
        common = self.common
        return {
            "samples": common["samples"],
            "cache_size": common["cache_size"],
            "line_height": common["line_height"],
            "oversample": common["oversample"],
            "min_render_size": common["min_render_size"],
            "missing_glyph": common["missing_glyph"],
            "direction": common["direction"],
            "script": common["script"],
            "language": common["language"],
            "features": common["features"],
            "variations": common["variations"],
            "trim_trailing_spaces": False,
            "decoration": common["common_decoration"],
            "decoration_thickness": common["common_decoration_thickness"],
            "word_spacing": common["word_spacing"],
            "letter_spacing": common["letter_spacing"],
            "vertical_metrics": common["vertical_metrics"],
        }

    @staticmethod
    def _new_line(first=False, last=False):
        return {_SEGMENTS: [], _WIDTH: 0.0, "first": first, "last": last}

    def measure(self, value, style):
        common = self.common
        return self.measure_shaped_text(
            value,
            size=style["size"],
            path=style["path"],
            fallbacks=style[_FALLBACK_FONTS],
            line_height=common["line_height"],
            missing_glyph=style["missing_glyph"],
            direction=style["direction"],
            script=style["script"],
            language=style["language"],
            features=style["features"],
            variations=style["variations"],
            trim_trailing_spaces=False,
            word_spacing=style["word_spacing"],
            letter_spacing=style["letter_spacing"],
        )

    def add_segment(self, value, style):
        if not value:
            return
        line = self.lines[-1]
        if line[_SEGMENTS] and line[_SEGMENTS][-1][1] == style:
            previous = line[_SEGMENTS].pop()
            value = previous[0] + value
        metrics = self.measure(value, style)
        line[_SEGMENTS].append((value, style, metrics))
        line[_WIDTH] = sum(part[2][_WIDTH] for part in line[_SEGMENTS])

    def next_line(self, *, paragraph_first=False, paragraph_last=False):
        line = self.lines[-1]
        self._trim_line(line)
        line["last"] = paragraph_last
        self.lines.append(self._new_line(first=paragraph_first))

    def _trim_line(self, line):
        common = self.common
        if not common["trim_trailing_spaces"] or common[_WHITE_SPACE] == "pre":
            return
        while line[_SEGMENTS] and line[_SEGMENTS][-1][0].endswith((" ", "\t")):
            content, style, _metrics = line[_SEGMENTS].pop()
            content = content.rstrip(" \t")
            self.add_segment(content, style)
        line[_WIDTH] = sum(part[2][_WIDTH] for part in line[_SEGMENTS])

    def available_width(self):
        if self.box_width is None:
            return None
        indent = self.common["first_line_indent"] if self.lines[-1]["first"] else 0.0
        return self.box_width - indent

    def append_unit(self, unit, style):
        if self.common[_WHITE_SPACE] == "normal":
            unit = re.sub(r"[ \t]+", " ", unit)
        if unit:
            self._append_measured_unit(unit, style)

    def _append_measured_unit(self, unit, style):
        limit = self.available_width()
        width = self.measure(unit, style)[_WIDTH]
        line = self.lines[-1]
        if self._must_wrap_before_unit(line, width, limit, unit):
            if self.common[_WHITE_SPACE] == "normal" and unit.isspace():
                return
            self.next_line()
            limit = self.available_width()
        if self._must_wrap_unit(width, limit):
            self._append_graphemes(unit, style, limit)
        else:
            self.add_segment(unit, style)

    def _must_wrap_before_unit(self, line, width, limit, unit):
        common = self.common
        wraps = limit is not None and common[_WHITE_SPACE] != "pre"
        return bool(wraps and line[_WIDTH] and line[_WIDTH] + width > limit)

    def _must_wrap_unit(self, width, limit):
        return (
            limit is not None
            and self.common[_WHITE_SPACE] != "pre"
            and width > limit
        )

    def _append_graphemes(self, unit, style, limit):
        chunk = ""
        for grapheme in self.graphemes(unit):
            candidate = chunk + grapheme
            available = self.available_width() or limit
            if chunk and self.measure(candidate, style)[_WIDTH] > available:
                self.add_segment(chunk, style)
                self.next_line()
                chunk = grapheme
            else:
                chunk = candidate
        self.add_segment(chunk, style)

    def _span_style(self, item, allowed):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            content, overrides = item
            item = {"text": content, **overrides}
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise TypeError("each inline span must be span(text, **style) or (text, style)")
        overrides = {key: item[key] for key in item if key != "text"}
        unknown = set(overrides) - allowed
        if unknown:
            raise TypeError(f"unsupported inline text styles: {', '.join(sorted(unknown))}")
        return item["text"], {**self.default_style, **overrides}

    def _append_paragraphs(self, text, style, line_break_units):
        paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for index, paragraph in enumerate(paragraphs):
            if paragraph:
                self._append_paragraph(paragraph, style, line_break_units)
            if index < len(paragraphs) - 1:
                self.next_line(paragraph_first=True, paragraph_last=True)

    def _append_paragraph(self, paragraph, style, line_break_units):
        if self.common[_WHITE_SPACE] == "pre":
            self.append_unit(paragraph, style)
            return
        for unit in line_break_units(paragraph):
            self.append_unit(unit, style)

    def add_spans(self, line_break_units):
        allowed = {
            "size", "path", _FALLBACK_FONTS, "missing_glyph", "direction",
            "script", "language", "features", "variations", "decoration",
            "decoration_thickness", "word_spacing", "letter_spacing",
        }
        for item in self.value:
            text, style = self._span_style(item, allowed)
            self._append_paragraphs(text, style, line_break_units)
        self.lines[-1]["last"] = True
        if not self.lines[-1][_SEGMENTS] and len(self.lines) > 1:
            self.lines.pop()

    def truncate(self):
        limit = self.common["max_lines"]
        if limit is None or len(self.lines) <= limit:
            return
        self.lines = self.lines[:limit]
        self.lines[-1]["last"] = True
        self.lines[-1]["truncated"] = True
        if self.common[_OVERFLOW] == "ellipsis":
            self._append_ellipsis()

    def _append_ellipsis(self):
        line = self.lines[-1]
        style = line[_SEGMENTS][-1][1] if line[_SEGMENTS] else self.default_style
        limit = self.available_width()
        while self._ellipsis_does_not_fit(line, style, limit):
            self._remove_last_grapheme(line)
        self.add_segment("…", style)

    def _ellipsis_does_not_fit(self, line, style, limit):
        if limit is None or not line[_SEGMENTS]:
            return False
        return line[_WIDTH] + self.measure("…", style)[_WIDTH] > limit

    def _remove_last_grapheme(self, line):
        content, style, _metrics = line[_SEGMENTS].pop()
        clusters = list(self.graphemes(content))
        if clusters:
            clusters.pop()
        shortened = "".join(clusters)
        if shortened:
            self.add_segment(shortened, style)

    def _vertical_metrics(self):
        common = self.common
        size = float(common["size"])
        line_ascents = [
            max((part[2]["ascent"] for part in line[_SEGMENTS]), default=size * _ASCENT_RATIO)
            for line in self.lines
        ]
        line_descents = [
            min((part[2]["descent"] for part in line[_SEGMENTS]), default=-size * _DESCENT_RATIO)
            for line in self.lines
        ]
        return max(line_ascents, default=size * _ASCENT_RATIO), min(
            line_descents, default=-size * _DESCENT_RATIO
        )

    def _largest_run_size(self):
        return max(
            (
                float(style["size"])
                for line in self.lines
                for _content, style, _metrics in line[_SEGMENTS]
            ),
            default=float(self.common["size"]),
        )

    def _block_metrics(self):
        ascent, descent = self._vertical_metrics()
        advance = self._largest_run_size() * float(self.common["line_height"])
        height = ascent - descent + max(len(self.lines) - 1, 0) * advance
        overflow = self._measured_overflow(height)
        return ascent, descent, advance, height, overflow

    def _measured_overflow(self, height):
        if self.box_width is None:
            return False
        return height > self.box_height or any(
            line[_WIDTH] + (self.common["first_line_indent"] if line["first"] else 0.0)
            > self.box_width
            for line in self.lines
        )

    def _validate_overflow(self, measured):
        if self.common[_OVERFLOW] != "error":
            return
        if measured or any(line.get("truncated") for line in self.lines):
            raise ValueError("text layout exceeds the configured box")

    def _first_baseline(self, ascent, descent, block_height, advance):
        if self.box_width is None:
            return self.y
        if self.common["valign"] == "top":
            return self.y + self.box_height - ascent
        line_offset = max(len(self.lines) - 1, 0) * advance
        if self.common["valign"] == "bottom":
            return self.y - descent + line_offset
        return self.y + (self.box_height - block_height) / 2 - descent + line_offset

    def _render_line(self, index, line, first_baseline, advance):
        common = self.common
        width = line[_WIDTH]
        indent = common["first_line_indent"] if line["first"] else 0.0
        start_x = self._line_start(width, indent)
        spaces = sum(content.count(" ") for content, _style, _m in line[_SEGMENTS])
        extra = self._justify_space(width, indent, spaces, line["last"])
        baseline = first_baseline - index * advance
        return self._render_segments(line, start_x, baseline, extra)

    def _line_start(self, width, indent):
        if self.box_width is None:
            return self.x + indent
        if self.common[_ALIGN] == "center":
            return self.x + (self.box_width - width) / 2
        if self.common[_ALIGN] == "right":
            return self.x + self.box_width - width
        return self.x + indent

    def _justify_space(self, width, indent, spaces, last):
        if self.common[_ALIGN] != "justify" or last or not spaces:
            return 0.0
        target = self.box_width - indent if self.box_width is not None else width
        return max(0.0, target - width) / spaces

    def _render_segments(self, line, cursor, baseline, extra):
        nodes = []
        for content, style, metrics in line[_SEGMENTS]:
            pieces = re.split(r"( +)", content) if extra else (content,)
            cursor = self._render_pieces(pieces, content, style, metrics, cursor, baseline, extra, nodes)
        return nodes

    def _render_pieces(self, pieces, content, style, metrics, cursor, baseline, extra, nodes):
        for piece in pieces:
            piece_metrics = metrics if piece == content else self.measure(piece, style)
            if piece.strip():
                options = {key: style[key] for key in self.base_options}
                nodes.append(
                    self.text_factory(
                        piece, size=style["size"], ox=cursor, oy=baseline,
                        path=style["path"], fallback_fonts=style[_FALLBACK_FONTS],
                        **options,
                    )
                )
            cursor += piece_metrics[_WIDTH]
            if extra and piece.isspace():
                cursor += extra * len(piece)
        return cursor

    def render(self):

        self.add_spans(line_break_units)
        self.truncate()
        ascent, descent, advance, block_height, measured = self._block_metrics()
        self._validate_overflow(measured)
        baseline = self._first_baseline(ascent, descent, block_height, advance)
        nodes = [
            node
            for index, line in enumerate(self.lines)
            for node in self._render_line(index, line, baseline, advance)
        ]
        result = self._join_nodes(nodes)
        return self._clip(result)

    def _join_nodes(self, nodes):
        if not nodes:
            return self.text_factory(
                "", size=self.common["size"], ox=self.common["ox"],
                oy=self.common["oy"], path=self.common["path"],
            )
        result = nodes[0]
        for node in nodes[1:]:
            result = result.union(node)
        return result

    def _clip(self, result):
        if self.box_width is None or self.common[_OVERFLOW] != "clip":
            return result
        from .core.api import SDFNamespace

        shape = SDFNamespace().rect(
            (self.x + self.box_width / 2, self.y + self.box_height / 2),
            (self.box_width, self.box_height),
        )
        return result.intersection(shape)


def render_inline_text(value, common, text_factory):
    return InlineTextLayout(value, common, text_factory).render()
