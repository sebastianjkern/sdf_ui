"""Token-aware primal GLSL extraction helpers."""

import re


def _primal_only(source):
    """Erase dual tangent expressions from a generated forward program.

    Forward execution never observes a tangent. Replacing each dual
    constructor's second argument and each tangent read allows the GLSL
    compiler to remove the remaining dead derivative-only branches.
    """
    items = _tokenize(source)
    source = _replace_ranges(source, _constructor_tangent_ranges(items))
    return _replace_ranges(source, _tangent_read_ranges(_tokenize(source)))


_TOKEN_PATTERN = re.compile(
    r"(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|(?P<space>\s+)|"
    r"(?P<name>[A-Za-z_]\w*)|(?P<other>.)"
)
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _tokenize(source):
    return [
        (match.group(), match.start(), match.end())
        for match in _TOKEN_PATTERN.finditer(source)
        if match.lastgroup not in {"comment", "space"}
    ]


def _replace_ranges(source, ranges):
    selected = []
    for start, end in sorted(ranges):
        if selected and start < selected[-1][1]:
            if end > selected[-1][1]:
                raise ValueError("Overlapping shader expression rewrites")
            continue
        selected.append((start, end))
    for start, end in reversed(selected):
        source = source[:start] + "0.0" + source[end:]
    return source


def _constructor_tangent_ranges(items):
    ranges = []
    for index, (name, _, _) in enumerate(items):
        if name == "D" and index + 1 < len(items) and items[index + 1][0] == "(":
            ranges.append(_constructor_tangent_range(items, index))
    return ranges


def _constructor_tangent_range(items, index):
    return _scan_constructor_arguments(items[index + 2 :])


def _scan_constructor_arguments(items):
    depth, comma = 1, None
    for token, start, end in items:
        match token:
            case "(":
                depth += 1
            case ")":
                depth -= 1
                if depth == 0:
                    return _finish_constructor_range(comma, start)
            case "," if depth == 1:
                comma = _record_constructor_comma(comma, end)
    raise ValueError("Invalid dual constructor in shared shader program")


def _record_constructor_comma(comma, end):
    if comma is not None:
        raise ValueError("Invalid dual constructor in shared shader program")
    return end


def _finish_constructor_range(comma, start):
    if comma is None:
        raise ValueError("Invalid dual constructor in shared shader program")
    return comma, start


def _tangent_read_ranges(items):
    ranges = []
    index = 0
    while index < len(items):
        item = _tangent_read_at(items, index)
        if item is None:
            index += 1
            continue
        index, value_range = item
        if value_range is not None:
            ranges.append(value_range)
    return ranges


def _tangent_read_at(items, index):
    token, start, _ = items[index]
    if not _IDENTIFIER.fullmatch(token) or (index > 0 and items[index - 1][0] == "."):
        return None
    cursor = index + 1
    while cursor < len(items):
        if items[cursor][0] == "[":
            cursor = _skip_array_index(items, cursor)
            continue
        if not _is_member_access(items, cursor):
            break
        field = items[cursor + 1][0]
        cursor += 2
        if field == "d":
            return cursor, (start, items[cursor - 1][2])
    return max(index + 1, cursor), None


def _skip_array_index(items, cursor):
    depth = 1
    cursor += 1
    while cursor < len(items) and depth:
        depth += (items[cursor][0] == "[") - (items[cursor][0] == "]")
        cursor += 1
    if depth:
        raise ValueError("Unclosed shader array index")
    return cursor


def _is_member_access(items, cursor):
    return (
        items[cursor][0] == "."
        and cursor + 1 < len(items)
        and _IDENTIFIER.fullmatch(items[cursor + 1][0])
    )
