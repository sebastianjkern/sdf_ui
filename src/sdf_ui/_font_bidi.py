"""Unicode bidirectional and script run segmentation for text shaping."""

from __future__ import annotations

from bidi import algorithm as bidi_algorithm
from fontTools.unicodedata import script as unicode_script


def isolate_tokens(value):
    tokens = []
    index = 0
    while index < len(value):
        char = value[index]
        if ord(char) in {0x2066, 0x2067, 0x2068}:
            end = _isolate_end(value, index)
            if end is not None:
                inner = value[index + 1 : end - 1]
                tokens.append(("isolate", inner, _isolate_base_level(char, inner)))
                index = end
                continue
            index += 1
            continue
        if ord(char) != 0x2069:
            tokens.append(("char", char, None))
        index += 1
    return tokens


def _isolate_end(value, start):
    depth = 1
    for end in range(start + 1, len(value)):
        code = ord(value[end])
        if code in {0x2066, 0x2067, 0x2068}:
            depth += 1
        elif code == 0x2069:
            depth -= 1
            if depth == 0:
                return end + 1
    return None


def _isolate_base_level(char, inner):
    isolate_type = ord(char)
    if isolate_type == 0x2067:
        return 1
    if isolate_type == 0x2066:
        return 0
    return bidi_algorithm.get_base_level(inner)


def bidi_runs(value, direction, base_level=None):
    """Return visual-order logical text runs with resolved bidi directions."""
    if direction in {"ltr", "rtl"}:
        return [(value, direction)]
    tokens = isolate_tokens(value)
    bidi_text = "".join(
        token[1] if token[0] == "char" else "א" if token[2] else "A" for token in tokens
    )
    storage = bidi_algorithm.get_empty_storage()
    if base_level is None:
        base_level = bidi_algorithm.get_base_level(bidi_text)
    storage["base_level"] = base_level
    storage["base_dir"] = ("L", "R")[base_level]
    bidi_algorithm.get_embedding_levels(bidi_text, storage)
    for index, char in enumerate(storage["chars"]):
        char["_source_index"] = index
    bidi_algorithm.explicit_embed_and_overrides(storage)
    bidi_algorithm.resolve_weak_types(storage)
    bidi_algorithm.resolve_neutral_types(storage, False)
    bidi_algorithm.resolve_implicit_levels(storage, False)
    bidi_algorithm.reorder_resolved_levels(storage, False)

    runs = []
    current = []
    level = None

    def flush():
        if current:
            indexes = sorted(item["_source_index"] for item in current)
            run_text = "".join(tokens[index][1] for index in indexes)
            runs.append((run_text, "rtl" if level % 2 else "ltr"))
            current.clear()

    for item in storage["chars"]:
        level = _consume_bidi_item(item, tokens, current, level, runs, flush)
    flush()
    return runs or [(value, "ltr")]


def _consume_bidi_item(item, tokens, current, level, runs, flush):
    token = tokens[item["_source_index"]]
    if token[0] != "isolate":
        item_level = item["level"]
        if current and item_level != level:
            flush()
        current.append(item)
        return item_level
    flush()
    runs.extend(bidi_runs(token[1], None, base_level=token[2]))
    return None


def script_runs(value, forced_script=None):
    if forced_script:
        return [(value, forced_script)]
    scripts = [unicode_script(char) for char in value]
    for index, script in enumerate(scripts):
        if script not in {"Zyyy", "Zinh", "Zzzz"}:
            continue
        neighbor = next(
            (item for item in scripts[index + 1 :] if item not in {"Zyyy", "Zinh", "Zzzz"}),
            None,
        )
        previous = next(
            (item for item in reversed(scripts[:index]) if item not in {"Zyyy", "Zinh", "Zzzz"}),
            None,
        )
        scripts[index] = previous or neighbor or "Zyyy"
    runs = []
    for char, script in zip(value, scripts):
        if runs and runs[-1][0] == script:
            runs[-1] = (script, runs[-1][1] + char)
        else:
            runs.append((script, char))
    return [(text, script) for script, text in runs]
