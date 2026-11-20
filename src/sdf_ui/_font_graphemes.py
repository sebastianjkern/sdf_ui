"""Cluster Unicode text into fallback-safe graphemes."""

from __future__ import annotations

import unicodedata
from functools import lru_cache


@lru_cache(maxsize=2048)
def graphemes(value):
    clusters = []
    current = ""
    regional_count = 0
    join_next = False
    for char in value:
        is_regional, extender, code = _properties(char)
        if _starts_cluster(current, extender, join_next, is_regional, regional_count):
            clusters.append(current)
            current = ""
            regional_count = 0
        current += char
        regional_count = _regional_count(regional_count, is_regional, extender)
        join_next = code == 0x200D
    if current:
        clusters.append(current)
    return tuple(clusters)


def _properties(char):
    code = ord(char)
    variation = is_variation_selector(char)
    regional = 0x1F1E6 <= code <= 0x1F1FF
    extender = unicodedata.category(char) in {"Mn", "Mc", "Me"} or variation or code == 0x200D
    return regional, extender, code


def _starts_cluster(current, extender, join_next, regional, regional_count):
    return bool(
        current
        and not extender
        and not join_next
        and not (regional and regional_count % 2 == 1)
    )


def _regional_count(count, regional, extender):
    if regional:
        return count + 1
    return 0 if not extender else count


def is_variation_selector(char):
    code = ord(char)
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF
