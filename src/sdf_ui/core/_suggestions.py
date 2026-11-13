"""Helpers for actionable errors in dynamically exposed API names."""

from difflib import get_close_matches
from typing import Iterable


def unknown_name_message(subject: str, name: str, candidates: Iterable[str]) -> str:
    """Format an unknown-name message with close matches when available."""
    suggestions = get_close_matches(name, tuple(candidates), n=3, cutoff=0.55)
    message = f"{subject} '{name}'."
    if suggestions:
        message += " Did you mean " + ", ".join(
            repr(suggestion) for suggestion in suggestions
        ) + "?"
    return message
