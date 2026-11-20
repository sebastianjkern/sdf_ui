"""Forbid comments in Python source code.

The rule is unconditional: every ``COMMENT`` token found by :mod:`tokenize`
is a violation, regardless of its content, position (stand-alone line or
trailing), or any marker prefix such as ``TODO``. The lexer-accurate token
stream also means a ``#`` inside a string or docstring is never flagged.

Exempted are functional linter / type-checker directive comments whose
removal would change tool behavior: ``# noqa`` suppressions, Ruff per-file
directives (``# ruff: ...``), and ``# type: ...`` mypy directives and
comment-style annotations.

The check runs on ``src/sdf_ui`` only, so unrelated commits do not pay for
the scan. It is wired as a local pre-commit hook and as a CI step in
``.github/workflows/ci.yml``.
"""

import io
import re
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

# Package root discovery from script location (never from cwd).
_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "sdf_ui"

_RULE = "comment"

# Directive comments are exempt: the leading "#" is already stripped by the
# caller before matching, so the match starts at the directive keyword.
_DIRECTIVE_RE = re.compile(r"^(?:noqa\b|type\s*:|ruff\s*:)")


def _is_exempt(comment_text: str) -> bool:
    """Return True for directive comments (noqa, ruff, type checker)."""
    body = comment_text.lstrip("#").lstrip()
    return bool(_DIRECTIVE_RE.match(body))


def _find_comments_in_file(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, comment_text)`` for every non-exempt comment.

    Line numbers are 1-based. Directive comments (see ``_DIRECTIVE_RE``)
    never yield. Read errors and tokenize failures are reported to stderr
    and swallowed so one unreadable file cannot abort the scan.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"comment lint: read error for {path}: {exc}", file=sys.stderr)
        return
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT and not _is_exempt(tok.string):
                yield tok.start[0], tok.string
    except (tokenize.TokenError, IndentationError) as exc:
        print(f"comment lint: tokenize error for {path}: {exc}", file=sys.stderr)
        return


def lint_file(path: Path) -> list[str]:
    """Return one violation string per comment found in *path*."""
    return [
        f"{path}:{lineno}: [{_RULE}] {text}"
        for lineno, text in _find_comments_in_file(path)
    ]


def lint_tree(root: Path) -> Iterator[tuple[Path, list[str]]]:
    """Lint every Python module below *root* and yield ``(file, violations)``."""
    for path in sorted(root.rglob("*.py")):
        problems = lint_file(path)
        if problems:
            yield path, problems


def main() -> int:
    root = _SRC_ROOT
    if not root.is_dir():
        print(f"comment lint: package root not found: {root}", file=sys.stderr)
        return 1
    violations = list(lint_tree(root))
    if not violations:
        print("comment lint: no comments found")
        return 0
    total = sum(len(problems) for _, problems in violations)
    for path, problems in violations:
        print(f"{path}")
        for problem in problems:
            print(f"  [!] {problem}")
    print(f"\ncomment lint: {total} violation(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
