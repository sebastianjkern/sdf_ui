---
name: sdf-ui-ast-lint-rule
description: "Implement or review a custom AST-based linting check for sdf_ui (anything ruff/mypy/complexipy cannot see: layer rules, lazy-parameter contracts, cache growth, API/stub drift). Use when adding, changing, or wiring a new quality rule into build_tools/lint/, the pre-commit hooks, or CI."
---

# Add an AST linting rule for sdf_ui

Custom checks live in `build_tools/lint/` (one folder for all custom lint
rules, so they are not scattered beside non-linting build helpers), each as a
standalone stdlib-only script (`build_tools/lint/arch_lint.py` is the
reference: package-root detection, per-file AST walk, rule table, exit code 1
on violations). Wire each script into both `.pre-commit-config.yaml` (local
hook) and `.github/workflows/ci.yml` (same command), because pre-commit alone
does not gate pushes and PRs.

## Before writing any rule

1. Read the nearest existing script in `build_tools/lint/` and its hook
   entries in `.pre-commit-config.yaml` to copy the wiring style, and the
   corresponding CI step in `.github/workflows/ci.yml`.
2. Extract ground truth from the current tree first, with `ast` — never with
   regex. For import rules: resolve relative imports, skip `TYPE_CHECKING`
   blocks, and exempt `__init__.py` re-exports. For call- or attribute-based
   rules: dump the real call sites before deciding thresholds.
3. Derive the rule from what the code actually does and from the architecture
   skills (engine-feature, learning-support, shader-lifecycle), not from an
   idealized layering. A rule that fails on today's tree is either wrong or
   needs an explicit, documented justification before it lands — do not ship
   it red.
4. Keep false positives near zero. A vacuous or noisy check gets deleted or
   muted by the next person who trips on it, which silently undoes the rule.

## Implementing the script

- One script per concern; plain `ast` + `pathlib`, no dependencies. Add a new
  script to `build_tools/lint/`, never to `build_tools/` itself.
- Type-annotate everything; mypy runs on `build_tools/` in CI, which covers
  the `build_tools/lint/` subfolder.
- Module-scope *and* function-scope (deferred) imports both count for import
  rules — deferred imports are exactly where a bad dependency hides.
- Make the package root discoverable from the script location
  (`Path(__file__).resolve().parents[2]` for scripts in `build_tools/lint/`),
  never from the current directory. Adjust the index if the folder depth
  changes.
- Print file, rule, module, and line number per violation; exit 1 only on
  violations; print a short success line otherwise.

## Prove it detects (mandatory)

An early version of the first such rule (the layer lint) reported "no
violations" because a path-parsing bug skipped every file. Never trust a clean
pass alone. Before wiring the hook:

1. Plant violations in a scratch tree (or /tmp fixture) and assert the rule
   fires, including the sneaky forms: deferred imports inside function bodies,
   relative imports, `from sdf_ui import plot` alias forms, nested
   `TYPE_CHECKING` blocks.
2. Assert allowed edges and exemptions stay clean.
3. Only then run the rule against the real `src/sdf_ui` tree. Discrepancies
   mean either the rule needs updating or the code does — decide explicitly.

## Gates the script itself must pass

- `uv run --locked python -m ruff format build_tools/` then `ruff check`
- `uv run --locked python -m mypy build_tools`
- slop duplicate-string limit (4): share repeated layer/segment literals as
  top-level frozenset constants (see `_HELPERS`, `_LOW_LAYERS` in
  `build_tools/lint/arch_lint.py`); docstring wording counts toward
  duplicate-string limits, so rephrase repeated prose.
- complexipy: max cyclomatic complexity 15; split walkers so no function
  exceeds it.

Known trap in this repo: the pre-commit `slop` hook entry may pass an outdated
`--output quiet` flag; the installed slop only accepts `human` or `json`. Run
`uv run --locked slop lint --root <path> --output human` locally. Never fix
unrelated failing hooks while wiring a new one; note them in the PR instead.

## Wiring

In `.pre-commit-config.yaml`, add a `repo: local` hook mirroring `api-stubs`:

```yaml
      - id: arch-lint
        name: architecture layer lint
        entry: uv run --locked python build_tools/lint/arch_lint.py
        language: system
        pass_filenames: false
        files: ^(src/)
```

Scope `files:` to the area the rule reads (`src/` unless the rule is about
`tests/` or `build_tools/` itself), so unrelated commits do not pay for the
scan. Add the same command as a CI step next to the closest existing one
(existing custom-lint examples: `arch-lint`, `comment-lint`, and
`gpu-lifecycle-lint`, all pointing at `build_tools/lint/<script>.py`).

## Closing

Run the full local gate set once, report what passed and what could not be
verified, and end with the command a reviewer can rerun:

    uv run --locked python build_tools/lint/<script>.py

Consult the engine-feature skill for which architectural constraints are
load-bearing (layer graph, deferred-import escape hatches), and the
shader-lifecycle skill for bounding caches and resources before writing a
rule that inspects long-lived state.
