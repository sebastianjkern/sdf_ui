# Development process

## Code Quality Rules

- No comments, except ruff hints

## Development hints

Always do root cause analysis. Find out where the issue originates from, not how to later circumnavigate it.

Run the static checks from the precommit hooks repeatedly to check code quality regressions directly.

Run tests only after the implementation/bug fixes etc. is done. 

If you encounter preventable bugs, that could have been caught by a linter, but are not yet covered by the existing linters, do not only write regressions tests, but also use the /sdf-ui-ast-lint-rule skill to implement a new simple custom linting rule capturing the problem, and potential similar problems. Commit and push these changes in a focused commit directly.

If you write render code always output visual inspection images in form of examples, or focused renders to let me inspect the state of the problems.

If you encounter potential side issues, document them in Tasks.md directly evaluating the problem difficulty and directly noting what the root cause is, if known, and files that are part of this issue, if known.
