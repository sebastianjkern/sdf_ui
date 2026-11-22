# Development process

## Code Quality Rules

- No comments, except ruff hints
- Skip standalone documentation. Short docstrings are fine.

## Development hints

Always do root cause analysis. Find out where the issue originates from, not how to later circumnavigate it.

Run the static checks from the precommit hooks repeatedly to check code quality regressions directly.

Run tests only after the implementation/bug fixes etc. is done. 

If you encounter preventable bugs, that could have been caught by a linter, but are not yet covered by the existing linters, do not only write regressions tests, but also use the /sdf-ui-ast-lint-rule skill to implement a new simple custom linting rule capturing the problem, and potential similar problems. Commit and push these changes in a focused commit directly.

If you write render code always output visual inspection images in form of examples, or focused renders to let me inspect the state of the problems.

Log potential side issues in Tasks.md with their difficulty, known root cause, and affected files.

## Implementation

Always implement features in gpt6 luna subagents set to medium to keep your context tidy. Make sure you check for newly occuring bugs, and check if linting rules using the /sdf-ui-ast-lint-rule skill could have helped flag the error and similar ones directly.
