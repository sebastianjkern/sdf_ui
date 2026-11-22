---
name: sdf-ui-example
description: Create, update, run, or troubleshoot repository examples for sdf-ui. Use for example source placement, CLI entry points, generated output paths, and visual previews.
---

# Build and run sdf-ui examples

## Place files in the checkout

- Put the runnable module at `examples/<name>.py`. Keep it standalone with a module entry point. Use `examples/_<name>.py` for a private helper that is shared or substantial enough to need its own file. Use the `examples/__main__.py` catalog when the example belongs in the shared runner.
- Write generated files under `out/examples/<name>/`. Keep related images, frames, SVG, G-code, and intermediate render artifacts together there. `out/` is ignored by Git.
- Put a selected preview in `examples/previews/` when it needs to travel with the source or be linked from the README. Link other visual inspection files from the generated output directory.

## Make the entry point independent of the caller's directory

- Resolve the default output against the checkout. In a top-level example module use `Path(__file__).resolve().parents[1] / "out" / "examples" / "<name>"`. Create the chosen directory before writing files. Give `--output` a help string that identifies whether it takes a directory or file. Honor explicit absolute paths and resolve explicit relative paths from the process working directory.
- Give randomized artwork a repeatable default seed. Accept `--seed` when a different seed is a meaningful variation.
- Run `uv sync --locked` to prepare the environment. Launch from the checkout with `uv run --locked python -m examples.<name>`. From another directory use `uv --directory /absolute/path/to/sdf_ui run --locked python -m examples.<name>` so Python can import the checkout-local `examples` package.

## Check the deliverables

Run the entry point with its default output and with an explicit override. Exercise the documented invocation from a nested directory and verify the artifact paths. For render changes, inspect the result at its intended size and provide the output image for review. Use the text-rendering skill when lettering is involved.
