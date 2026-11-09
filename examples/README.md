# Examples

Run examples from the repository root after `uv sync`:

```bash
uv run python -m examples --help
uv run python -m examples render_api
uv run python -m examples.learning_ui --mode all --steps 250
uv run python -m examples.learned_dashboard --steps 80
uv run python -m examples.gpu_graph --steps 150
```

The first command lists the original rendering examples. The learning and GPU
examples have their own command-line options; append `--help` for details.
Generated images and checkpoints go under `out/`.
