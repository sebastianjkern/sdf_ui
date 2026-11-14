# Examples

Run examples from the repository root after `uv sync`:

```bash
uv run python -m examples --help
uv run python -m examples render_api
uv run python -m examples.learning_ui --mode all --steps 250
uv run python -m examples.learned_dashboard --steps 80
uv run python -m examples.gpu_graph --steps 150
uv run python -m examples.center_div --steps 60
```

The first command lists the original rendering examples. The learning and GPU
examples have their own command-line options; append `--help` for details.
Generated images and checkpoints go under `out/`.

``center_div`` exports one PNG per optimization step and assembles them into
``out/center_div/center_div.gif``. The individual frames are kept alongside
the animation for inspection or reuse by future animation tooling.
