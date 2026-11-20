# SDF UI Tasks

## Contributor setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run
`uv sync` from the repository root. Run `uv run pytest` to run the test suite,
and `uv run pre-commit install` to install the Git hook. Run
`uv run pre-commit run --all-files` to check tracked files. If an older checkout
set `core.hooksPath` to `.githooks`, unset that local setting with
`git config --local --unset core.hooksPath` before installing the hook.

After changing built-in operation metadata, regenerate and commit the core
stubs with `uv run python build_tools/generate_api_stubs.py`. Add `--check` to
check for drift without writing files.

## Completed work

## Graph architecture and rendering

- [x] Move shader image bindings, uniform metadata, and forward/backward knowledge into their owning extensions.
- [x] Add scene and training render graphs with reusable subtree caches.
- [x] Make display dispatch sizing explicit and configurable.
- [x] Add SDF differentiation contracts and a GPU training prototype, including losses over named and intermediate graph outputs.
- [x] Add test coverage for shader dispatch, texture lifetime, and composition correctness.

## Shapes, text, and shading

- [x] Add half-plane and arc primitives, convex and general polygons, capsules, and rounded segments.
- [x] Add scale, translation, and rotation transforms across shape graphs.
- [x] Add segment-width and shape-outline operations.
- [x] Cache glyph contours and glyph SDFs by font, size, and character; add basic kerning and multi-glyph text runs.
- [x] Add fixed-box text wrapping and horizontal/vertical alignment using font ascent and descent metrics.
- [x] Add text measurement, ordered font fallbacks, and configurable missing-glyph behavior.
- [x] Shape text with HarfBuzz OpenType features, script/language/direction runs, variable font axes, and bidirectional ordering; rasterize TrueType and CFF glyph outlines.
- [x] Wrap text at Unicode line-break boundaries and report or clip text-box overflow.
- [x] Add justified text-box alignment, selectable overflow behavior, and trailing-space control.
- [x] Add SDF text decorations using OpenType underline and strike metrics.
- [x] Measure glyph ink bounds and return per-line metrics and paragraph ink bounds.
- [x] Add styled inline text spans with per-run fonts, sizes, shaping settings, wrapping, and alignment.
- [x] Resolve system fonts through Fontconfig or standard platform font directories.
- [x] Add linear and radial gradient APIs using composed shading passes.
- [x] Add RGB/LAB color-space conversion helpers.

## Animation and integration

- [x] Add animated parameters driven by normalized per-frame progress.
- [x] Add GIF and MP4 export helpers that render an animation sweep.
- [x] Add image-to-SDF conversion and GPU texture upload helpers.
- [x] Export numbered PNG frame sequences from progress-driven animation parameters.
- [x] Add reusable cubic and sine easing expressions for animated parameters.
- [x] Add darken and lighten masked blend modes alongside mix, multiply, screen, and over.
- [x] Add built-in/custom palette sampling and a parameterized side-by-side preset gallery.

## Development checklist

The renderer already supports progress-driven parameters and GIF/MP4 export.
Open work is grouped below by rough implementation effort and expected
project value. “Easy” means
a focused API or shader change; “hard” means broader engine or pipeline work.
These are planning estimates, not promises.

The easy, important quadrant has been implemented: named/custom palette
sampling, parameterized preset comparisons, numbered PNG frame export,
reusable progress easing, and darken/lighten masked blends. Existing `mix`,
`multiply`, and `screen` mask modes remain available.

| Effort | Important | Less important |
| --- | --- | --- |
| **Easy** | Reusable visual button/control widget with content-aware text fitting; annotated learning-layout preview export. | First-class XOR composition; icon, badge, and glyph convenience helpers; naming cleanup across shaders and public methods; more showcase art examples. |
| **Hard** | Shader/loss adapter boundaries and learning-graph organization; bounds-aware shaders and subframe rendering; variable-radius capsules and mask-to-SDF conversion; glyph SDF atlas; native timelines and keyframes; soft-light blending; explicit sRGB management; reusable effects; improved image-to-SDF conversion and image controls; common image formats and color profiles; safer texture cleanup. | Lazy-load example catalog entries so an unrelated example's import or optional dependency failure does not prevent other examples from running; custom SDF font pipeline; orthographic 2.5D rendering, depth composition, camera and world transforms; script hot reload; additional noise families; boid/swarm showcase. |
