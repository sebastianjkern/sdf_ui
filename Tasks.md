# SDF UI Tasks

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

## Prioritized feature matrix

Effort and importance are initial estimates. Easy items should fit a focused
API or shader change; hard items touch multiple engine layers or introduce a
new pipeline. The matrix is mirrored in the README Development Checklist.

The easy/important quadrant has started: numbered PNG frame export, progress
easing expressions, and darken/lighten masked blends are implemented above.

| Effort | Important | Less important |
| --- | --- | --- |
| **Easy** | Palette-driven coloring helpers; parameterized example presets and side-by-side comparisons. | First-class XOR composition; icon, badge, and glyph convenience helpers; naming cleanup across shaders and public methods; more showcase art examples. |
| **Hard** | Shader/loss adapter boundaries and learning-graph organization; bounds-aware shaders and subframe rendering; variable-radius capsules and mask-to-SDF conversion; glyph SDF atlas; native timelines and keyframes; soft-light blending; explicit sRGB management; reusable effects; improved image-to-SDF conversion and image controls; common image formats and color profiles; safer texture cleanup. | Custom SDF font pipeline; orthographic 2.5D rendering, depth composition, camera and world transforms; script hot reload; additional noise families; boid/swarm showcase. |
