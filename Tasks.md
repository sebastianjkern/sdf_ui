# Completed SDF UI Tasks

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
- [x] Add linear and radial gradient APIs using composed shading passes.
- [x] Add RGB/LAB color-space conversion helpers.

## Animation and integration

- [x] Add animated parameters driven by normalized per-frame progress.
- [x] Add GIF and MP4 export helpers that render an animation sweep.
- [x] Add image-to-SDF conversion and GPU texture upload helpers.
