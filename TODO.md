# SDF UI Roadmap

This list is organized by feature area and focuses on the gaps that would make the engine feel more complete, more reliable, and easier to use.

## Core Rendering

- Add a scene graph or render graph so repeated subtrees can be cached instead of recomputed every frame.
- Add dirty-region or subframe rendering so unchanged parts of the canvas do not need a full dispatch.
- Make dispatch sizing explicit and configurable per target device instead of deriving only from canvas size.
- Add bounds-aware shaders and clipping so work can be skipped outside the visible region.

## SDF Primitives

- Add half-plane and arc primitives directly, instead of forcing them through composition.
- Add stroke-aware primitives so outlines and path widths can be expressed natively.
- Add polygon support, starting with convex polygons and then general polygons.
- Add signed distance support for capsules, capsules with variable radius, and rounded segments.

## Composition and Booleans

- Add more robust masked blending modes and sharpen the current masked union output.
- Add a dedicated stroke/combine pipeline for layered shape construction.
- Add path-level smoothing controls so joins and transitions can be tuned per shape.
- Add better boolean diagnostics or visualization helpers for debugging complex shape graphs.

## Text and Typography

- Cache glyph contours and glyph SDFs by font, size, and character.
- Add kerning, baseline handling, and multi-glyph layout for real text runs.
- Add line wrapping, alignment, and text box support.
- Add font fallback and missing-glyph handling.
- Add a custom SDF font pipeline so text rendering does not depend on ad hoc contour extraction.

## Color and Shading

- Add more blend modes such as screen, multiply variants, darken, lighten, and soft light.
- Add gradient primitives that are first-class rather than composed from other shapes.
- Add stronger color-space tooling, including explicit sRGB/LAB conversion helpers and previews.
- Add noise variants beyond the current Perlin and film grain shaders.

## Postprocessing

- Add configurable blur kernels and a faster separable blur path.
- Add sharpen, glow, and edge-detect style post effects.
- Add dithering controls for palette quantization workflows.
- Add a proper effect stack so postprocessing can be ordered and reused cleanly.

## Animation

- Add native animation timelines instead of frame-by-frame scripting only.
- Add keyframe interpolation helpers for positions, colors, transforms, and opacity.
- Add time-based shader uniforms and reusable easing functions.
- Add export helpers for frame sequences and video output with less manual plumbing.

## Image Integration

- Add direct image-to-SDF conversion that is more robust than the current rough threshold approach.
- Add image sampling modes, scaling controls, and better aspect-ratio handling.
- Add texture loading helpers for common image formats and color profiles.
- Add masked image fills with clearer controls for alpha, tint, and inflation.

## 3D and Projection

- Add orthographic projection of 3D SDFs.
- Add depth-aware composition if 3D primitives are introduced.
- Add a simple camera model and transform stack for world-space scene building.

## Tooling and UX

- Add hot reload for scripts so artists can iterate without restarting.
- Add a YAML or node-based editor only if it maps cleanly to the shader graph model.
- Add richer example scripts with parameterized presets and side-by-side comparisons.
- Add test coverage for shader dispatch, texture lifetime, and composition correctness.
- Add safer resource lifecycle handling so texture cleanup is easier to reason about.

## API Cleanup

- Normalize naming between shader files, shader constants, and user-facing methods.
- Make return types and context requirements explicit across the API.
- Add stronger validation for texture compatibility before composition.
- Add a small layer of convenience wrappers for common workflows like icons, badges, and glyphs.
