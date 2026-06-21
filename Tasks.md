# SDF UI Tasks

This list focuses on the improvements that best match `sdf_ui`'s strengths as an SDF-first, plugin-driven renderer.

## Performance and Core Rendering

- [ ] Add dirty-region or subframe rendering so unchanged parts of the canvas do not need a full dispatch.
- [ ] Add a scene graph or render graph so repeated subtrees can be cached instead of recomputed every frame.
- [ ] Add bounds-aware shaders and clipping so work can be skipped outside the visible region.
- [ ] Make dispatch sizing explicit and configurable per target device instead of deriving only from canvas size.

## Procedural Fields and Patterns

- [ ] Add first-class Voronoi SDF support for texture-like procedural scenes.
- [ ] Add vector field and flow field primitives for distortion, motion, and pattern control.
- [ ] Add domain warping, turbulence, and contour-map style field operators.
- [ ] Add more field-debugging views so distance values, seams, and boolean transitions are easier to inspect.

## Shape Composition and Modeling

- [ ] Add half-plane and arc primitives directly instead of forcing them through composition.
- [ ] Add polygon support, starting with convex polygons and then general polygons.
- [ ] Add stroke-aware primitives so outlines and path widths can be expressed natively.
- [ ] Add signed distance support for capsules, variable-radius capsules, and rounded segments.
- [ ] Add first-class transforms for scale, translation, and rotation across shape graphs.
- [ ] Add a dedicated stroke/combine pipeline for layered shape construction.
- [ ] Add stronger masked blending modes and improve the current masked-union output.
- [ ] Add path-level smoothing controls so joins and transitions can be tuned per shape.
- [ ] Add mask-to-SDF conversion utilities for turning layered masks back into distance fields.
- [ ] Explore SDF differentiation utilities for field analysis and effect construction.

## Text and Typography

- [ ] Cache glyph contours and glyph SDFs by font, size, and character.
- [ ] Add kerning, baseline handling, and multi-glyph layout for real text runs.
- [ ] Add line wrapping, alignment, and text box support.
- [ ] Add font fallback and missing-glyph handling.
- [ ] Add a custom SDF font pipeline so text rendering does not depend on ad hoc contour extraction.

## Animation and Motion

- [ ] Add native animation timelines instead of frame-by-frame scripting only.
- [ ] Add keyframe interpolation helpers for positions, colors, transforms, and opacity.
- [ ] Add time-based shader uniforms and reusable easing functions.
- [ ] Add export helpers for frame sequences and video output with less manual plumbing.

## Shading and Effects

- [ ] Add more blend modes such as screen, darken, lighten, soft light, and richer multiply variants.
- [ ] Add first-class gradient primitives instead of relying only on composed shading passes.
- [ ] Add stronger color-space tooling, including explicit sRGB/LAB conversion helpers and previews.
- [ ] Add more noise variants beyond the current Perlin and film grain shaders.
- [ ] Add glow, bevel, inner shadow, sharpen, and edge-detect post effects.
- [ ] Add a reusable effect stack so postprocessing chains can be ordered, named, and shared cleanly.
- [ ] Add XOR as a first-class layer or boolean operation.

## Image and Texture Integration

- [ ] Add more robust image-to-SDF conversion beyond the current rough threshold approach.
- [ ] Add image sampling modes, scaling controls, and better aspect-ratio handling.
- [ ] Add clearer masked image fills with direct alpha, tint, and inflation controls.
- [ ] Add texture loading helpers for common image formats and color profiles.

## 2.5D and Projection

- [ ] Add orthographic projection of 3D SDFs.
- [ ] Add height-to-normal and emboss-style shading for 2.5D looks.
- [ ] Add depth-aware composition if 3D primitives are introduced.
- [ ] Add a simple camera model and transform stack for world-space scene building.

## Tooling and UX

- [ ] Add hot reload for scripts so artists can iterate without restarting.
- [ ] Add richer example scripts with parameterized presets and side-by-side comparisons.
- [ ] Add convenience wrappers for common workflows like icons, badges, and glyphs.
- [ ] Add a higher-level coloring API for palette-driven scene construction.
- [ ] Add test coverage for shader dispatch, texture lifetime, and composition correctness.
- [ ] Add safer resource lifecycle handling so texture cleanup is easier to reason about.
- [ ] Normalize naming between shader files, shader constants, and user-facing methods.

## Generative Art Experiments

- [ ] Add showcase examples aimed at generative art workflows, not just UI-style compositions.
- [ ] Explore boid-based scene generation and field-driven swarm behaviors as a demo workload.
