---
name: sdf-ui-engine-feature
description: Implement or review sdf-ui engine features, including low-level SDF, color, and field operations and high-level widgets, plots, and effects. Use when deciding source-tree placement, public API shape, graph integration, rendering behavior, or feature acceptance criteria.
---

# Implement an sdf-ui engine feature

## Start with the component layer

Trace a similar existing feature from its public entry point through its node,
plugin, shader, graph lowering, renderer, stubs, and tests. Keep the public scene
API in `src/sdf_ui/core`, `src/sdf_ui/plot.py`, or `src/sdf_ui/widgets/` as
appropriate. Put reusable GPU operations in `src/sdf_ui/gpu/nodes/` under the
matching primitive, shading, layer, or postprocessing family. Keep learning
execution in the shared graph and differentiation systems instead of adding a
parallel feature-specific trainer.

Classify the feature before coding:

- A geometric primitive returns a distance or field and should compose with
  existing SDF operations.
- A shading or color operation changes a color/field result while preserving
  its declared color-space and alpha behavior.
- A layer combines inputs and must define order, ownership, and output mode.
- A postprocessing operation that samples neighbors must declare its sampling
  behavior and any required halo or full-frame constraint.
- A widget or plot is a reusable composition of existing scene nodes where
  possible. Put its layout and coordinate mapping in the high-level module and
  keep its child nodes available to the shared render and learning graphs.

Do not make a host callback the only implementation for a feature expected to
work in compiled graphs. Follow the existing plugin registration and typed
shader interface for built-in GPU operations. Update public `.pyi` files where
the runtime API is exposed through stubs.

## Parameter contract

Every meaningful continuous public component parameter must accept the
repository's lazy `Expr`/`param(...)` values and remain connected to the current
parameter leaves at each execution. Do not evaluate a lazy expression during
construction or retain an eager `Tensor` derived from trainable values in a
persistent scene. Keep structural choices such as shader selection, array
length, and graph topology explicit compile-time controls when the engine
requires them.

Keep literal inputs convenient. Validate finite values, shapes, units, and
degenerate geometry at construction or execution boundaries. For bounded or
positive learnable values, use the learning API's smooth transforms or existing
bounded scalar declarations. Preserve literal behavior and default rendering.

## Acceptance criteria

- The feature has a clear public location and follows a nearby API's naming,
  argument conventions, and composition behavior.
- Literal values and lazy parameter expressions both work through the normal
  public entry point.
- Inputs, outputs, color mode, alpha, dimensions, coordinate conventions, and
  degenerate cases have defined behavior.
- The feature lowers through the shared graph when it is a built-in operation,
  preserves shared subexpressions, and exposes stable parameter names.
- Runtime and type stubs agree.
- Tests cover public API construction, expected output, parameter updates after
  graph construction, and interactions with adjacent operations.
- Repeated rendering does not cause unbounded cache entries, image allocations,
  or leaked GPU resources. See the shader lifecycle skill for detailed checks.

For a widget, also verify its geometry against the full box and content box,
small or zero-sized dimensions, clipping behavior, and label placement. For a
plot, verify data-to-canvas transforms at both range endpoints and behavior
outside the range or with degenerate ranges.

## Review evidence

Run focused checks for the touched entry points and graph execution paths. Inspect
one rendered result at representative dimensions when pixel geometry, text, or
color is involved. Report known limitations such as fixed sampling topology or
zero-gradient discrete control so callers can make informed choices.
