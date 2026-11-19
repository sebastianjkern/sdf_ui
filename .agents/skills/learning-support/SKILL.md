---
name: sdf-ui-learning-support
description: Add or review first-class learning API support for sdf-ui parameters in primitives, shaders, plots, widgets, layouts, and effects. Use whenever a feature exposes values that users may want to optimize or animate.
---

# Give components first-class learning support

The learning API is a property of the full component path. A parameter is not
first-class merely because its Python constructor accepts a `Parameter`.
It must resolve from the current leaves, reach the correct shader uniform or
composed expression, contribute gradients through the documented training path,
and behave correctly when cached executions are reused.

## Trace each parameter end to end

For every public numeric input, record whether it is continuous, bounded,
discrete, structural, or derived. Trace a continuous parameter through:

1. The public function or component and its type stub.
2. Lazy expression storage in the scene node, without eager resolution.
3. Graph lowering and the operation's named uniform or composed child nodes.
4. CPU reference evaluation and GPU forward execution.
5. Backward propagation to the parameter leaf, including reductions over pixels.
6. Re-execution after optimizer updates, checkpointing where applicable, and
   cache dependency tracking.

Use `Expr` and `param(...)` for values that must stay live in a persistent
scene. Use `ParameterStore.bind` or `ParameterStore.scalar` when the feature's
usage benefits from named weights and smooth constraints. A derived eager
`Tensor` stored in a scene can become stale after an update. Keep derived
operations lazy or resolve them afresh during each execution.

At the high level, preserve parameter flow through wrappers. `Panel`,
`progress_bar`, `Plot` annotations, and future widgets should pass expressions
into child nodes or expose their trainable layout values directly. Avoid
converting a parameter to `float`, a NumPy scalar, or a fixed constructor value
before graph lowering.

## Gradients and discrete choices

For shader-backed operations, use the shared `ShaderDifferentiation` contract.
Define uniform names and shapes once, keep the CPU reference aligned with the
shader, and implement or register the GPU derivative path required by the
native learning graph. Declare intentionally zero-gradient parameters and
explain the branch convention in the contract. Treat integer indices, topology,
sample counts, and branch selectors as structural controls when their gradient
is undefined or the graph requires recompilation.

For composed high-level features, prefer existing differentiable nodes over
host-side rasterization. Text can use a pre-rendered static glyph SDF while
keeping placement, scale, tint, and leader geometry differentiable when the
underlying operation supports them. Be explicit about which text-shaping inputs
remain static.

## Acceptance criteria

- Each continuous public parameter accepts literals and lazy expressions.
- A parameter changed after scene construction changes the next execution
  without retaining stale eager expressions.
- Parameter names and uniform mappings are stable through lowering and shared
  outputs do not duplicate or detach their leaves.
- CPU reference gradients are finite and agree with finite differences within
  a stated tolerance on smooth branches.
- Native GPU training propagates nonzero gradients for representative
  parameters, including values wrapped by high-level components.
- A representative loss decreases under a small optimizer run when the chosen
  problem is expected to be learnable.
- Discrete, fixed, bounded, and unsupported parameters have explicit semantics.
- Training does not read pixels back to the CPU unless a Python custom loss
  explicitly requires those pixels.
- Cache reuse after parameter changes produces the same result and gradients as
  a fresh execution.
- Checkpoint schema/value behavior remains consistent when the component adds
  parameters to a training model.

Test at least one scalar geometry or field loss and one rendered pixel loss
where the feature affects both geometry and appearance. Add a composed scene
case for any high-level wrapper. Verify that a graph can be compiled once and
run across multiple optimizer updates.
