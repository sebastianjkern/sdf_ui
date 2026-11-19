---
name: sdf-ui-shader-lifecycle
description: Implement or review sdf-ui GPU shaders and render passes with correct parameter gradients, cache invalidation, texture ownership, and bounded memory use. Use when editing GLSL/WGSL, plugin render functions, graph passes, or render-session caching.
---

# Implement GPU shaders with correct lifecycle behavior

## Follow the shared shader path

Inspect a neighboring operation in the same family. Keep shader source,
`ShaderInterface`, differentiation contract, plugin registration, CPU reference,
and graph lowering in agreement. Shader image bindings and uniform names must
match the declared interface. If a shader samples neighboring pixels, declare
that sampling behavior so dirty rectangles and graph execution can apply the
right constraints.

For every uniform, decide whether it is a differentiable value or a fixed
control. Continuous uniforms need a reference derivative and native GPU
gradient path. Discrete values such as integer steps or mode selectors need a
documented zero-gradient or rebuild rule. Check the rendered math before
quantization when the training contract differentiates floating-point values.

## Resource ownership and cache rules

Identify who owns every output texture and every temporary buffer. A renderer
cache owns cached node outputs for the renderer/session lifetime. A shader must
not close an input or a cached output. Close newly allocated outputs if dispatch
fails. Release scratch textures in `finally` blocks after their last GPU use.
Respect command-batch deferred destruction rules so a resource is not destroyed
before queued work consumes it.

Uncached intermediate scene nodes may be recycled after their final graph edge
has run. Preserve intermediates that are cache checkpoints, returned outputs,
or still referenced by another branch. When batching passes, compute which
intermediate stages are externally requested and retain only those outputs.
Do not introduce a cache entry per frame or per continuously changing value
unless it is bounded and has an eviction or invalidation policy.

Keep cache identities correct:

- A shader pipeline cache must include the full source and entry point, along
  with any backend state that affects compilation.
- Uniform buffers reused across executions must receive updated payloads when
  values change. During command batching, buffers with distinct payloads must
  not alias mutable contents before submitted work has consumed them.
- Render-node cache keys must account for scene structure, inputs, parameter
  versions/values, dimensions, and relevant execution modes. A structural graph
  fingerprint alone is not a rendered-result key.
- A cached texture must remain open and belong to the expected context and
  dimensions before it is returned.

Prefer the existing context and renderer cache mechanisms. Do not add a second
long-lived resource cache without showing why the existing session-owned
mechanisms cannot serve the operation.

## Memory and repeated-render acceptance criteria

- New outputs are closed on every exception path after allocation.
- Scratch resources are released after last use, including when a later pass
  fails.
- Cached outputs remain valid until their owning renderer/graph closes or clears
  the cache; callers do not accidentally own or destroy those cached resources.
- Parameter changes update uniform data and invalidate only results that depend
  on those parameters.
- Repeated rendering of an unchanged scene reaches cache hits and stable cache
  entry/allocation counts after warm-up.
- Repeated rendering with changing parameters or animation does not grow cache
  entries or GPU image allocations without bound.
- Batched and unbatched execution produce equivalent outputs, including when
  intermediate outputs are requested or shared by multiple branches.
- Failure during allocation, dispatch, or output wrapping leaves no orphaned
  output textures and the renderer remains usable when recovery is supported.
- GPU resource destruction is ordered after queued commands that reference the
  resource.

Add regression coverage for cache reuse, cache invalidation, exception cleanup,
and a multi-frame or long-chain case. Check both cache counters and image
allocation counts where available. Compare an optimized or cropped path against
a full render for equivalent pixels. Run shader differentiation checks for
every continuous uniform and verify that gradients remain finite near branch
boundaries and degenerate geometry.

## Common failure patterns to check

- Closing a cached child texture while recycling an uncached sibling.
- Retaining every uncached intermediate in a long overlay chain.
- Reusing a mutable uniform buffer for two payloads in one command batch.
- Caching by scene topology while omitting parameter versions from execution
  results.
- Baking trainable values into shader source, which creates pipeline variants
  and prevents ordinary uniform updates.
- Producing different CPU and shader branch/tie behavior, yielding misleading
  gradient tests.
- Reading images back in a training loop when GPU losses can keep pixels on the
  device.
