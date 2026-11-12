Shader differentiation contracts
=================================

Every registered builtin GLSL pass has an executable differentiation contract.
The contracts cover scalar and RGBA input-image adjoints, uniform-parameter
adjoints reduced over pixels, and multiple outputs. They differentiate the
shader's real arithmetic before ``imageStore`` clipping/quantization.

The contracts include analytic CPU reference rules and native GPU derivative
bodies. The CPU rules are a numerical oracle for the GPU implementation. The
compiled training graph schedules generated GLSL forward passes and their
backward programs; see :doc:`gpu_render_graph` for execution and limitations.

Inspect and execute a contract
------------------------------

::

    import numpy as np
    from sdf_ui.backends.opengl.nodes.registry import registry

    shader = registry.get("circle").shader
    contract = shader.differentiation
    print(contract.inputs, contract.parameters, contract.outputs)
    print(contract.behavior, contract.zero_gradient_parameters, contract.notes)

    gradients = shader.backward(
        size=(64, 48),
        inputs={},
        params={"offset": [32.0, 24.0], "radius": 10.0},
        output_gradients=np.ones((48, 64)),
    )
    print(gradients.parameters["radius"])  # -3072

Names are GLSL uniform names, including image names, rather than plugin API
parameter names. The circle shader calls its center ``offset``; the public
``sdf.circle`` API calls it ``center``. Contracts parse the shader declarations
and validate input/output/uniform shapes before evaluation.

``contract.evaluate(size, inputs, params)`` returns a Tensor for one output or
a tuple in output declaration order. Existing Tensor arguments retain their
graphs, allowing an ordinary combined loss and ``loss.backward()``.

``contract.vjp(...)`` and ``shader.backward(...)`` instead return independent,
read-only ``ShaderGradients.inputs`` and ``ShaderGradients.parameters`` arrays.
They never overwrite caller-owned parameter gradients. Supply an output seed
array with the same shape as the forward output. For ``masked_union``, supply
both arrays as a tuple or a mapping with ``destTex`` and ``maskTex`` keys. Both
seeds contribute to the same source adjoints.

The local VJP recomputes the reference forward expression. It requires no
OpenGL context and currently runs on CPU. It is suitable for gradient checking
and optimization. The native GPU executor consumes the same typed contracts.

Coverage and mathematical choices
---------------------------------

* Primitives: abs, arc, Bezier, capsule, circle, ellipse, grid, half-plane,
  interpolation, intersection, line, masked union, rounded rectangle, repeat,
  ring, sector, smooth union, subtraction, transforms, triangle, and union.
* Shading: clear color, fill, fill from texture, film grain, isolines, lighting,
  outline, partial derivatives, and Perlin noise.
* Layers/postprocessing: overlay, layer mask, multiply, transparency, invert,
  both color conversions, additive dithering, hard one-bit dithering, and all
  four registered horizontal/vertical blur passes.

``min``/``max`` use piecewise derivatives with split adjoints at equal inputs;
``abs`` and vector norms use zero at the origin. Closest-point choices use the
active feature. Bezier curve parameters use the closest-point envelope
derivative, with cubic roots used only to select that feature. Normals and
lighting differentiate through their finite-difference source samples.

Raw shader domain restrictions still apply. Undefined operations such as
division by zero, coincident endpoints in the unguarded line shader, or singular
skews are rejected rather than assigned invented gradients. Nonsmooth feature
switches are not promises of smooth or convex optimization.

Integer addressing gives zero gradients to repeat spacing and blur offsets
away from address changes. Source texels still receive accumulated gradients.
Bilinear transforms differentiate coordinates and scatter source gradients to
all contributing texels. Blur backward uses the transpose of the actual
asymmetric reflected-border stencil; applying the blur again is not equivalent.
Out-of-range image loads use zero, matching the shader image-load convention.

Hard one-bit dithering has zero derivative almost everywhere, undefined at its
thresholds. Additive dithering preserves the original color derivative away from
thresholds. No straight-through estimator is silently substituted. Noise shaders
with fixed pixel coordinates have no effective trainable inputs; film grain's
declared ``time`` uniform is unused and has zero derivative.

Contracts identify these cases through ``behavior`` and
``zero_gradient_parameters``. They are testable descriptions of mathematical
behavior, not claims that every shader is useful for gradient optimization.

Use in the current graph
-------------------------

The reference renderer can execute eligible plugin passes through their shader
contracts. Explicit adapters preserve public parameter names, color-space
conversions, composite blur/shadow operations, and multi-output selection.
Custom GPU render functions require an explicit adapter or differentiable hook.

Run a combined SDF-field, image, and semantic-layout optimization::

    uv run python -m examples.shader_optimization --steps 200

Add ``--gpu`` to render the learned scene through the existing GLSL pipeline.
The output directory contains before/after images, parameter values, and losses.
The same parameter leaves contribute to all loss terms before one update.

Native render infrastructure and next steps
-------------------------------------------

The compiled graph lowers composites into typed passes and distinct output
ports. Parameter expressions resolve against shared leaves for each execution.
Forward shaders use FP32 storage; native backward shaders accumulate pixel
adjoints with atomic scatter and reduce uniform gradients in two stages.
Multiple pixel, field, and geometry loss roots contribute before a single
optimizer update. Programs and bounded forward values are cached separately
from execution tapes, with explicit barriers between dependent GPU operations.

Each ShaderFile exposes a ShaderOperation that supplies forward source, typed
contracts, the CPU reference rule, and GPU backward source to the executors.
All 43 standalone primitive, shading, layer, and postprocessing shader operations
use shared dual-number programs to emit both forward GLSL and backward code.
This representation covers piecewise choices, image sampling, and multi-output
primitives. Composite shapes lower into those operations. The CPU reference
remains independently implemented.
Host-side polygon, text, and glyph callbacks are not standalone shader operations;
materialize them as inputs when using a compiled graph.

For standalone shapes, the authoritative runtime math lives in
``sdf_ui.backends.opengl.shader_sources``. ``ShaderProgram`` emits a primal-only forward shader and a
backward shader from the same typed dual arithmetic
program. The original GLSL files still supply port declarations and serve as
an independent forward conformance reference. When changing a shape, update its
shared program, the GLSL reference and the independent CPU rule, then run the
primitive conformance tests. The representation is a GLSL-based typed program,
not a general compiler that differentiates arbitrary GLSL.
Circle, ring, absolute value, union, intersection, subtraction, and interpolation use
generated reverse-mode VJPs, while other operations currently use seeded local
directional derivatives. The CPU formulas remain independent, so reference and finite-difference tests
can detect formula drift. The backward emitter skips inactive output pixels and
known zero-gradient parameter components. A compiler generating reverse-mode
kernels for the remaining operations and pass fusion could improve throughput further. For example, Slang supports generated and custom derivatives
(`Slang autodiff documentation
<https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/07-autodiff.html>`_).
A compiler integration would require a separate compatibility decision. Opaque
Python render callbacks still need explicit graph lowering and derivative rules.
