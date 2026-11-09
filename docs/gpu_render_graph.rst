Native GPU render graphs
========================

``Canvas.compile`` is the native training entry point. It accepts the
existing SDF/color nodes and lowers them to a reusable graph with named outputs,
typed shader ports, shared subexpressions, and stable parameter leaves.

The full raster path executes on the GPU: generated GLSL forward shaders with
FP32 image storage, pixel losses, analytic backward shaders, texture-gradient
accumulation, and uniform reductions. ``GraphTrainer`` defaults to native GPU
Adam; native GPU SGD is also available. No CPU renderer, numerical backward
approximation, or implicit backend fallback is used.

The same lowered plan can also be executed with display formats or the CPU
reference implementation::

    from sdf_ui.graph import lower_graph
    from sdf_ui.learning import DifferentiableRenderer

    plan = lower_graph({"field": field, "image": image})
    reference = DifferentiableRenderer(canvas.size, weights).render_plan(plan)
    displayed = canvas.render_plan(plan, weights)
    with canvas.compile(plan, weights) as graph:
        with graph.run() as execution:
            field_pixels = execution.read("field")

The display executor stores color passes in RGBA8, while the training and CPU
reference paths use floating-point values. The original ``Canvas.render`` and
CPU ``render`` methods remain available for opaque legacy plugin callbacks.
CPU plan execution requires symbolic inputs rather than materialized GPU textures.

Small scalar/vector parameter expressions and geometric losses use the existing
CPU autodiff engine. Reduced uniform gradients connect the GPU graph back to
those expressions. Only small loss/parameter/gradient vectors cross the CPU/GPU
boundary during training. Pixel readback is an explicit operation.

Run the product demonstration
-----------------------------

From the development checkout::

    uv run python -m examples.gpu_graph --steps 150

The demo creates its target field and image on the GPU, trains a circle's
position, positive radius, and color with one field/image/geometry objective,
and writes before/after images, metrics, and a resumable checkpoint under
``out/learning/gpu_graph``. Training statistics include image readbacks, native
backward dispatches, cache hits, and program compilations.

Resume with the same model and targets::

    uv run python -m examples.gpu_graph --steps 50 --resume out/learning/gpu_graph/checkpoint.json

High-level training session
---------------------------

The session owns its graph, scene-target captures, and optimizer. The Canvas
and any externally supplied GPU images remain caller-owned::

    from sdf_ui import Canvas, sdf
    from sdf_ui.learning import ParameterStore

    weights = ParameterStore()
    radius = weights.scalar("radius", initial=3, min=0)
    field = sdf.circle((16, 16), radius)
    image = field.fill("#72c497", "#17212b").to_rgb()

    with Canvas((32, 32)) as canvas:
        with canvas.training({"field": field, "image": image}, weights) as session:
            session.match("field", sdf.circle((16, 16), 9))
            session.fit(150, lr=0.1)
            print(session.evaluate(), session.report())
            session.save("image", "learned.png")
            session.save_checkpoint("checkpoint.json")

``scalar`` returns a lazy expression, and ``initial`` is the actual rendered
value. Optional bounds are strict and approached asymptotically by smooth
transforms. ``weights.value(name)`` and ``assign_value(name, value)`` work in
rendered units; ``weights[name]`` remains the underlying raw Parameter. The
session captures target scenes once at the graph size and current bindings.
Use ``load_checkpoint`` on a session with an identical objective to resume.

``report(layout=layout)`` includes the device, parameter values, loss terms,
last step, disconnected parameters after training, and layout violations. It
does not read output images. Run the example with::

    uv run python -m examples.high_level_learning

Minimal training example
------------------------

::

    from sdf_ui import Canvas, sdf
    from sdf_ui.graph import GraphTrainer
    from sdf_ui.learning import Objective, ParameterStore

    weights = ParameterStore()
    radius = weights.bind("radius", 3.0)
    field = sdf.circle((16, 16), radius.softplus())
    image = field.fill("#72c497", "#17212b").to_rgb()

    with Canvas((32, 32)) as canvas:
        with canvas.capture(sdf.circle((16, 16), 9)) as target:
            objective = Objective().mse("field", target, name="fit")
            with canvas.compile({"field": field, "image": image}, weights) as graph:
                with GraphTrainer(graph, objective, lr=0.1) as trainer:
                    history = trainer.fit(150)
                    trainer.save("checkpoint.json")
                graph.save("image", "learned.png")

``canvas.capture`` evaluates the target scene once and keeps its floating-point
image on the GPU for the lifetime of the ``with`` block. It works for SDF and
color scenes. Arrays remain available for applications that already have them.

``param`` expressions are rebuilt against current leaves for each execution.
Use ``softplus`` for positive values, ``sigmoid`` for bounded fractions, or
explicit arithmetic to express units and constraints. Percent expressions also
retain their gradients. A derived eager Tensor stored in a persistent scene
becomes stale after an optimizer update; use lazy Exprs for persistent bindings.

Combined objectives and individual execution
--------------------------------------------

``Objective`` provides ``mse``, ``mae``, and ``huber`` for named color or SDF
outputs. Each loss is a mean over all elements, including alpha for RGBA roots.
A scalar target broadcasts to the output; array targets must match exactly.
GPUImage targets are snapshotted on the GPU. NumPy targets are copied and
uploaded once for an unchanged objective. Weights are finite nonnegative
coefficients; geometry callbacks can express signed scalar objectives.
Each pixel loss also accepts ``mask=``: a nonnegative scalar or array matching
the output, or an H-by-W array shared across RGBA channels. A GPU mask must
match the output shape. The weighted loss is averaged over all output elements,
not divided by the sum of mask weights. Set ``normalization="mask"`` to divide
by the sum of mask weights instead; an all-zero mask is invalid in that mode.
GPU masks and targets are snapshotted. Mask sums are reduced on the GPU, with
only the scalar sum transferred to the CPU.

::

    objective = (
        Objective()
        .mse("field", target_field, weight=0.02, name="shape")
        .huber("image", target_image, delta=0.1, name="appearance")
        .geometry(lambda: layout.offscreen_loss(weights), weight=0.001,
                  name="containment", identity={"layout": "v1", "margin": 4})
    )

    # Manual execution is useful for gradient inspection or a custom trainer.
    for parameter in graph.parameters:
        parameter.zero_grad()
    with graph.run(objective) as execution:
        print(execution.loss, execution.metrics)
        gradients = execution.backward()
        print(gradients, execution.stats)
        # Explicit readbacks for debugging:
        pixels = execution.read("image")
        pixel_gradients = execution.gradient("image")

The graph accumulates every loss seed before traversing reverse dependencies.
Shared nodes and shared parameter leaves receive all branch contributions.
``masked_union`` has separate distance and mask ports, both feeding the same
operation's backward pass. The scalar expression/geometry graph then receives
the reduced uniform seeds. The optimizer runs once after this whole process.

``execution.backward`` accumulates into leaf gradients; zero them before an
independent update. A backward tape is consumed once, detects parameter
mutation, and rejects an objective modified after its forward pass. Close the
execution before starting another one or updating graph input textures.

``GraphTrainer.step`` returns pre-update loss and per-term metrics plus execution
statistics. ``fit`` returns those dictionaries and optionally calls a callback
after each completed update. Use ``objective.set_target(name, target)`` and
``set_weight(name, weight)`` between executions to train across changing data.
These refresh target snapshots without recompiling shader programs; render
values remain cached according to their actual inputs.

Validation-based early stopping is optional::

    trainer.fit(
        200,
        validation=lambda: trainer.evaluate(),
        validation_interval=5,
        patience=3,
        min_delta=1e-5,
    )

This example monitors the training objective for simplicity. For model
selection use a separate validation set, and keep final test cases separate.
The initial state is evaluated before the first update. ``patience`` counts
subsequent validation evaluations, and the default restores both model
and optimizer state from the best validation step. Callbacks still receive
every step; evaluated steps include ``validation_loss``. Restoring the best
state requires a checkpointable objective, including explicit identities for
geometry callbacks. The same options are available on graph sessions, dataset
trainers, layout trainers, and named policy training; validation is a loss to
minimize, so negate reward if using policy evaluation. Portable checkpoints
fingerprint GPU target snapshots and may explicitly read those targets back;
this can add overhead when a new best checkpoint is saved.

Graph, dataset, layout, and policy trainer checkpoints use format version 2.
Graph and layout checkpoints validate the parameter-store schema where a store
is supplied. Version 1 and earlier unversioned trainer checkpoints cannot be
resumed directly;
restore them in the original version, export the parameter values, and assign
those values to identically declared parameters in a new run. Optimizer state
is not migrated by this procedure. The dashboard example's ``--initial-policy``
option can start a new optimizer from an existing exported ``policy.json``.

Training across scenes and policy rewards
-----------------------------------------

Graphs sharing one Canvas and the same Parameter leaves can represent different
scenes, content, themes, or targets. A batch averages their weighted gradients
and makes one optimizer update. Evaluation makes no update::

    from sdf_ui.graph import GraphCase, GraphDatasetTrainer

    cases = [GraphCase(graph_a, objective_a, "compact"),
             GraphCase(graph_b, objective_b, "detailed", weight=2)]
    with GraphDatasetTrainer(cases, lr=0.01) as trainer:
        for _ in range(100):
            trainer.step()
        print(trainer.evaluate(held_out_cases))
        trainer.save("batch-checkpoint.json")

``load`` checks the names, weights, graph fingerprints, and objective identities
of all training cases before restoring parameters and optimizer state.

Use ``canvas.compile(outputs, weights, size=(width, height))`` to give each
graph a different viewport while sharing the same OpenGL context and parameter
leaves. ``canvas.capture(scene, size=(width, height))`` can create a matching
target. Sources must match their graph's size. Alternate-size graphs support
training and reference execution; display-plan execution uses the Canvas size.

Continuous policies can use the rendered objective as a reward::

    from sdf_ui.graph import RenderEnvironment
    from sdf_ui.learning import GaussianPolicy, PolicyGradientTrainer

    environment = RenderEnvironment(graph, actions=["x"], objective=objective)
    policy = GaussianPolicy(obs_dim=1, action_dim=1)
    trainer = PolicyGradientTrainer(policy)
    trainer.train(environment, lambda rng: [1.0], steps=20, batch_size=16)

Optionally bind observation components to graph parameters using
``observations=[...]``, or supply ``reward(obs, action, execution)`` to score
the live rendered result. The action bindings are overwritten on each call.
This lower-level environment assigns raw Parameter values; use the named
environment below to bind bounded ``scalar`` expressions in rendered units.
For categorical actions or bounded coordinates, pass an ``action_transform``
that converts the sampled action into the bound scalar values; the reward
callback receives the original action.
This uses REINFORCE: it differentiates the policy's log probability, not
through the sampled action or the renderer into the policy weights.

To train a Gaussian policy directly from differentiable graph losses, use its
lazy deterministic mean expressions as scene parameters::

    policy = GaussianPolicy(obs_dim=1, action_dim=1)
    radius = policy.mean_expressions([0.5])[0].softplus()
    scene = sdf.circle((16, 16), radius)
    with canvas.compile({"field": scene}) as graph:
        with GraphTrainer(graph, Objective().mse("field", target)) as trainer:
            trainer.fit(100)

Each expression captures a fixed observation and references the live policy
weights by default. Use ``input_value`` for a changing observation::

    from sdf_ui import input_value

    width = input_value("width")
    radius = policy.mean_expressions([width])[0].softplus()
    with canvas.compile({"field": sdf.circle((16, 16), radius)},
                        inputs={"width": 0.5}) as graph:
        graph.set_inputs(width=0.8)  # between executions

Inputs are finite, non-trainable scalars; their values are included in graph
trainer checkpoints. Compile cases for different observations and batch them with
``GraphDatasetTrainer`` to train across contexts. The direct path optimizes
the deterministic mean; REINFORCE remains available for sampled actions and
non-differentiable rewards. When alternating the two, use separate optimizers
deliberately and checkpoint both trainers' optimizer states.

Named reward environments
~~~~~~~~~~~~~~~~~~~~~~~~~

``graph.environment`` binds names to graph inputs and scalar parameters. A
reward callback receives read-only named Python scalars, and ``fit`` supplies a
seeded scalar random source to the observation callback::

    from sdf_ui import input_value
    from sdf_ui.learning import GaussianPolicy, PolicyGradientTrainer

    width = input_value("width")
    x = weights.scalar("x", initial=2)
    field = sdf.circle((width, 4), x)
    with canvas.compile({"field": field}, weights, inputs={"width": 4}) as graph:
        env = graph.environment(
            observations={"width": width}, actions={"x": x},
            objective=Objective().mse("field", 0),
            reward=lambda observation, action, run: -run.loss,
            identity={"reward": "field-mse-v1"},
        )
        trainer = PolicyGradientTrainer(GaussianPolicy(1, 1))
        trainer.fit(env, lambda random: {"width": random.uniform(2, 6)},
                    steps=20, batch_size=16)
        print(trainer.evaluate(env, [{"width": 3}, {"width": 5}]))
        print(env.act(trainer.policy, {"width": 4}).x)

For bounded action parameters, provide an ``action_transform`` mapping the
policy's sampled values into the parameter's open bounds. The callback receives
Python scalars. Evaluating an environment changes its bound graph values.
Named policy checkpoints require an explicit JSON-compatible environment
identity that describes the objective, reward, and action transformation.
They validate that identity, binding names, and scalar schemas on reuse.

GPU inputs, export, and existing engine integration
---------------------------------------------------

A materialized ``SDFTexture`` or ``ColorTexture`` from the same open Canvas can
be a static graph input. The graph makes its own FP32 GPU snapshot using a
sampler-copy pass, supporting normalized RGB/RGBA and float SDF textures. It
does not read those pixels to the CPU or take ownership of the original texture.
Inputs must match canvas dimensions and retain their shape/color-space schema.

This lets existing texture loading or one-time font/glyph generation feed the
native graph. You can optimize transforms, colors, masking, and effects applied
to that frozen content. Font shaping and the generation of discrete glyph
topology are not differentiated automatically.

Call ``graph.refresh_inputs()`` between executions after changing an original
texture. Refresh is atomic: invalid replacement inputs leave the previous
snapshots intact. ``execution.input_gradient(index)`` explicitly reads an input
texture adjoint for integration with another model or tool.

``graph.snapshot(name)`` creates an independently owned FP32 GPU image without
readback. It can serve as a target for another graph sharing the same context.
Close the returned GPUImage when finished. ``graph.save`` or ``execution.save``
converts a color root to RGB on the GPU, then explicitly reads/flips/quantizes
pixels for file export. The existing ``Canvas.render`` display API remains
available alongside this compiled execution entry point.

Compilation, caches, and lifetime
---------------------------------

The plan lowers builtin composites such as blur, shadow, masks, gradients,
transforms, and convex polygons into normal shader passes. Shared original
nodes are evaluated once per execution. Every registered builtin shader has a
native GPU derivative body. Fixed iteration counts, polygon side counts, blur
bases, and output indices determine topology and require recompilation to change.
Integer addressing and hard thresholds retain their mathematically zero
derivatives rather than receiving invented learning signals.

Programs are cached by source per Canvas context and reused by its graphs.
Each backend owns reusable scratch buffers. Value caches retain only the current
images for each pass, keyed by resolved FP32 uniform values and input-image
generations. Static branches stay cached as trainable branches change. The cache
does not retain every past parameter version. Execution tapes hold fresh scalar
expressions and parameter versions separately from these images.

``graph.cache_info`` reports current pass/image storage. ``graph.clear_cache``
forces value recomputation. A changed Parameter binding identity, canvas size,
or texture schema requires a new compiled graph. Ordinary Parameter.assign
updates values and versions without recompiling.

Use context managers or explicit ``close`` methods. A Canvas closes its
registered graphs; a graph closes live executions, its targets/caches/buffers,
and trainers' owned optimizers. Supplied optimizers and original textures remain
caller-owned. One GL context/thread must remain current for GPU work.

Optimizers and checkpoints
--------------------------

``GPUAdam(canvas, parameters, ...)`` and ``GPUSGD(canvas, parameters, ...)``
implement the same update semantics as the reference optimizers, including
momentum, decay, and disconnected parameters. Candidate values and moments are
computed on GPU and committed only after finite-value validation. Small leaf
values are synced back to Parameters so the next scalar/geometry execution sees
the update. Existing CPU optimizers can also be supplied explicitly.
Import these optimizers from ``sdf_ui.learning`` or ``sdf_ui.backends.opengl``;
their implementation lives in the OpenGL backend.

Trainer checkpoints contain weights, optimizer moments/counters, step count,
graph schema/source fingerprint, and objective signature. Recreate the graph,
targets, and callbacks, then call ``trainer.load(path)``. The loader checks the
graph fingerprint, pixel targets, loss settings, and explicit geometry identities.
Every geometry callback must provide an ``identity`` before saving or loading a
checkpoint. The identity must be JSON-serializable and should include the
callback's version and all configuration values that affect its result. The
caller must update it when captured values or global state change. External
input textures and runtime images are not serialized.
Saving/loading a checkpoint may explicitly read GPU target snapshots to compare
their contents; ``trainer.step`` never reads activation images. External input
textures and dataset generation remain application-owned.

Custom shaders
--------------

The full stack accepts explicit custom ``ShaderFile`` contracts. Supply a
``ShaderDifferentiation`` with named inputs, uniform shapes, and outputs; pass it
as ``ShaderFile(..., derivative_contract=contract, gpu_derivative=body)``.
``body`` is GLSL defining ``void evaluate(ivec2 pos, out D o[8])`` using the shared
dual arithmetic helpers, ``U_uniform`` parameter macros, and ``ld``/``load4``
image reads. The emitter handles VJP seeding, pixel scatter, and reductions.
An optional ``contract.reference`` callable provides CPU conformance/VJP tests.
Register that descriptor with an ordinary shader-backed Plugin.

For a shared forward/backward program, import ``ShaderProgram`` from
``sdf_ui.plugins`` and instead use
``ShaderFile(name, path, derivative_contract=contract, program=ShaderProgram(name, body))``.
The same dual-number ``evaluate`` body supplies both passes; ``path`` supplies
the independent GLSL reference and any uniform initializers. A program body
must not use tangent values to make primal branch decisions. Supplying both a
shared program and a separate GPU derivative is rejected.

Composite plugins can also supply ``lowering_func(builder, plugin, inputs, raw,
values)`` on their Plugin descriptor. The callback returns a port or a tuple of
ports and uses ``builder.emit`` and ``builder.convert`` to compose passes;
``values(resolve)`` resolves the captured parameter bindings for each execution.
The same callback is used when creating plans for all three executors.

Opaque Python render callbacks need explicit lowering. A CPU differentiable hook
alone does not become a GPU implementation. There is no silent renderer fallback.
The native emitter currently supports up to 128 unique sampled scalar image
inputs per output pixel and reports overflow. Shader math uses piecewise
derivatives at feature switches and finite subgradient conventions at origins.

Execution and numerical details
-------------------------------

The native forward path uses the SDF shader operations with image formats changed
to FP32. All 43 standalone primitive, shading, layer and postprocessing forward
and backward programs are emitted from shared dual-number programs. The forward
compiler removes tangent calculations. Circle, ring, and the pointwise Boolean and
interpolation expressions also emit direct reverse-mode VJPs; other operations use local directional derivatives
contracted with upstream seeds. Inactive pixels and known zero-gradient uniforms
are skipped. Input gradients accumulate with float CAS atomics. Uniform contributions
reduce within workgroups, then globally. Explicit GL memory barriers separate
all dependent passes and reductions.

This prototype favors complete operation coverage and verifiable math over
maximum backward throughput: its local derivative work scales with sampled
inputs and uniform components. Compiler-generated reverse kernels or fused
specialized adjoints can optimize that later. Large buffers and accumulated
values use float32; shared scatter sums may vary slightly by execution order.
FP32 image storage and OpenGL 4.3 compute/SSBO support are required. Reduced-
precision or CPU fallback is intentionally absent from this execution path.

Tests cover all builtin GPU VJPs against the CPU oracle, original FP32 forward
values, sampling borders/ties, full graph gradients, multi-root objectives,
custom shaders, materialized texture inputs, optimizer convergence/checkpoints,
bounded caches, and training with CPU image paths/readback disabled.

Unified execution and inspection
--------------------------------

``ParameterStore.bind(name, initial)`` creates a parameter and returns a reusable
scene expression. A subsequent ``bind(name)`` returns another reference to the
same parameter. ``canvas.compile(outputs, parameters)`` accepts named roots or
an existing plan. The compiled graph renders all roots with an explicit backend::

    radius = weights.bind("radius", 4.0)
    field = sdf.circle((16, 16), radius.softplus())
    with canvas.compile({"field": field}, weights) as graph:
        display = graph.render(backend="display")
        reference = graph.render(backend="reference")
        training = graph.render(backend="training")
        try:
            print(graph.explain())
        finally:
            for image in training.values():
                image.close()

Display values are ordinary materialized textures, reference values are CPU
Tensors, and training values are independently owned FP32 GPUImage snapshots.
All three use the compiled graph's input snapshots. Reference execution reads
materialized input snapshots to the CPU explicitly; the display and training
paths do not. Unsupported operations raise errors; no implicit backend fallback
is performed.

``graph.explain()`` lists structural parameter-to-uniform dependencies,
zero-gradient uniforms, sampling behavior, and topology controls. Dependencies
are conservative: branch choices, zero loss weights, or cancellation can still
produce zero numerical gradients. ``sdf_ui.graph.inspect_graph(outputs)`` reports
lowering failures and topology without creating an OpenGL context. Integer
selectors and fixed topology need recompilation to change. Inspection checks
shader sources and derivative availability, but GPU driver compilation and
runtime numerical behavior are verified only when the graph is compiled or run.

The extension API is available under ``sdf_ui.plugins``. Graph execution types
are available under ``sdf_ui.graph``; OpenGL device types are under
``sdf_ui.backends.opengl``.

Benchmarking
------------

Run a synchronized wall-clock benchmark on the active OpenGL device::

    uv run python -m examples.benchmark_graph --size 64 --steps 20 --output out/learning/benchmark.json

The report includes the GL vendor, renderer, version, image size, warmup count,
and mean/minimum/median/maximum times. It measures full uncached forward and
loss seeding, backward including small gradient transfers, and complete training
steps. OpenGL software renderers are identified in the output and must not be
mistaken for hardware GPU measurements.
