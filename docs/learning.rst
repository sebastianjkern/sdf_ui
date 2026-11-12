Learning layouts and graphics
=============================

The CPU reference backend provides three working training paths:

* Optimize semantic geometry with differentiable layout losses.
* Learn layout decisions from ordinary scalar reward callbacks.
* Fit shapes and colors through a floating-point render graph.

These reference paths use NumPy on the CPU. They do not require PyTorch, Handwired, a
network service, or an OpenGL context. Existing GPU rendering can display the
same scene with the learned parameters. The differentiable backend is a
separate execution backend, not a gradient approximation around GPU readback.
For native GPU forward and backward execution, see :doc:`gpu_render_graph`.

Run the demonstrations
----------------------

From an installed development checkout::

    uv run python -m examples.learning_ui --mode all --steps 250

The command writes before/after images, learned parameter checkpoints, a
policy-training checkpoint, and ``metrics.json`` under ``out/learning``.
Individual modes are ``geometry``, ``reward``, and ``image``. All examples run
without a GPU. ``--output`` changes the destination directory.

The geometry example aligns three labeled buttons and repairs their spacing.
The reward example learns positions over multiple viewport widths and a
discrete responsive orientation choice. The image example fits a circle's
center and radius to a target distance field.

Parameters and reusable expressions
------------------------------------

Declare parameters once and keep the scene structure stable::

    from sdf_ui import Canvas, param, sdf
    from sdf_ui.learning import ParameterStore

    weights = ParameterStore()
    weights.define("radius", 8.0)
    scene = sdf.circle((32, 32), param("radius").softplus())

    # Use the trained values in the GPU display renderer.
    with Canvas((64, 64)) as canvas:
        scene.fill("#62bb47", "#17212b").to_rgb().save(
            "circle.png", ctx=canvas, params=weights
        )

``Parameter`` is a named graph weight. ``ParameterStore`` is a mapping accepted
by graph rendering and layout methods. Weights are trainable by default; pass
``trainable=False`` to ``define`` or ``bind`` to keep a value fixed during
optimization. ``assign(value)`` preserves shape and advances its version.

Parameter-store checkpoints are versioned and contain raw values plus names,
shapes, and scalar bounds. Loading validates the complete schema before changing
any parameter. Existing unversioned value-only files must be converted explicitly
by recreating the original declarations and assigning their values; they are not
accepted as checkpoints.

``param("name")`` builds a lazy expression, resolved anew for each execution.
Arithmetic, powers, sine/cosine, percentages, ReLU, softplus, sigmoid, tanh,
square root, minimum, and maximum preserve gradients when evaluated in learning
mode. Ordinary GPU rendering snapshots the same expressions into uniform values.

Tensor arithmetic is eager. Rebuild derived tensor computations inside the
loss callback on every step. Use ``param(...)`` expressions for derived values
stored in persistent scene or layout descriptions. Reusing a tape after one of
its parameters changes raises a stale-graph error.

Semantic layout optimization
----------------------------

Bounds use ``(x, y, width, height)`` with full width and height. ``Element.shape``
converts these to the center and half extents expected by ``sdf.rect``.
Coordinates follow the engine's field coordinates; exported images flip the
vertical axis for display.

::

    from sdf_ui.learning import (
        Adam, Bounds, Element, Layout, LayoutTrainer, ParameterStore,
    )

    weights = ParameterStore()
    left = weights.define("left", 10.0)
    right = weights.define("right", 35.0)
    layout = Layout([
        Element("save", Bounds(left, 30, 72, 28), role="button"),
        Element("cancel", Bounds(right, 30, 72, 28), role="button"),
    ], size=(240, 100))

    def loss():
        return (
            layout.gap_loss(["save", "cancel"], gap=16)
            + layout.offscreen_loss()
            + (left - 20) ** 2
        )

    trainer = LayoutTrainer(
        weights.parameters(), loss, Adam(weights.parameters(), lr=1.0)
    )
    history = trainer.fit(200)
    snapshot = layout.snapshot()
    print(snapshot.bounds, snapshot.overlap_area)

Available losses include pairwise overlap area, offscreen area, minimum size,
target sizes, alignment, and consecutive gaps. Alignment supports ``left``,
``right``, ``top``, ``bottom``, ``center_x``, and ``center_y``. Gap loss accepts
ordered element names and an ``x`` or ``y`` axis. Losses are unnormalized;
choose coefficients appropriate to the scene's coordinate scale.

Area penalties can have zero derivatives when elements are separated or
coincident in particular ways. Alignment, gap, or anchor terms provide direction
where an area penalty cannot. Minimum-size loss is a preference; use positive
parameter transformations and fixed minimum extents when a size is mandatory.

Snapshots contain immutable numeric bounds and metadata, not a live tape.
They can safely be inspected by reward callbacks. Add ``containment_loss``
and ``content_overflow_loss`` to an objective to penalize children outside
their parents and content outside its element. These are soft losses, not
automatic guarantees. ``layout.violations(params)`` reports measured overlap,
offscreen area, minimum-size deficits, containment, and content overflow so
an application can reject a candidate before presenting it.
``prevent_overlap`` and ``keep_inside_parent`` return soft penalties;
``is_valid`` and ``assert_valid`` check the measured constraints. For layout
expressions using ``input_value``, pass ``inputs=graph.inputs`` to ``Layout``
so subsequent graph input updates are visible to the layout as well.

``LayoutTrainer.step()`` returns the loss measured before its update.
``fit(steps, callback)`` invokes ``callback(step_index, loss)`` after each update.
The loss callback must return a differentiable scalar connected to the
trainer's parameters; converting it to a float first discards its gradients.
Pass an explicit JSON-compatible ``identity=`` describing the loss when using
layout-trainer checkpoints or validation with best-state restoration. Pass a
``ParameterStore`` itself to include its scalar schemas in the checkpoint.
The identity is supplied by the application: update it when the callback's
meaning, referenced data, or rule configuration changes.

Procedural dashboard example
----------------------------

Run from the repository root::

    uv run python -m examples.learned_dashboard --steps 80

The first run compiles several GPU graphs. On a software OpenGL renderer it can
take several minutes; use ``--steps 5`` for a quick smoke test, though a short
run is not expected to improve every held-out case.

The example learns one policy that allocates horizontal space among three
dashboard cards. Observations are the viewport width and three measured
content widths. The policy produces positive relative shares; the procedural
layout turns those shares into card widths, positions, and SDF shapes. Each
training case uses a different viewport and content mix. Two additional cases
are held out from training.

The loss combines three SDF field comparisons, a rendered appearance loss,
and semantic minimum-size, content-overflow, containment, offscreen, and
overlap penalties. Card positions are constructed as an ordered row with fixed
gaps and margins, so non-overlap and viewport containment are also built into
the parameterization for valid inputs. The policy chooses *widths*, not card
order or row-versus-column structure.

The example writes ``out/learning/dashboard/comparison.png``, individual
before/after/reference images, ``metrics.json``, ``checkpoint.json``, and a
standalone ``policy.json``. The
comparison shows a training case and held-out cases; white bars depict measured
content. A/B/C labels are added after rendering and are not differentiated.
``metrics.json`` includes per-case losses, widths, violations, and training
image-readback counts. Inspect held-out metrics as well as the training loss.

Optionally enable early stopping against a separate validation scenario::

    uv run python -m examples.learned_dashboard --steps 80 --patience 3 \
        --validation-interval 5 --min-delta 0.00001

The held-out scenarios remain separate. The best validation checkpoint is
restored before the final evaluation; ``metrics.json`` records both the number
of updates actually run and the restored step.

The references are *synthetic*: a simple transparent rule assigns each card
its content width plus padding and divides spare space equally. The rule can
solve this demonstration directly. It exists to make the learned behavior
verifiable, not to claim the policy outperforms the rule or learns actual user
preferences. A real application would supply measured content and objectives
validated for its own use case.

Evaluate the saved policy on a new, feasible single-row scenario without
additional training::

    uv run python -m examples.learned_dashboard --steps 0 \
        --resume out/learning/dashboard/checkpoint.json \
        --scenario 150 20 15 28 --output out/learning/dashboard_custom

The four numbers are viewport width followed by the three content widths in
pixels. ``--scenario`` can be repeated. The reference rule rejects scenarios
whose content cannot fit in one row. Loading a checkpoint restores policy and
optimizer values; the initialization seed is only used for a fresh policy.
To start a new optimizer from an existing exported policy instead, use
``--initial-policy out/learning/dashboard/policy.json``. This preserves the
weights but starts optimizer moments and the step count from zero.

To obtain bounds without constructing an OpenGL context or rendering images::

    uv run python -m examples.learned_dashboard \
        --policy out/learning/dashboard/policy.json --predict 150 20 15 28

This prints the learned x, y, width, and height of each card and any measured
constraint violations as JSON. The
``predict(policy, width, content)`` function provides the same operation from
Python. It returns geometry only; it does not supply interactive controls,
accessibility semantics, or an automatic fallback for infeasible content.

Learning from scalar rewards
----------------------------

Reinforcement learning backpropagates through a policy's action log
probabilities. The renderer and reward callback need not be differentiable.
You can evaluate geometry, render supported built-in shader operations, or reward a completed task.

::

    import numpy as np
    from sdf_ui.learning import GaussianPolicy, PolicyGradientTrainer

    policy = GaussianPolicy(obs_dim=1, action_dim=1, seed=7)
    trainer = PolicyGradientTrainer(policy, seed=8)

    def observation(rng):
        return np.array([rng.uniform(-1, 1)])

    def reward(obs, action):
        # For example, an action can specify a normalized widget position.
        return -float((action[0] - obs[0] * 0.5) ** 2)

    history = trainer.train(reward, observation, steps=200, batch_size=48)
    print(policy.deterministic_action([0.6]))

``GaussianPolicy`` learns continuous actions with configurable fixed exploration
standard deviation. ``CategoricalPolicy`` learns integer choices such as row
versus column. Both provide ``sample(observation, rng)``, returning an action
and differentiable log probability, and ``deterministic_action(observation)``
for evaluation. The trainer uses reward-weighted policy gradients with a
baseline and advantage normalization. Evaluate deterministic held-out contexts
as well as the sampled training reward.

For delayed rewards, use the episodic interface::

    def rollout(act, rng):
        rewards = []
        position = 0.0
        for step in range(5):
            action = act(np.array([position]))
            position += float(action[0])
            rewards.append(-abs(position - 1.0))
        return rewards

    trainer.train_episodes(rollout, steps=100, batch_size=16)

The rollout calls ``act(observation)`` once per transition and returns exactly
one reward per action. The trainer calculates discounted reward-to-go using
``gamma``. Recreate/reset environment state inside every rollout. This prototype
uses an on-policy REINFORCE trainer; collected episodes become stale when the
policy changes and cannot be reused for another update.

Differentiable images
---------------------

``DifferentiableRenderer((width, height), params=weights).render(scene)`` returns
a ``Tensor``. SDF outputs have shape ``(height, width)`` and colors have shape
``(height, width, 4)``. Color nodes retain the existing normalized LAB/RGB
semantics. Call ``to_rgb()`` when constructing a target in RGB.

The backend keeps floating-point intermediates throughout the forward and
backward passes; display clipping and conversion to bytes happen only in
``renderer.save(scene, path)``. Its save method renders afresh, converts color
outputs to RGB, and flips rows for image export. GPU and learning results may
differ slightly because GPU display passes quantize to RGBA8 and some devices
store SDFs at reduced precision.

Use a distance-field target for long-range shape fitting when possible.
Pixel fill gradients are strongest near edges; distant silhouettes can have
little or no gradient. Nonsmooth operations use the derivative conventions
documented by ``Tensor``: useful gradients do not imply a globally convex loss.

The renderer validates the supported operation set before execution and rejects
unsupported operations with ``UnsupportedGradientError``. There is no silent
gradient detachment. A fresh per-render execution cache shares common nodes;
old tapes and parameter versions are not accumulated in a persistent cache.

Supported differentiable operations include ``circle``/``disc``,
``rect``/``rounded_rect`` (including rotation and per-corner radii), ``union``,
``intersection``, ``subtract``, ``smooth_union``, ``interpolate``, ``abs``,
``translate``, ``scale``, ``rotate``, ``fill``, ``generate_mask``,
``clear``/``clear_color``, ``to_rgb``, ``to_lab``, ``alpha_overlay``,
``transparency``, and ``multiply``. Transform gradients use clamped bilinear
sampling of the source field, matching the display transform semantics.
Query ``renderer.supported_operations()`` for the supported operation set. See
:doc:`shader_differentiation` for built-in shader coverage and the distinction
between reference VJPs and GPU execution.

Checkpoints and numerical behavior
----------------------------------

``weights.save(path)`` and ``weights.load(path)`` store/load JSON parameter
values. Declare the same names and shapes before loading. Loading validates the
entire state before mutating values.

Policies and optimizers expose ``state_dict()``/``load_state_dict()``.
``PolicyGradientTrainer`` checkpoints include policy weights, optimizer state,
and random-generator state so training can resume reproducibly. For direct
optimization, save the parameter store and optimizer state together.

Call ``optimizer.zero_grad()`` before each new training backward pass.
Gradients from shared paths accumulate, including broadcast dimensions and
repeated indexed reads. Nonfinite values and gradients, shape mismatches,
and parameter mutation between forward and backward raise errors. Backward
requires a scalar output or an explicitly supplied output gradient.

The reference backend uses float64 arrays for gradient accuracy and performs
no GPU acceleration. Start with modest image sizes for pixel training.
