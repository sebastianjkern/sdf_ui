"""Compiled parameter-driven SDF graphs with native GPU forward/backward passes."""

import json
import weakref
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
from PIL import Image

from ..backends.base import Image as BackendImage
from ..backends.base import shape_size
from ..core.expressions import Expr, evaluate_expr
from ..learning.autodiff import Parameter, Tensor
from ..learning.objective import Objective
from ..learning.render import _tensor
from ..util import hex_col
from .lowering import GraphPlan, lower_graph


def _cache_hit(cached, key):
    return (
        cached is not None
        and cached[0] == key
        and not any(image.closed for image in cached[1])
    )


def _close_images(images):
    for image in images:
        image.close()


def _validate_graph_cases(cases, case_type):
    if not cases or any(not isinstance(case, case_type) for case in cases):
        raise ValueError("A batch requires GraphCase instances")
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("Case names must be unique")


def _validate_shared_graph_case(case, graph):
    if case.graph.canvas is not graph.canvas or {
        id(parameter) for parameter in case.graph.parameters
    } != {id(parameter) for parameter in graph.parameters}:
        raise ValueError("Cases must share a Canvas and parameter leaves")


def _validate_trainer_checkpoint(trainer, state, parameter_store_type):
    graph = trainer.graph
    if (
        not isinstance(state, Mapping)
        or state.get("format_version") != 2
        or "parameter_schema" not in state
    ):
        raise ValueError("Invalid trainer checkpoint")
    schema = (
        graph.params.schema()
        if isinstance(graph.params, parameter_store_type)
        else None
    )
    if (
        state.get("graph") != graph.fingerprint
        or state.get("objective") != trainer._objective_signature()
        or state.get("parameter_schema") != schema
    ):
        raise ValueError(
            "Checkpoint graph or objective differs from the current trainer"
        )


def _validate_saved_inputs(expected_names, saved_inputs):
    if not isinstance(saved_inputs, Mapping) or set(saved_inputs) != set(
        expected_names
    ):
        raise ValueError("Checkpoint graph input names differ")
    for name, value in saved_inputs.items():
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"Invalid checkpoint input '{name}'")
    return saved_inputs


def _validate_saved_parameters(parameters, values):
    if not isinstance(values, Mapping) or set(values) != {
        parameter.name for parameter in parameters
    }:
        raise ValueError("Checkpoint parameter names differ")
    validated = {}
    for parameter in parameters:
        value = np.asarray(values[parameter.name], dtype=float)
        if value.shape != parameter.shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid checkpoint value for '{parameter.name}'")
        with np.errstate(over="ignore"):
            if not np.all(np.isfinite(value.astype(np.float32))):
                raise ValueError("Checkpoint values exceed float32 range")
        validated[parameter.name] = value
    return validated


class RenderGraph:
    """A lowered GPU graph with bounded value caches and stable parameter leaves.

    ``outputs`` maps root names to existing TextureNodes. ``params`` supplies
    reusable ``param(name)`` bindings. Direct Parameter leaves in scenes also
    work. Scalar/vector expressions and geometric losses execute on CPU; all
    raster forward, backward, loss seeding, and reductions execute on GPU.
    Only scalar loss values and reduced uniform gradients are read back during
    training. Pixel readback is explicit through an execution's read/save APIs.
    """

    def __init__(self, canvas, outputs, params=None, *, size=None, inputs=None):
        if getattr(canvas, "_closed", False):
            raise RuntimeError("Canvas is closed")
        self.canvas = canvas
        self._canvas_size = tuple(canvas.size)
        size = self._canvas_size if size is None else tuple(size)
        if len(size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
            for v in size
        ):
            raise ValueError("Graph size must be two positive integers")
        self.size = tuple(int(v) for v in size)
        self.params = {} if params is None else params
        if not isinstance(self.params, Mapping):
            raise TypeError("Graph params must be a mapping")
        self._inputs = {}
        self._input_version = 0
        self._set_inputs({} if inputs is None else inputs)
        self.plan = outputs if isinstance(outputs, GraphPlan) else lower_graph(outputs)
        self._source_specs = {
            id(node): (
                node.kind,
                getattr(node, "mode", None),
                tuple(node.tex.size),
                node.tex.components,
                node.tex.dtype,
            )
            for node in self.plan.sources
        }
        self.backend = canvas.create_graph_backend()
        self._cache = {}
        self._sources = []
        self._active = None
        self._running = False
        self._closed = False
        self._trainers = weakref.WeakSet()
        self._targets = {}
        self._mask_denominators = {}
        self._target_signature = None
        self._adjoints = None
        self._uniform_gradients = None
        self._loss_values = None
        self._loss_count = 0
        self._offsets = {}
        self._uniform_offsets = []
        self._parameters = ()
        self._binding_schema = self._schema()
        try:
            self._collect_parameters()
            self._allocate_graph_storage()
        except BaseException:
            self.close()
            raise
        if not hasattr(canvas, "_training_graphs"):
            canvas._training_graphs = weakref.WeakSet()
        canvas._training_graphs.add(self)

    def _collect_parameters(self):
        leaves = {
            id(value): value
            for value in self.params.values()
            if isinstance(value, Parameter)
        }
        for values in self._uniforms():
            for value in values.values():
                leaves.update(
                    {id(parameter): parameter for parameter in value._versions}
                )
        self._parameters = tuple(leaves.values())
        names = [parameter.name for parameter in self._parameters]
        if len(set(names)) != len(names):
            raise ValueError(
                "Graph parameter names must be unique across distinct leaves"
            )

    def _allocate_graph_storage(self):
        offset = self._import_graph_sources()
        offset, uniform_offset, source_hashes = self._allocate_pass_storage(offset)
        self._allocate_training_buffers(offset, uniform_offset, source_hashes)

    def _import_graph_sources(self):
        offset = 0
        for index, node in enumerate(self.plan.sources):
            image = self._import_source(node)
            self._sources.append(image)
            self._offsets[(-index - 1, 0)] = offset
            offset += int(np.prod(image.shape))
        self.backend.validate_images(self._sources)
        self.backend.check_status()
        return offset

    def _allocate_pass_storage(self, offset):
        uniform_offset = 0
        source_hashes = []
        for index, operation in enumerate(self.plan.passes):
            contract = operation.contract
            for output_index, argument in enumerate(contract.outputs):
                self._offsets[(index, output_index)] = offset
                offset += self.size[0] * self.size[1] * shape_size(argument.shape)
            self._uniform_offsets.append(uniform_offset)
            uniform_offset += sum(shape_size(arg.shape) for arg in contract.parameters)
            source_hashes.extend(self.backend.shader_fingerprints(contract.name))
        return offset, uniform_offset, source_hashes

    def _allocate_training_buffers(self, offset, uniform_offset, source_hashes):
        self._adjoint_count = offset
        self._uniform_count = uniform_offset
        self._adjoints = self.backend.create_buffer(offset * 4)
        self._uniform_gradients = self.backend.create_buffer(uniform_offset * 4)
        schema = tuple(
            (parameter.name, parameter.shape) for parameter in self._parameters
        )
        fingerprint = repr((self.plan.fingerprint, self.size, schema, source_hashes))
        self.fingerprint = sha256(fingerprint.encode()).hexdigest()

    @property
    def parameters(self):
        return self._parameters

    @property
    def trainable_parameters(self):
        return tuple(parameter for parameter in self._parameters if parameter.trainable)

    @property
    def inputs(self):
        return MappingProxyType(self._inputs)

    def _set_inputs(self, values):
        if not isinstance(values, Mapping):
            raise TypeError("Graph inputs must be a mapping")
        checked = {}
        for name, value in values.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
            ):
                raise ValueError("Graph inputs must be named finite scalars")
            checked[name] = float(value)
        self._inputs.update(checked)
        self._input_version += 1

    def set_inputs(self, values=None, **kwargs):
        """Update non-trainable scalar inputs between executions."""
        self._check()
        if self._active is not None and not self._active.closed:
            raise RuntimeError("Close the active execution before changing inputs")
        combined = dict(values or {})
        if set(combined) & set(kwargs):
            raise ValueError("Input was supplied twice")
        combined.update(kwargs)
        self._set_inputs(combined)
        return self

    def environment(
        self,
        *,
        actions,
        observations=None,
        objective=None,
        reward=None,
        action_transform=None,
        identity=None,
    ):
        """Create a named policy environment backed by this graph."""
        from .environment import NamedRenderEnvironment

        return NamedRenderEnvironment(
            self,
            actions=actions,
            observations=observations,
            objective=objective,
            reward=reward,
            action_transform=action_transform,
            identity=identity,
        )

    @property
    def closed(self):
        return self._closed

    def _schema(self):
        return tuple(
            (name, id(value), value.shape)
            for name, value in self.params.items()
            if isinstance(value, Parameter)
        )

    def _import_source(self, node):
        spec = (
            node.kind,
            getattr(node, "mode", None),
            tuple(node.tex.size),
            node.tex.components,
            node.tex.dtype,
        )
        if spec != self._source_specs[id(node)]:
            raise ValueError("Input texture schema changed; compile a new graph")
        context = node.context
        if not self.backend.owns_context(context):
            raise ValueError(
                "Graph texture inputs must belong to this open Canvas context"
            )
        if tuple(node.tex.size) != self.size:
            raise ValueError("Graph texture inputs must match the canvas size")
        return self.backend.import_texture(node.tex, 1 if node.kind == "sdf" else 4)

    def refresh_inputs(self):
        """Snapshot current external textures between executions, without readback."""
        self._check()
        if self._active is not None and not self._active.closed:
            raise RuntimeError(
                "Close the active execution before refreshing input textures"
            )
        replacements = []
        try:
            self.backend.clear_status()
            for node in self.plan.sources:
                replacements.append(self._import_source(node))
            self.backend.validate_images(replacements)
            self.backend.check_status()
        except BaseException:
            for image in replacements:
                image.close()
            raise
        for image in self._sources:
            image.close()
        self._sources = replacements

    def _check(self):
        if self._closed or getattr(self.canvas, "_closed", False):
            raise RuntimeError("Render graph or canvas is closed")
        if tuple(self.canvas.size) != self._canvas_size:
            raise ValueError("Canvas size changed; compile a new graph")
        if self._schema() != self._binding_schema:
            raise ValueError(
                "Parameter binding identities changed; compile a new graph"
            )

    def check(self):
        """Validate that the graph, Canvas, and parameter bindings are live."""
        self._check()

    def require_idle(self):
        """Reject operations that cannot run during an active execution."""
        self._check()
        if self._running or (self._active is not None and not self._active.closed):
            raise RuntimeError("Close the active execution before this operation")

    def percent(self, value):
        return self.percent_x(value)

    def percent_x(self, value):
        return value / 100 * self.size[0]

    def percent_y(self, value):
        return value / 100 * self.size[1]

    def percent_of_min(self, value):
        return value / 100 * min(self.size)

    def _resolve(self, value):
        if isinstance(value, (Expr, str, Tensor)):
            value = evaluate_expr(value, self, self.params, track_grad=True)
        if isinstance(value, (tuple, list)):
            return tuple(self._resolve(v) for v in value)
        if isinstance(value, Mapping):
            return {name: self._resolve(v) for name, v in value.items()}
        if isinstance(value, str) and value.startswith("#"):
            return hex_col(value)
        return value

    def _uniforms(self):
        result = []
        for operation in self.plan.passes:
            raw = operation.uniforms(self._resolve)
            # Include declared default uniforms so offsets, layouts, and VJP
            # result ranges never depend on which optional values were supplied.
            defaults = self.backend.defaults(operation.shader.name)
            unknown = set(raw) - {a.name for a in operation.contract.parameters}
            if unknown:
                raise ValueError(f"Unknown lowered uniforms: {sorted(unknown)}")
            values = {}
            for argument in operation.contract.parameters:
                value = _tensor(
                    raw.get(
                        argument.name,
                        defaults.get(argument.name, np.zeros(argument.shape)),
                    )
                )
                if value.shape != argument.shape:
                    raise ValueError(
                        f"Uniform '{argument.name}' expects {argument.shape}, got {value.shape}"
                    )
                value._check_versions()
                values[argument.name] = value
            result.append(values)
        return result

    def run(self, objective=None):
        self._check()
        if self._running or (self._active is not None and not self._active.closed):
            raise RuntimeError(
                "Close the previous graph execution before starting another"
            )
        if objective is not None and not isinstance(objective, Objective):
            raise TypeError("Graph objective must be an Objective")
        self._running = True
        try:
            execution = GraphExecution(self, objective)
            self._active = execution
            return execution
        finally:
            self._running = False

    execute = run

    def render(self, *, backend="display"):
        """Render all roots as display textures, reference Tensors or FP32 GPU images.

        Returned GPU images and display textures belong to the caller. The
        training mode returns owned snapshots and performs no pixel readback.
        """
        self._check()
        if self._running or (self._active is not None and not self._active.closed):
            raise RuntimeError("Close the active execution before rendering")
        if backend == "display":
            return self._render_display()
        if backend == "reference":
            return self._render_reference()
        if backend == "training":
            return self._render_training()
        raise ValueError("backend must be display, reference, or training")

    def _render_display(self):
        if self.size != self._canvas_size:
            raise ValueError(
                "Display rendering requires the Canvas size; use training or reference rendering for this graph"
            )
        from ..backends.plan import execute_display

        sources = [
            self.backend.display_image(image, getattr(node, "mode", None))
            for image, node in zip(self._sources, self.plan.sources)
        ]
        return execute_display(
            self.plan,
            self.canvas,
            self.params,
            sources=sources,
            expression_context=self,
        )

    def _render_reference(self):
        from ..backends.plan import execute_reference
        from ..learning.render import DifferentiableRenderer

        sources = [image.read() for image in self._sources]
        renderer = DifferentiableRenderer(self.size, self.params)
        renderer.inputs = self.inputs
        return execute_reference(self.plan, renderer, sources=sources)

    def _render_training(self):
        images = {}
        try:
            with self.run() as execution:
                for name, image in execution.outputs.items():
                    images[name] = self.backend.copy_image(image)
            return images
        except BaseException:
            for image in images.values():
                image.close()
            raise

    def explain(self):
        """Report topology, derivative conventions and parameter dependencies.

        Dependencies are structural; a branch, zero loss or cancellation may
        still yield zero numerical gradients in a particular execution.
        """
        from .inspection import inspect_graph

        self._check()
        report = inspect_graph(self.plan, self.params)
        parameters = {
            p.name: {
                "shape": list(p.shape),
                "continuous_uniforms": [],
                "zero_gradient_uniforms": [],
            }
            for p in self.parameters
        }
        for operation, values in zip(self.plan.passes, self._uniforms()):
            for argument in operation.contract.parameters:
                value = values[argument.name]
                zero = argument.name in operation.contract.zero_gradient_parameters
                for parameter in value._versions:
                    key = "zero_gradient_uniforms" if zero else "continuous_uniforms"
                    entry = {"shader": operation.shader.name, "uniform": argument.name}
                    if entry not in parameters[parameter.name][key]:
                        parameters[parameter.name][key].append(entry)
        report["parameters"] = parameters
        return report

    def snapshot(self, name):
        """Copy a root to an owned FP32 GPU image without pixel readback."""
        with self.run() as execution:
            return self.backend.copy_image(execution.outputs[name])

    def save(self, name, path):
        with self.run() as execution:
            return execution.save(name, path)

    def clear_cache(self):
        self._check()
        if self._active is not None and not self._active.closed:
            raise RuntimeError("Cannot clear caches during a live execution")
        for _, images in self._cache.values():
            for image in images:
                image.close()
        self._cache.clear()

    def cache_info(self):
        return {
            "input_images": len(self._sources),
            "input_bytes": sum(
                int(np.prod(image.shape)) * 4 for image in self._sources
            ),
            "passes": len(self._cache),
            "images": sum(len(v[1]) for v in self._cache.values()),
            "bytes": sum(
                int(np.prod(image.shape)) * 4
                for _, images in self._cache.values()
                for image in images
            ),
        }

    def objective_signature(self, objective, *, portable=False):
        """Fingerprint an objective and any currently captured target snapshots."""
        self._check()
        if not isinstance(objective, Objective):
            raise TypeError("Expected an Objective")
        current = (id(objective), objective.version, objective.signature())
        targets = self._targets if self._target_signature == current else {}
        overrides = {}
        for index, term in enumerate(objective.pixel_losses):
            if index in targets:
                overrides[term.name] = targets[index]
            if ("mask", index) in targets:
                overrides[("mask", term.name)] = targets[("mask", index)]
        return objective.signature(portable=portable, target_overrides=overrides)

    def _prepare_targets(self, objective, images):
        signature = (id(objective), objective.version, objective.signature())
        if signature != self._target_signature:
            for target in self._targets.values():
                target.close()
            self._targets.clear()
            self._mask_denominators.clear()
            try:
                self._capture_objective_targets(objective, images)
            except BaseException:
                for target in self._targets.values():
                    target.close()
                self._targets.clear()
                self._target_signature = None
                raise
            self._target_signature = signature
        self._ensure_loss_buffer(len(objective.pixel_losses))

    def _capture_objective_targets(self, objective, images):
        for index, term in enumerate(objective.pixel_losses):
            if term.output not in self.plan.outputs:
                raise ValueError(f"Unknown objective root '{term.output}'")
            prediction = images[term.output]
            self._targets[index] = self._copy_loss_target(term, prediction)
            mask = self._copy_loss_mask(term, prediction)
            if mask is not None:
                self._targets[("mask", index)] = mask
                if term.normalization == "mask":
                    self._mask_denominators[index] = self.backend.mask_weight_sum(mask)

    def _copy_loss_target(self, term, prediction):
        if isinstance(term.target, BackendImage):
            if term.target.shape != prediction.shape:
                raise ValueError("GPU target shape differs from prediction")
            return self.backend.copy_image(term.target)
        if term.target.shape == ():
            pixels = np.full(prediction.shape, term.target.item(), dtype=np.float32)
        elif term.target.shape == prediction.shape:
            pixels = term.target
        else:
            raise ValueError(
                f"Target shape {term.target.shape} differs from root '{term.output}' {prediction.shape}"
            )
        return self.backend.upload(pixels)

    def _copy_loss_mask(self, term, prediction):
        if term.mask is None:
            return None
        if isinstance(term.mask, BackendImage):
            if term.mask.shape != prediction.shape:
                raise ValueError("GPU mask shape differs from prediction")
            return self.backend.copy_image(term.mask)
        if term.mask.shape == ():
            pixels = np.full(prediction.shape, term.mask.item(), dtype=np.float32)
        elif term.mask.shape == prediction.shape:
            pixels = term.mask
        elif prediction.channels == 4 and term.mask.shape == prediction.shape[:2]:
            pixels = np.repeat(term.mask[..., None], 4, axis=2)
        else:
            raise ValueError(
                f"Mask shape {term.mask.shape} differs from root '{term.output}' {prediction.shape}"
            )
        return self.backend.upload(pixels)

    def _ensure_loss_buffer(self, count):
        if self._loss_count != count or self._loss_values is None:
            if self._loss_values is not None:
                self._loss_values.release()
            self._loss_values = self.backend.create_buffer(count * 4)
            self._loss_count = count

    def close(self):
        if self._closed:
            return
        if self._active is not None:
            self._active.close()
        for trainer in list(self._trainers):
            trainer.close()
        for _, images in self._cache.values():
            for image in images:
                image.close()
        self._cache.clear()
        for image in self._sources:
            image.close()
        self._sources.clear()
        for target in self._targets.values():
            target.close()
        self._targets.clear()
        for buffer in (self._adjoints, self._uniform_gradients, self._loss_values):
            if buffer is not None:
                buffer.release()
        self.backend.close()
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GraphExecution:
    """One versioned forward/loss/backward tape; images remain on GPU."""

    def __init__(self, graph, objective):
        self.graph = graph
        self.closed = False
        self.backward_called = False
        self.objective = objective
        self._objective_version = None if objective is None else objective.version
        self.uniforms = []
        self.geometry = []
        self.outputs = MappingProxyType({})
        self.loss = None
        self.metrics = {}
        self._start_stats = dict(graph.backend.stats)
        self._versions = {p: p.version for p in graph.parameters}
        self._input_version = graph._input_version
        self._dependencies = []
        self._images = []
        try:
            self._forward()
            if objective is not None:
                self._evaluate_objective()
            self._validate_completion()
        except BaseException:
            self.close()
            # A failed forward must not leave an invalid value eligible for a
            # later cache hit that would bypass the GPU finite-value check.
            for _, images in graph._cache.values():
                for image in images:
                    image.close()
            graph._cache.clear()
            raise

    def _forward(self):
        self.uniforms = self.graph._uniforms()
        self.graph.backend.clear_status()
        self.cache_hits = 0
        for index, (operation, uniforms) in enumerate(
            zip(self.graph.plan.passes, self.uniforms)
        ):
            self._forward_pass(index, operation, uniforms)
        self.outputs = MappingProxyType(
            {
                name: self._port_image(port)
                for name, port in self.graph.plan.outputs.items()
            }
        )
        self.graph.backend.check_status()

    def _forward_pass(self, index, operation, uniforms):
        graph = self.graph
        values = graph.backend.uniforms(operation.shader.name, uniforms)
        input_images = tuple(self._port_image(port) for port in operation.inputs)
        key = (
            tuple((name, value.tobytes()) for name, value in values.items()),
            tuple(image.generation for image in input_images),
        )
        cached = graph._cache.get(index)
        if _cache_hit(cached, key):
            output_images = cached[1]
            self.cache_hits += 1
        else:
            output_images = self._execute_forward_pass(operation, values, input_images)
            if cached is not None:
                _close_images(cached[1])
            graph._cache[index] = (key, output_images)
        self._images.append(output_images)
        self._dependencies.append(self._pass_dependencies(operation, uniforms))

    def _execute_forward_pass(self, operation, values, input_images):
        ports = dict(zip((arg.name for arg in operation.contract.inputs), input_images))
        outputs = self.graph.backend.forward(
            operation.shader.name, ports, values, self.graph.size
        )
        self.graph.backend.validate_images(outputs)
        return outputs

    def _pass_dependencies(self, operation, uniforms):
        dependencies = set()
        for value in uniforms.values():
            dependencies.update(value._versions)
        for port in operation.inputs:
            if port.pass_index >= 0:
                dependencies.update(self._dependencies[port.pass_index])
        if dependencies - set(self.graph.parameters):
            raise ValueError(
                "Trainable parameter leaves changed after graph compilation"
            )
        return dependencies

    def _evaluate_objective(self):
        if not self.objective.pixel_losses and not self.objective.geometry_losses:
            raise ValueError("Objective must contain at least one loss term")
        graph = self.graph
        graph._prepare_targets(self.objective, self.outputs)
        graph._adjoints.clear()
        graph._uniform_gradients.clear()
        graph._loss_values.clear()
        self._seed_pixel_losses()
        self._evaluate_geometry_losses()
        self.loss = float(sum(self.metrics.values()))

    def _seed_pixel_losses(self):
        graph = self.graph
        for index, term in enumerate(self.objective.pixel_losses):
            port = graph.plan.outputs[term.output]
            graph.backend.seed_pixel_loss(
                term,
                self.outputs[term.output],
                graph._targets[index],
                graph._targets.get(("mask", index)),
                graph._adjoints,
                graph._offsets[(port.pass_index, port.output_index)],
                graph._loss_values,
                index,
                graph._mask_denominators.get(index),
            )
        graph.backend.check_status()
        count = len(self.objective.pixel_losses)
        values = (
            np.frombuffer(graph._loss_values.read(size=count * 4), dtype=np.float32)
            if count
            else ()
        )
        for term, value in zip(self.objective.pixel_losses, values):
            self.metrics[term.name] = float(value)

    def _evaluate_geometry_losses(self):
        for term in self.objective.geometry_losses:
            value = _tensor(term.callback())
            if value.size != 1 or not np.isfinite(value.item()):
                raise ValueError("Geometry callbacks must return finite scalar losses")
            if set(value._versions) - set(self.graph.parameters):
                raise ValueError(
                    "Geometry loss uses parameters absent from graph params"
                )
            weighted = value * term.weight
            self.geometry.append(weighted)
            self.metrics[term.name] = weighted.item()

    def _validate_completion(self):
        self._check_versions()
        if (
            self.objective is not None
            and self.objective.version != self._objective_version
        ):
            raise RuntimeError("Objective changed during execution")

    @property
    def stats(self):
        result = {
            name: value - self._start_stats.get(name, 0)
            for name, value in self.graph.backend.stats.items()
        }
        result["cache_hits"] = self.cache_hits
        return result

    def _check(self):
        if self.closed:
            raise RuntimeError("Graph execution is closed")
        self.graph._check()

    def _port_image(self, port):
        if port.pass_index < 0:
            return self.graph._sources[-port.pass_index - 1]
        return self._images[port.pass_index][port.output_index]

    def _check_versions(self):
        self.graph._check()
        if self._input_version != self.graph._input_version:
            raise RuntimeError("Stale graph execution: inputs changed after forward")
        if any(p.version != version for p, version in self._versions.items()):
            raise RuntimeError(
                "Stale graph execution: parameters changed after forward"
            )

    def backward(self):
        self._check()
        self._check_versions()
        if self.objective is None:
            raise ValueError("Backward requires an objective")
        if self.objective.version != self._objective_version:
            raise RuntimeError("Objective changed after forward; execute it again")
        if self.backward_called:
            raise RuntimeError("This execution's backward pass was already consumed")
        graph = self.graph
        needed = self._needed_passes()
        reaches_source = self._passes_reaching_sources()
        graph.backend.clear_status()
        for index in reversed(range(len(graph.plan.passes))):
            self._backward_pass(index, needed, reaches_source)
        graph.backend.check_status()
        raw = np.frombuffer(graph._uniform_gradients.read(), dtype=np.float32)
        terms = self._uniform_gradient_terms(raw, needed)
        self._check_versions()
        if terms:
            chain = sum(terms)
            if chain.requires_grad:
                chain.backward()
        self.backward_called = True
        return {
            p.name: None if p.grad is None else p.grad.copy() for p in graph.parameters
        }

    def _needed_passes(self):
        graph = self.graph
        needed = set()
        pending = [
            graph.plan.outputs[term.output].pass_index
            for term in self.objective.pixel_losses
        ]
        while pending:
            index = pending.pop()
            if index < 0 or index in needed:
                continue
            needed.add(index)
            pending.extend(port.pass_index for port in graph.plan.passes[index].inputs)
        return needed

    def _passes_reaching_sources(self):
        reaches_source = []
        for operation in self.graph.plan.passes:
            reaches_source.append(
                any(
                    port.pass_index < 0 or reaches_source[port.pass_index]
                    for port in operation.inputs
                )
            )
        return reaches_source

    def _backward_pass(self, index, needed, reaches_source):
        if index not in needed or not (
            self._dependencies[index] or reaches_source[index]
        ):
            return
        graph = self.graph
        operation = graph.plan.passes[index]
        inputs = {
            argument.name: self._port_image(port)
            for argument, port in zip(operation.contract.inputs, operation.inputs)
        }
        input_offsets = [
            graph._offsets[(port.pass_index, port.output_index)]
            for port in operation.inputs
        ]
        output_offsets = [
            graph._offsets[(index, output_index)]
            for output_index in range(len(operation.contract.outputs))
        ]
        graph.backend.backward_into(
            operation.shader.name,
            inputs,
            self.uniforms[index],
            graph._adjoints,
            input_offsets,
            output_offsets,
            graph._uniform_gradients,
            graph._uniform_offsets[index],
            size=graph.size,
        )

    def _uniform_gradient_terms(self, raw, needed):
        terms = list(self.geometry)
        for index, operation in enumerate(self.graph.plan.passes):
            if index in needed:
                terms.extend(self._pass_uniform_gradient_terms(index, operation, raw))
        return terms

    def _pass_uniform_gradient_terms(self, index, operation, raw):
        terms = []
        base = self.graph._uniform_offsets[index]
        for argument in operation.contract.parameters:
            count = shape_size(argument.shape)
            value = self.uniforms[index][argument.name]
            if value.requires_grad:
                seed = raw[base : base + count].reshape(argument.shape)
                terms.append((value * Tensor(seed)).sum())
            base += count
        return terms

    def read(self, name):
        self._check()
        return self.outputs[name].read()

    def gradient(self, name):
        self._check()
        if not self.backward_called:
            raise RuntimeError("Run backward before reading image gradients")
        port = self.graph.plan.outputs[name]
        image = self.outputs[name]
        offset = self.graph._offsets[(port.pass_index, port.output_index)]
        payload = self.graph._adjoints.read(
            size=int(np.prod(image.shape)) * 4, offset=offset * 4
        )
        self.graph.backend.stats["image_readbacks"] += 1
        return np.frombuffer(payload, dtype=np.float32).reshape(image.shape).copy()

    def input_gradient(self, index):
        """Explicitly read a static input's pixel adjoint for external integrations."""
        self._check()
        if not self.backward_called:
            raise RuntimeError("Run backward before reading input gradients")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self.graph._sources)
        ):
            raise ValueError("Input texture index is outside the compiled source set")
        image = self.graph._sources[index]
        offset = self.graph._offsets[(-index - 1, 0)]
        payload = self.graph._adjoints.read(
            size=int(np.prod(image.shape)) * 4, offset=offset * 4
        )
        self.graph.backend.stats["image_readbacks"] += 1
        return np.frombuffer(payload, dtype=np.float32).reshape(image.shape).copy()

    def save(self, name, path):
        self._check()
        port = self.graph.plan.outputs[name]
        if port.kind != "color":
            raise ValueError("Shade an SDF root before saving it as an image")
        converted = ()
        try:
            image = self.outputs[name]
            if port.mode != "RGB":
                converted = self.graph.backend.forward(
                    "to_rgb", {"origTex": image}, {}, self.graph.size
                )
                image = converted[0]
            pixels = np.flipud(
                np.clip(np.rint(image.read() * 255), 0, 255).astype(np.uint8)
            )
            Image.fromarray(pixels).save(path)
            return Path(path)
        finally:
            for image in converted:
                image.close()

    def close(self):
        if not self.closed:
            self.uniforms.clear()
            self.geometry.clear()
            self._images.clear()
            self._versions.clear()
            if self.graph._active is self:
                self.graph._active = None
            self.closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GraphTrainer:
    """Train a compiled graph with one complete combined-loss update per step.

    The default optimizer is native GPU Adam. Existing CPU SGD/Adam can also
    consume the small leaf-gradient vectors, explicitly selected by the caller.
    The graph/context remain caller-owned. Checkpoints contain parameter values,
    optimizer moments, graph/objective fingerprints, and the step counter.
    """

    def __init__(self, graph, objective, optimizer=None, *, lr=0.01):
        graph._check()
        if not isinstance(objective, Objective):
            raise TypeError("GraphTrainer requires an Objective")
        if not graph.trainable_parameters:
            raise ValueError("Graph has no registered trainable parameters")
        self.graph = graph
        self.objective = objective
        self._owns_optimizer = optimizer is None
        self.optimizer = (
            graph.backend.create_optimizer(graph.trainable_parameters, lr=lr)
            if optimizer is None
            else optimizer
        )
        actual = tuple(getattr(self.optimizer, "parameters", ()))
        if len(actual) != len(graph.trainable_parameters) or {
            id(p) for p in actual
        } != {id(p) for p in graph.trainable_parameters}:
            if self._owns_optimizer:
                self.optimizer.close()
            raise ValueError("Optimizer must own exactly this graph's parameter leaves")
        if not all(
            callable(getattr(self.optimizer, name, None))
            for name in ("zero_grad", "step", "state_dict", "load_state_dict")
        ):
            raise TypeError("Graph optimizer must implement updates and checkpoints")
        self.steps = 0
        self.last_metrics = None
        self.closed = False
        graph._trainers.add(self)

    def _check(self):
        if self.closed:
            raise RuntimeError("Graph trainer is closed")
        self.graph._check()

    def check(self):
        """Validate that this trainer and its graph are open."""
        self._check()

    def step(self):
        self._check()
        self.optimizer.zero_grad()
        with self.graph.run(self.objective) as execution:
            execution.backward()
            if not any(p.grad is not None for p in self.graph.trainable_parameters):
                raise ValueError("Objective has no gradient path to graph parameters")
            self.optimizer.step()
            self.steps += 1
            result = {
                "step": self.steps,
                "loss": execution.loss,
                "terms": dict(execution.metrics),
                "stats": execution.stats,
            }
        self.last_metrics = result
        return result

    def step_batch(self, cases):
        """Apply one weighted-mean gradient update over compiled cases."""
        from ..learning.training import normalized_case_weights
        from .dataset import GraphCase

        self._check()
        cases = tuple(cases)
        _validate_graph_cases(cases, GraphCase)
        weights = normalized_case_weights(cases)
        parameters = self.graph.parameters
        gradients, losses = {}, {}
        try:
            for case, weight in zip(cases, weights):
                self._accumulate_case(case, weight, parameters, gradients, losses)
            if not gradients:
                raise ValueError("Batch has no gradient path to graph parameters")
            self.optimizer.zero_grad()
            for parameter, gradient in gradients.items():
                parameter.grad = gradient
            self.optimizer.step()
        except BaseException:
            self.optimizer.zero_grad()
            raise
        self.steps += 1
        result = {
            "step": self.steps,
            "loss": sum(
                weight * losses[case.name] for case, weight in zip(cases, weights)
            ),
            "cases": losses,
        }
        self.last_metrics = result
        return result

    def _accumulate_case(self, case, weight, parameters, gradients, losses):
        _validate_shared_graph_case(case, self.graph)
        self.optimizer.zero_grad()
        with case.graph.run(case.objective) as execution:
            execution.backward()
            losses[case.name] = execution.loss
        for parameter in parameters:
            if parameter.grad is not None:
                gradients[parameter] = (
                    gradients.get(parameter, 0) + parameter.grad * weight
                )

    def evaluate(self):
        """Evaluate the current objective without updating parameters."""
        self._check()
        with self.graph.run(self.objective) as execution:
            return {"loss": execution.loss, "terms": dict(execution.metrics)}

    def fit(
        self,
        steps,
        callback=None,
        *,
        validation=None,
        patience=None,
        min_delta=0.0,
        restore_best=True,
        validation_interval=1,
    ):
        self._check()
        from ..learning.training import fit_steps

        return fit_steps(
            self,
            steps,
            callback,
            validation=validation,
            patience=patience,
            min_delta=min_delta,
            restore_best=restore_best,
            validation_interval=validation_interval,
        )

    def _objective_signature(self):
        # Checkpointing is explicit: GPU target snapshots may be read here to
        # identify their content portably. No activation image is read in step().
        return self.graph.objective_signature(self.objective, portable=True)

    def state_dict(self):
        self._check()
        from ..learning.parameters import ParameterStore

        if self.graph._active is not None and not self.graph._active.closed:
            raise RuntimeError("Close the execution before saving a trainer checkpoint")
        return {
            "format_version": 2,
            "graph": self.graph.fingerprint,
            "objective": self._objective_signature(),
            "steps": self.steps,
            "parameters": {p.name: p.numpy().tolist() for p in self.graph.parameters},
            "parameter_schema": self.graph.params.schema()
            if isinstance(self.graph.params, ParameterStore)
            else None,
            "inputs": dict(self.graph.inputs),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self._check()
        from ..learning.parameters import ParameterStore

        if self.graph._active is not None and not self.graph._active.closed:
            raise RuntimeError("Close the execution before restoring a checkpoint")
        _validate_trainer_checkpoint(self, state, ParameterStore)
        steps = state.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Invalid checkpoint step count")
        validated = _validate_saved_parameters(
            self.graph.parameters, state.get("parameters")
        )
        saved_inputs = _validate_saved_inputs(
            self.graph.inputs, state.get("inputs", {})
        )
        self.optimizer.load_state_dict(state.get("optimizer"))
        for parameter in self.graph.parameters:
            parameter.assign(validated[parameter.name])
        self.graph.set_inputs(saved_inputs)
        self.steps = steps
        self.last_metrics = None
        self.graph.clear_cache()

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    def load(self, path):
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def close(self):
        if not self.closed:
            if self._owns_optimizer:
                self.optimizer.close()
            self.closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
