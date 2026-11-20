"""WebGPU parameter optimizers used by render-graph training."""

from __future__ import annotations

import struct
from collections.abc import Mapping

import numpy as np
import wgpu

from ..learning.optimizers import SGD, Adam

_MOMENTUM_TEXT = 'momentum'

__all__ = ["GPUAdam", "GPUSGD"]

_INPUT = struct.Struct("<ffII")
_STATE = struct.Struct("<ffII")
_STATE_DTYPE = np.dtype(
    [("first", "<f4"), ("second", "<f4"), ("step", "<u4"), ("pad", "<u4")]
)
_SHADER = """
struct Input { value: f32, gradient: f32, enabled: u32, _pad: u32, };
struct State { first: f32, second: f32, step: u32, _pad: u32, };
struct Config {
    count: u32, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
    _pad0: u32, _pad1: u32,
};
struct Status { bad: atomic<u32>, };
@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<storage, read> previous: array<State>;
@group(0) @binding(2) var<storage, read_write> candidate: array<State>;
@group(0) @binding(3) var<storage, read_write> values: array<f32>;
@group(0) @binding(4) var<storage, read_write> status: Status;
@group(0) @binding(5) var<uniform> config: Config;
fn finite_value(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= config.count) { return; }
    var state = previous[i];
    values[i] = inputs[i].value;
    if (inputs[i].enabled == 0u) { candidate[i] = state; return; }
    let gradient = inputs[i].gradient + config.weight_decay * inputs[i].value;
    if (!finite_value(gradient)) {
        atomicOr(&status.bad, 1u); return;
    }
    var direction = gradient;
    if (config._pad0 == 1u) {
        if (config.beta1 > 0.0) {
            state.first = config.beta1 * state.first + gradient;
            direction = select(state.first, gradient + config.beta1 * state.first,
                config._pad1 != 0u);
            state.step = 1u;
        } else {
            state.first = 0.0;
            state.step = 0u;
        }
        state.second = 0.0;
    } else {
        if (state.step == 0xffffffffu) {
            atomicOr(&status.bad, 1u); return;
        }
        state.step += 1u;
        state.first = config.beta1 * state.first + (1.0 - config.beta1) * gradient;
        state.second = config.beta2 * state.second + (1.0 - config.beta2) * gradient * gradient;
        let first_correction = 1.0 - pow(config.beta1, f32(state.step));
        let second_correction = 1.0 - pow(config.beta2, f32(state.step));
        direction = (state.first / first_correction) /
            (sqrt(state.second / second_correction) + config.eps);
    }
    let value = inputs[i].value - config.lr * direction;
    if (!finite_value(state.first) || !finite_value(state.second) ||
        !finite_value(value) || !finite_value(direction)) {
        atomicOr(&status.bad, 1u); return;
    }
    candidate[i] = state;
    values[i] = value;
}
"""


def _f32(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite in float32")
    return result


class GPUAdam(Adam):
    """Bias-corrected Adam whose update kernel runs on the WebGPU device."""

    def __init__(
        self, canvas, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0,
    ):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.canvas = canvas
        self._closed = False
        self._spans = []
        count = 0
        for parameter in self.parameters:
            _f32(parameter.numpy(), f"parameter {parameter.name}")
            width = max(1, parameter.size)
            self._spans.append(slice(count, count + width))
            count += width
        self._count = count
        self._mode = 0
        self._nesterov = 0
        self._allocate_buffers(canvas, count)

    def _allocate_buffers(self, canvas, count):
        self._pipeline = canvas.compute_pipeline(_SHADER, label="Adam optimizer")
        storage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        self._input = canvas.create_buffer(count * _INPUT.size, usage=storage, label="Adam inputs")
        self._previous = canvas.create_buffer(count * _STATE.size, usage=storage, label="Adam state")
        self._candidate = canvas.create_buffer(count * _STATE.size, usage=storage, label="Adam candidate")
        self._values = canvas.create_buffer(count * 4, usage=storage, label="Adam values")
        self._status = canvas.create_buffer(4, usage=storage, label="Adam status")
        self._uniform = canvas.create_buffer(
            32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="Adam config",
        )

    def _check_open(self):
        from ._lifecycle import check_optimizer_open

        check_optimizer_open(self)

    def zero_grad(self):
        self._check_open()
        super().zero_grad()

    def step(self):
        self._check_open()
        inputs, states, active = self._pack_update_inputs()
        config = self._pack_config()
        updated, next_states = self._run_update(inputs, states, config)
        self._apply_update(updated, next_states, active)

    def _pack_update_inputs(self):
        gradients = dict(self._gradients())
        inputs = bytearray(self._count * _INPUT.size)
        states = bytearray(self._count * _STATE.size)
        active = []
        for parameter, span in zip(self.parameters, self._spans):
            gradient = gradients.get(parameter)
            next_step = self._pack_parameter_update(parameter, span, gradient, inputs, states)
            if next_step is not None:
                active.append((parameter, span, next_step))
        return inputs, states, active

    def _pack_parameter_update(self, parameter, span, gradient, inputs, states):
        values = _f32(parameter.numpy(), f"parameter {parameter.name}").reshape(-1)
        state = self._state.get(parameter.name, {})
        previous, second, step = self._parameter_moments(state, values)
        gradient_values = self._gradient_values(gradient, parameter)
        for local, position in enumerate(range(span.start, span.stop)):
            self._pack_optimizer_slot(
                inputs, states, position, local, values, gradient_values,
                previous, second, step,
            )
        if gradient is None:
            return None
        return step + 1 if self._mode == 0 else int(self.betas[0] > 0)

    @staticmethod
    def _parameter_moments(state, values):
        previous = np.zeros_like(values, dtype=np.float32)
        second = np.zeros_like(values, dtype=np.float32)
        step = int(state.get("step", 0))
        if state:
            previous = _f32(state["exp_avg"], "Adam first moment").reshape(-1)
            second = _f32(state["exp_avg_sq"], "Adam second moment").reshape(-1)
        return previous, second, step

    @staticmethod
    def _gradient_values(gradient, parameter):
        if gradient is None:
            return None
        return _f32(gradient, f"gradient for {parameter.name}").reshape(-1)

    @staticmethod
    def _pack_optimizer_slot(inputs, states, position, local, values,
                             gradient_values, previous, second, step):
        has_value = local < values.size
        value = values[local] if has_value else 0.0
        enabled = int(gradient_values is not None and has_value)
        grad = float(gradient_values[local]) if enabled else 0.0
        first = float(previous[local]) if local < previous.size else 0.0
        second_value = float(second[local]) if local < second.size else 0.0
        _INPUT.pack_into(inputs, position * _INPUT.size, value, grad, enabled, 0)
        _STATE.pack_into(states, position * _STATE.size, first, second_value, step, 0)

    def _pack_config(self):
        beta1, beta2 = self.betas
        return struct.pack(
            "<I5fII", self._count, self.lr, beta1, beta2, self.eps,
            self.weight_decay, self._mode, self._nesterov,
        )

    def _run_update(self, inputs, states, config):
        self.canvas.queue.write_buffer(self._input, 0, inputs)
        self.canvas.queue.write_buffer(self._previous, 0, states)
        self.canvas.queue.write_buffer(self._status, 0, bytes(4))
        self.canvas.queue.write_buffer(self._uniform, 0, config)
        bind_group = self.canvas.bind_group(
            self._pipeline,
            {0: self._input, 1: self._previous, 2: self._candidate,
             3: self._values, 4: self._status, 5: self._uniform},
            label="Adam optimizer",
        )
        self.canvas.dispatch(
            self._pipeline, (bind_group,),
            group_count=((self._count + 63) // 64, 1, 1), label="Adam optimizer",
        )
        status = np.frombuffer(self.canvas.queue.read_buffer(self._status), dtype=np.uint32)[0]
        if status:
            raise FloatingPointError("nonfinite WebGPU Adam update or exhausted step")
        updated = np.frombuffer(self.canvas.queue.read_buffer(self._values), dtype=np.float32)
        next_states = self.canvas.queue.read_buffer(self._candidate)
        return updated, next_states

    def _apply_update(self, updated, next_states, active):
        next_state = {}
        for parameter, span, step in active:
            values = updated[span][: parameter.size].reshape(parameter.shape).astype(np.float64)
            parameter.assign(values)
            moments = np.frombuffer(
                next_states[span.start * _STATE.size:span.stop * _STATE.size],
                dtype=_STATE_DTYPE,
            )
            next_state[parameter.name] = {
                "step": step,
                "exp_avg": moments["first"][:parameter.size].reshape(parameter.shape).astype(np.float64),
                "exp_avg_sq": moments["second"][:parameter.size].reshape(parameter.shape).astype(np.float64),
            }
        self._state.update(next_state)

    def load_state_dict(self, state):
        self._check_open()
        config, raw = self._load_state_payload(state)
        candidate = Adam(self.parameters, **dict(config))
        staged = self._stage_adam_state(raw)
        candidate._state = staged
        for name, value in candidate._hyperparameters().items():
            setattr(self, name, value)
        self._state = staged

    def _load_state_payload(self, state):
        expected = {parameter.name: list(parameter.shape) for parameter in self.parameters}
        if (
            not isinstance(state, Mapping) or state.get("format_version") != 1
            or state.get("type") != type(self).__name__ or state.get("parameters") != expected
        ):
            raise ValueError("incompatible optimizer state")
        config = state.get("hyperparameters")
        if not isinstance(config, Mapping):
            raise ValueError("invalid optimizer hyperparameters")
        raw = state.get("state")
        if not isinstance(raw, Mapping):
            raise ValueError("invalid optimizer buffers")
        return config, raw

    def _stage_adam_state(self, raw):
        by_name = {parameter.name: parameter for parameter in self.parameters}
        staged = {}
        for name, entry in raw.items():
            if name not in by_name or not isinstance(entry, Mapping):
                raise ValueError("invalid optimizer parameter state")
            staged[name] = self._validate_entry(entry, by_name[name].shape)
        return staged

    def close(self):
        if self._closed:
            return
        self._closed = True
        for resource in (
            self._input, self._previous, self._candidate, self._values,
            self._status, self._uniform,
        ):
            if not self.canvas._defer_resource_destroy(resource):
                resource.destroy()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


class GPUSGD(GPUAdam):
    """WebGPU SGD with classical momentum, Nesterov momentum, and L2 decay."""

    def __init__(
        self, canvas, parameters, lr=0.01, momentum=0.0,
        weight_decay=0.0, nesterov=False,
    ):
        parameters = tuple(parameters.values()) if isinstance(parameters, Mapping) else tuple(parameters)
        config = {
            "lr": lr,
            _MOMENTUM_TEXT: momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        validated = SGD(parameters, **config)._hyperparameters()
        super().__init__(
            canvas,
            parameters,
            lr=validated["lr"],
            betas=(validated[_MOMENTUM_TEXT], 0.0),
            eps=1.0,
            weight_decay=validated["weight_decay"],
        )
        self.momentum = validated[_MOMENTUM_TEXT]
        self.nesterov = validated["nesterov"]
        self._mode = 1
        self._nesterov = int(self.nesterov)

    def _hyperparameters(self):
        return {
            "lr": self.lr,
            _MOMENTUM_TEXT: self.momentum,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
        }

    def state_dict(self):
        self._check_open()
        parameters = {parameter.name: parameter for parameter in self.parameters}
        state = {}
        if self.momentum:
            for name, entry in self._state.items():
                state[name] = {
                    "momentum_buffer": entry["exp_avg"].copy().tolist()
                }
        return {
            "format_version": 1,
            "type": type(self).__name__,
            "hyperparameters": self._hyperparameters(),
            "parameters": {name: list(parameter.shape) for name, parameter in parameters.items()},
            "state": state,
        }

    def load_state_dict(self, state):
        self._check_open()
        config, raw_state = self._load_state_payload(state)
        candidate = SGD(self.parameters, **dict(config))
        staged = self._stage_sgd_state(candidate, raw_state)
        config = candidate._hyperparameters()
        self.lr = config["lr"]
        self.momentum = config[_MOMENTUM_TEXT]
        self.weight_decay = config["weight_decay"]
        self.nesterov = config["nesterov"]
        self.betas = (self.momentum, 0.0)
        self.eps = 1.0
        self._nesterov = int(self.nesterov)
        self._state = staged

    def _stage_sgd_state(self, candidate, raw_state):
        by_name = {parameter.name: parameter for parameter in self.parameters}
        staged = {}
        for name, entry in raw_state.items():
            if name not in by_name or not isinstance(entry, Mapping):
                raise ValueError("invalid optimizer parameter state")
            checked = candidate._validate_entry(entry, by_name[name].shape)
            average = _f32(checked["momentum_buffer"], "momentum buffer")
            staged[name] = {
                "step": 1,
                "exp_avg": average,
                "exp_avg_sq": np.zeros_like(average),
            }
        return staged
