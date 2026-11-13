"""WebGPU parameter optimizers used by render-graph training."""

from __future__ import annotations

import struct
from collections.abc import Mapping

import numpy as np
import wgpu

from ..learning.optimizers import SGD, Adam

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
        self._pipeline = canvas.compute_pipeline(_SHADER, label="Adam optimizer")
        storage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        self._input = canvas.create_buffer(count * _INPUT.size, usage=storage, label="Adam inputs")
        self._previous = canvas.create_buffer(count * _STATE.size, usage=storage, label="Adam state")
        self._candidate = canvas.create_buffer(count * _STATE.size, usage=storage, label="Adam candidate")
        self._values = canvas.create_buffer(count * 4, usage=storage, label="Adam values")
        self._status = canvas.create_buffer(4, usage=storage, label="Adam status")
        self._uniform = canvas.create_buffer(
            32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="Adam config",
        )

    def _check_open(self):
        if self._closed:
            raise RuntimeError("GPU optimizer is closed")
        self.canvas._check_open()

    def zero_grad(self):
        self._check_open()
        super().zero_grad()

    def step(self):
        self._check_open()
        gradients = dict(self._gradients())
        inputs = bytearray(self._count * _INPUT.size)
        states = bytearray(self._count * _STATE.size)
        active = []
        for parameter, span in zip(self.parameters, self._spans):
            values = _f32(parameter.numpy(), f"parameter {parameter.name}").reshape(-1)
            gradient = gradients.get(parameter)
            state = self._state.get(parameter.name, {})
            previous = np.zeros_like(values, dtype=np.float32)
            second = np.zeros_like(values, dtype=np.float32)
            step = int(state.get("step", 0))
            if state:
                previous = _f32(state["exp_avg"], "Adam first moment").reshape(-1)
                second = _f32(state["exp_avg_sq"], "Adam second moment").reshape(-1)
            for local, position in enumerate(range(span.start, span.stop)):
                value = values[local] if local < values.size else 0.0
                grad = 0.0
                enabled = int(gradient is not None and local < values.size)
                if enabled:
                    grad = float(_f32(gradient, f"gradient for {parameter.name}").reshape(-1)[local])
                _INPUT.pack_into(inputs, position * _INPUT.size, value, grad, enabled, 0)
                _STATE.pack_into(
                    states, position * _STATE.size,
                    float(previous[local]) if local < previous.size else 0.0,
                    float(second[local]) if local < second.size else 0.0,
                    step, 0,
                )
            if gradient is not None:
                next_step = step + 1 if self._mode == 0 else int(self.betas[0] > 0)
                active.append((parameter, span, next_step))
        beta1, beta2 = self.betas
        config = struct.pack(
            "<I5fII", self._count, self.lr, beta1, beta2, self.eps,
            self.weight_decay, self._mode, self._nesterov,
        )
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
        expected = {parameter.name: list(parameter.shape) for parameter in self.parameters}
        if (
            not isinstance(state, Mapping) or state.get("format_version") != 1
            or state.get("type") != type(self).__name__ or state.get("parameters") != expected
        ):
            raise ValueError("incompatible optimizer state")
        config = state.get("hyperparameters")
        if not isinstance(config, Mapping):
            raise ValueError("invalid optimizer hyperparameters")
        candidate = Adam(self.parameters, **dict(config))
        raw = state.get("state")
        if not isinstance(raw, Mapping):
            raise ValueError("invalid optimizer buffers")
        by_name = {parameter.name: parameter for parameter in self.parameters}
        staged = {}
        for name, entry in raw.items():
            if name not in by_name or not isinstance(entry, Mapping):
                raise ValueError("invalid optimizer parameter state")
            staged[name] = self._validate_entry(entry, by_name[name].shape)
        candidate._state = staged
        for name, value in candidate._hyperparameters().items():
            setattr(self, name, value)
        self._state = staged

    def close(self):
        if self._closed:
            return
        self._closed = True
        for resource in (
            self._input, self._previous, self._candidate, self._values,
            self._status, self._uniform,
        ):
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
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        validated = SGD(parameters, **config)._hyperparameters()
        super().__init__(
            canvas,
            parameters,
            lr=validated["lr"],
            betas=(validated["momentum"], 0.0),
            eps=1.0,
            weight_decay=validated["weight_decay"],
        )
        self.momentum = validated["momentum"]
        self.nesterov = validated["nesterov"]
        self._mode = 1
        self._nesterov = int(self.nesterov)

    def _hyperparameters(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
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
        expected = {parameter.name: list(parameter.shape) for parameter in self.parameters}
        if (
            not isinstance(state, Mapping)
            or state.get("format_version") != 1
            or state.get("type") != type(self).__name__
            or state.get("parameters") != expected
        ):
            raise ValueError("incompatible optimizer state")
        config = state.get("hyperparameters")
        if not isinstance(config, Mapping):
            raise ValueError("invalid optimizer hyperparameters")
        candidate = SGD(self.parameters, **dict(config))
        raw_state = state.get("state")
        if not isinstance(raw_state, Mapping):
            raise ValueError("invalid optimizer buffers")
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
        config = candidate._hyperparameters()
        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]
        self.nesterov = config["nesterov"]
        self.betas = (self.momentum, 0.0)
        self.eps = 1.0
        self._nesterov = int(self.nesterov)
        self._state = staged
