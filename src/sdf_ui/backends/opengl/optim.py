"""OpenGL 4.3 optimizers for small CPU autodiff parameter leaves.

Import these optimizers from ``sdf_ui.learning``. Only flattened
leaf values/gradients cross to the GPU; updates and persistent moments are
computed there. Checkpoints follow the CPU optimizers' format (without
weights), with GPU-specific type names. Save parameter values separately.
The caller owns the context and must keep it current until ``close()``.
"""

from collections.abc import Mapping

import numpy as np

from ...learning.autodiff import _array
from ...learning.optimizers import SGD, Adam, _hyper, _Optimizer

__all__ = ["GPUSGD", "GPUAdam"]

_SLOT = np.dtype([("value", "f4"), ("m", "f4"), ("v", "f4"), ("step", "u4")])
_INPUT = np.dtype([("value", "f4"), ("grad", "f4"), ("active", "u4"), ("pad", "u4")])
_MAX_STEP = int(np.iinfo(np.uint32).max)

_SHADER = """#version 430
layout(local_size_x = 64) in;
struct Input { float value; float grad; uint enabled; uint pad; };
struct State { float value; float m; float v; uint step; };
layout(std430, binding=0) readonly buffer Inputs { Input inputs[]; };
layout(std430, binding=1) readonly buffer Previous { State previous[]; };
layout(std430, binding=2) writeonly buffer Candidate { State candidate[]; };
layout(std430, binding=3) buffer Status { uint invalid; };
layout(std430, binding=4) writeonly buffer Values { float values[]; };
uniform uint count;
uniform float lr, weight_decay;
__UNIFORMS__
bool finite_value(float x) { return !isnan(x) && !isinf(x); }
// Integer exponentiation in double precision avoids cancellation in Adam's
// early bias corrections when beta is close to one. GL 4.3 includes fp64.
float correction(float complement, uint n) {
    double base = 1.0lf - double(complement);
    double result = 1.0lf;
    while (n != 0u) {
        if ((n & 1u) != 0u) result *= base;
        base *= base;
        n >>= 1u;
    }
    return float(1.0lf - result);
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    State s = previous[i];
    if (inputs[i].enabled == 0u) { candidate[i] = s; return; }
    float w = inputs[i].value;
    precise float g = inputs[i].grad + weight_decay * w;
    if (!finite_value(g)) atomicOr(invalid, 1u);
    __UPDATE__
    if (!finite_value(s.value) || !finite_value(s.m) || !finite_value(s.v))
        atomicOr(invalid, 1u);
    candidate[i] = s;
    values[i] = s.value;
}
"""

_SGD_UPDATE = """
    precise float direction = g;
    if (use_momentum) {
        s.m = momentum * s.m + g;
        s.step = 1u;
        direction = nesterov ? g + momentum * s.m : s.m;
    }
    s.value = w - lr * direction;
"""

_ADAM_UPDATE = """
    if (s.step == 0xffffffffu) { atomicOr(invalid, 1u); return; }
    s.step += 1u;
    s.m = beta1 * s.m + complement1 * g;
    s.v = beta2 * s.v + (complement2 * g) * g;
    precise float direction = (s.m / correction(complement1, s.step)) /
        (sqrt(s.v / correction(complement2, s.step)) + eps);
    s.value = w - lr * direction;
"""


def _f32(value, name):
    value = _array(value, name)
    with np.errstate(over="ignore", invalid="ignore"):
        result = value.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite in float32")
    return result


class _GPUOptimizer(_Optimizer):
    def __init__(self, canvas, parameters):
        super().__init__(parameters)
        self._closed = False
        self._resources = []
        self._ctx = getattr(canvas, "_mgl_ctx", canvas)
        self._shapes = tuple(p.shape for p in self.parameters)
        self._slices = []
        offset = 0
        for p in self.parameters:
            _f32(p.numpy(), "parameter " + p.name)
            # Empty arrays still need one slot for their per-parameter counter.
            self._slices.append(slice(offset, offset + max(1, p.size)))
            offset += max(1, p.size)
        self._count = offset
        if offset > _MAX_STEP:
            raise ValueError("too many parameter elements")
        if (
            getattr(self._ctx, "version_code", 0) < 430
            or not callable(getattr(self._ctx, "compute_shader", None))
            or not callable(getattr(self._ctx, "buffer", None))
        ):
            raise ValueError("GPU optimizers require an OpenGL 4.3 context")
        try:
            source = _SHADER.replace("__UNIFORMS__", self._uniforms).replace(
                "__UPDATE__", self._update
            )
            self._shader = self._own(self._ctx.compute_shader(source))
            self._input = self._own(self._ctx.buffer(reserve=offset * _INPUT.itemsize))
            self._current = self._own(self._ctx.buffer(bytes(offset * _SLOT.itemsize)))
            self._candidate = self._own(
                self._ctx.buffer(reserve=offset * _SLOT.itemsize)
            )
            self._status = self._own(self._ctx.buffer(bytes(4)))
            self._values = self._own(self._ctx.buffer(reserve=offset * 4))
        except BaseException:
            self.close()
            raise

    def _own(self, resource):
        self._resources.append(resource)
        return resource

    def _check_open(self):
        if self._closed:
            raise RuntimeError("GPU optimizer is closed")
        if tuple(p.shape for p in self.parameters) != self._shapes:
            raise ValueError("optimizer parameter shapes changed")

    def zero_grad(self):
        self._check_open()
        super().zero_grad()

    def step(self):
        """Compute candidates on the GPU, then commit only if all are finite."""
        self._check_open()
        config = self._validate_config(self._hyperparameters())
        gradients = dict(self._gradients())
        payload = np.zeros(self._count, dtype=_INPUT)
        active = []
        for p, span in zip(self.parameters, self._slices):
            values = _f32(p.numpy(), "parameter " + p.name).reshape(-1)
            payload["value"][span][: p.size] = values
            if p not in gradients:
                continue
            grad = _f32(gradients[p], "gradient for " + p.name).reshape(-1)
            payload["grad"][span][: p.size] = grad
            payload["active"][span] = 1
            active.append((p, span))
        self._input.write(payload.tobytes())
        self._status.write(bytes(4))
        for binding, buffer in enumerate(
            (self._input, self._current, self._candidate, self._status, self._values)
        ):
            buffer.bind_to_storage_buffer(binding)
        self._shader["count"].value = self._count
        for name, value in self._shader_config(config).items():
            self._shader[name].value = value
        self._shader.run(group_x=(self._count + 63) // 64)
        self._ctx.memory_barrier()
        if np.frombuffer(self._status.read(), dtype=np.uint32)[0]:
            raise FloatingPointError("nonfinite GPU optimizer update or exhausted step")
        result = np.frombuffer(self._values.read(), dtype=np.float32)
        for p, span in active:
            p.assign(result[span][: p.size].reshape(p.shape))
        self._current, self._candidate = self._candidate, self._current

    def state_dict(self):
        """Return independent JSON-compatible moments/counters, without weights."""
        self._check_open()
        data = np.frombuffer(self._current.read(), dtype=_SLOT)
        state = {}
        for p, span in zip(self.parameters, self._slices):
            block = data[span]
            if isinstance(self, GPUAdam):
                if not block["step"][0]:
                    continue
                state[p.name] = {
                    "step": int(block["step"][0]),
                    "exp_avg": block["m"][: p.size].reshape(p.shape).tolist(),
                    "exp_avg_sq": block["v"][: p.size].reshape(p.shape).tolist(),
                }
            elif block["step"][0]:
                state[p.name] = {
                    "momentum_buffer": block["m"][: p.size].reshape(p.shape).tolist()
                }
        return {
            "format_version": 1,
            "type": type(self).__name__,
            "hyperparameters": self._hyperparameters(),
            "parameters": {p.name: list(p.shape) for p in self.parameters},
            "state": state,
        }

    def load_state_dict(self, state):
        """Validate and stage all settings/state before replacing live buffers."""
        self._check_open()
        if (
            not isinstance(state, Mapping)
            or type(state.get("format_version")) is not int
            or state["format_version"] != 1
            or state.get("type") != type(self).__name__
            or state.get("parameters")
            != {p.name: list(p.shape) for p in self.parameters}
        ):
            raise ValueError("incompatible optimizer state or parameters")
        config = self._validate_config(state.get("hyperparameters"))
        entries = state.get("state")
        if not isinstance(entries, Mapping):
            raise ValueError("invalid optimizer buffers")
        parameters = {
            p.name: (p, span) for p, span in zip(self.parameters, self._slices)
        }
        data = np.zeros(self._count, dtype=_SLOT)
        for name, entry in entries.items():
            if name not in parameters or not isinstance(entry, Mapping):
                raise ValueError("invalid optimizer parameter state")
            p, span = parameters[name]
            if not p.size:
                # JSON lists lose trailing dimensions after a zero-length axis;
                # the parameter schema above supplies those dimensions.
                entry = dict(entry)
                for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                    if key in entry:
                        value = _array(entry[key], key)
                        if not value.size:
                            entry[key] = value.reshape(p.shape)
            checked = self._validate_entry(entry, p.shape)
            if isinstance(self, GPUAdam):
                if checked["step"] > _MAX_STEP:
                    raise ValueError("Adam step exceeds GPU counter range")
                data["step"][span] = checked["step"]
                fields = (("m", "exp_avg"), ("v", "exp_avg_sq"))
            else:
                data["step"][span] = 1
                fields = (("m", "momentum_buffer"),)
            for field, key in fields:
                data[field][span][: p.size] = _f32(checked[key], key).reshape(-1)
        # Allocate/upload before touching the live state, including on GL errors.
        replacement = self._ctx.buffer(data.tobytes())
        old = self._current
        self._current = self._own(replacement)
        self._resources.remove(old)
        for name, value in config.items():
            setattr(self, name, value)
        old.release()

    def close(self):
        """Release owned buffers/program, leaving the caller's context alive."""
        if self._closed:
            return
        self._closed = True
        error = None
        for resource in reversed(self._resources):
            try:
                resource.release()
            except Exception as caught:
                if error is None:
                    error = caught
        self._resources.clear()
        if error is not None:
            raise error

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _config_scalars(config, names):
    if not isinstance(config, Mapping) or set(config) != set(names):
        raise ValueError("invalid optimizer hyperparameters")
    result = {name: _hyper(config[name], name) for name in names}
    for name, value in result.items():
        _f32(value, name)
    return result


class GPUSGD(_GPUOptimizer):
    """Native GPU SGD, classical momentum, Nesterov momentum and L2 decay."""

    _uniforms = "uniform float momentum; uniform bool nesterov, use_momentum;"
    _update = _SGD_UPDATE
    _hyperparameters = SGD._hyperparameters
    _validate_entry = SGD._validate_entry

    def __init__(
        self, canvas, parameters, lr=0.01, momentum=0, weight_decay=0, nesterov=False
    ):
        config = self._validate_config(
            dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        )
        self.__dict__.update(config)
        super().__init__(canvas, parameters)

    @staticmethod
    def _validate_config(config):
        if not isinstance(config, Mapping) or set(config) != {
            "lr",
            "momentum",
            "weight_decay",
            "nesterov",
        }:
            raise ValueError("invalid SGD hyperparameters")
        result = _config_scalars(
            {k: config[k] for k in ("lr", "momentum", "weight_decay")},
            ("lr", "momentum", "weight_decay"),
        )
        result["momentum"] = _hyper(config["momentum"], "momentum", upper=1)
        nesterov = config["nesterov"]
        if not isinstance(nesterov, bool) or (nesterov and not result["momentum"]):
            raise ValueError("Nesterov requires positive momentum")
        result["nesterov"] = nesterov
        return result

    def _shader_config(self, config):
        return dict(config, use_momentum=bool(config["momentum"]))


class GPUAdam(_GPUOptimizer):
    """Native GPU Adam with per-parameter bias correction and L2 decay."""

    _uniforms = "uniform float beta1, beta2, complement1, complement2, eps;"
    _update = _ADAM_UPDATE
    _hyperparameters = Adam._hyperparameters
    _validate_entry = Adam._validate_entry

    def __init__(
        self,
        canvas,
        parameters,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        config = self._validate_config(
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )
        self.__dict__.update(config)
        super().__init__(canvas, parameters)

    @staticmethod
    def _validate_config(config):
        if not isinstance(config, Mapping) or set(config) != {
            "lr",
            "betas",
            "eps",
            "weight_decay",
        }:
            raise ValueError("invalid Adam hyperparameters")
        result = _config_scalars(
            {k: config[k] for k in ("lr", "eps", "weight_decay")},
            ("lr", "eps", "weight_decay"),
        )
        if not _f32(result["eps"], "eps").item() > 0:
            raise ValueError("eps must be positive in float32")
        betas = config["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError("betas must contain two coefficients")
        result["betas"] = tuple(_hyper(b, "beta", upper=1) for b in betas)
        return result

    def _shader_config(self, config):
        beta1, beta2 = config["betas"]
        return dict(
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            eps=config["eps"],
            beta1=beta1,
            beta2=beta2,
            complement1=1 - beta1,
            complement2=1 - beta2,
        )
