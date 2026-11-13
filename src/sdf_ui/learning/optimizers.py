"""CPU SGD and Adam optimizers for learning parameters."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .autodiff import Parameter, _array, _finite


def _hyper(
    value: Any, name: str, upper: Optional[float] = None, positive: bool = False
) -> float:
    if isinstance(value, (bool, str)) or np.ndim(value) != 0:
        raise ValueError("{} must be a finite scalar".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be a finite scalar".format(name)) from None
    if (
        not np.isfinite(result)
        or result < 0
        or (positive and result == 0)
        or (upper is not None and result >= upper)
    ):
        raise ValueError("invalid {}".format(name))
    return result


class _Optimizer(ABC):
    def __init__(self, parameters: Iterable[Parameter]):
        if isinstance(parameters, Mapping):
            parameters = parameters.values()
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("optimizer requires at least one parameter")
        if not all(isinstance(p, Parameter) for p in self.parameters):
            raise TypeError("optimizers require Parameter objects")
        if not all(p.trainable for p in self.parameters):
            raise ValueError("optimizers accept trainable parameters only")
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("optimizer parameter names must be unique")
        self._state = {}  # type: Dict[str, Dict[str, Any]]

    @abstractmethod
    def _hyperparameters(self) -> Dict[str, Any]:
        """Return the concrete optimizer's serializable configuration."""
        raise TypeError("Use SGD or Adam")

    @abstractmethod
    def _validate_entry(
        self, entry: Mapping[str, Any], shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Validate a concrete optimizer's checkpoint buffers."""
        raise TypeError("Use SGD or Adam")

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def _gradients(self) -> Sequence[Tuple[Parameter, np.ndarray]]:
        gradients = []
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            grad = _array(parameter.grad, "gradient for " + parameter.name)
            if grad.shape != parameter.shape:
                raise ValueError("gradient shape mismatch for " + parameter.name)
            gradients.append((parameter, grad))
        if not gradients:
            raise RuntimeError("no gradients; call backward() before step()")
        return gradients

    def state_dict(self) -> Dict[str, Any]:
        """Return independent, JSON-compatible optimizer state (without weights)."""
        return {
            "format_version": 1,
            "type": type(self).__name__,
            "hyperparameters": self._hyperparameters(),
            "parameters": {p.name: list(p.shape) for p in self.parameters},
            "state": {
                name: {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in state.items()
                }
                for name, state in self._state.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate the entire state before replacing any optimizer settings."""
        if (
            not isinstance(state, Mapping)
            or state.get("format_version") != 1
            or state.get("type") != type(self).__name__
        ):
            raise ValueError("incompatible optimizer state")
        if state.get("parameters") != {p.name: list(p.shape) for p in self.parameters}:
            raise ValueError("optimizer state parameter names or shapes differ")
        config = state.get("hyperparameters")
        if not isinstance(config, Mapping):
            raise ValueError("invalid optimizer hyperparameters")
        try:
            candidate = type(self)(self.parameters, **dict(config))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid optimizer hyperparameters") from error
        raw_state = state.get("state")
        if not isinstance(raw_state, Mapping):
            raise ValueError("invalid optimizer buffers")
        parameters = {p.name: p for p in self.parameters}
        new_state = {}
        for name, entry in raw_state.items():
            if name not in parameters or not isinstance(entry, Mapping):
                raise ValueError("invalid optimizer parameter state")
            new_state[name] = self._validate_entry(entry, parameters[name].shape)
        for key, value in candidate._hyperparameters().items():
            setattr(self, key, value)
        self._state = new_state


class SGD(_Optimizer):
    """SGD with optional classical momentum, Nesterov momentum and L2 decay.

    Disconnected parameters (grad is None) are left unchanged, including decay
    and momentum. A step with no gradients at all raises RuntimeError.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(parameters)
        self.lr = _hyper(lr, "lr")
        self.momentum = _hyper(momentum, "momentum", upper=1.0)
        self.weight_decay = _hyper(weight_decay, "weight_decay")
        if not isinstance(nesterov, bool) or (nesterov and self.momentum == 0):
            raise ValueError("Nesterov requires positive momentum")
        self.nesterov = nesterov

    def _hyperparameters(self) -> Dict[str, Any]:
        return dict(
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )

    def _validate_entry(
        self, entry: Mapping[str, Any], shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        if set(entry) != {"momentum_buffer"}:
            raise ValueError("invalid SGD buffers")
        buffer = _array(entry["momentum_buffer"], "momentum buffer")
        if buffer.shape != shape:
            raise ValueError("momentum buffer shape mismatch")
        return {"momentum_buffer": buffer}

    def step(self) -> None:
        updates = []
        with np.errstate(all="ignore"):
            for parameter, grad in self._gradients():
                direction = grad + self.weight_decay * parameter._data
                state = {}
                if self.momentum:
                    previous = self._state.get(parameter.name, {}).get(
                        "momentum_buffer", np.zeros_like(direction)
                    )
                    buffer = _finite(
                        self.momentum * previous + direction, "momentum buffer"
                    )
                    state = {"momentum_buffer": buffer.copy()}
                    direction = (
                        direction + self.momentum * buffer if self.nesterov else buffer
                    )
                value = _finite(parameter._data - self.lr * direction, "SGD update")
                updates.append((parameter, value, state))
        for parameter, value, state in updates:
            parameter.assign(value)
            if state:
                self._state[parameter.name] = state


class Adam(_Optimizer):
    """Bias-corrected Adam with per-parameter steps and optional L2 decay."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters)
        self.lr = _hyper(lr, "lr")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas must contain two coefficients")
        self.betas = (
            _hyper(betas[0], "beta1", upper=1.0),
            _hyper(betas[1], "beta2", upper=1.0),
        )
        self.eps = _hyper(eps, "eps", positive=True)
        self.weight_decay = _hyper(weight_decay, "weight_decay")

    def _hyperparameters(self) -> Dict[str, Any]:
        return dict(
            lr=self.lr,
            betas=list(self.betas),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    def _validate_entry(
        self, entry: Mapping[str, Any], shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("invalid Adam buffers")
        step = entry["step"]
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, np.integer))
            or step < 1
        ):
            raise ValueError("invalid Adam step")
        m = _array(entry["exp_avg"], "Adam first moment")
        v = _array(entry["exp_avg_sq"], "Adam second moment")
        if m.shape != shape or v.shape != shape or np.any(v < 0):
            raise ValueError("invalid Adam moment shape or variance")
        return {"step": int(step), "exp_avg": m, "exp_avg_sq": v}

    def step(self) -> None:
        updates = []
        beta1, beta2 = self.betas
        with np.errstate(all="ignore"):
            for parameter, grad in self._gradients():
                grad = _finite(
                    grad + self.weight_decay * parameter._data, "Adam gradient"
                )
                previous = self._state.get(parameter.name, {})
                step = previous.get("step", 0) + 1
                m = _finite(
                    beta1 * previous.get("exp_avg", np.zeros_like(grad))
                    + (1 - beta1) * grad,
                    "Adam first moment",
                )
                v = _finite(
                    beta2 * previous.get("exp_avg_sq", np.zeros_like(grad))
                    + (1 - beta2) * grad * grad,
                    "Adam second moment",
                )
                direction = (m / (1 - beta1**step)) / (
                    np.sqrt(v / (1 - beta2**step)) + self.eps
                )
                value = _finite(parameter._data - self.lr * direction, "Adam update")
                updates.append(
                    (
                        parameter,
                        value,
                        {"step": step, "exp_avg": m.copy(), "exp_avg_sq": v.copy()},
                    )
                )
        for parameter, value, state in updates:
            parameter.assign(value)
            self._state[parameter.name] = state
