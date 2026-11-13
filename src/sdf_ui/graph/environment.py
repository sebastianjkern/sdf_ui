"""Connect contextual policies to a compiled renderer and reward callback."""

import json
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from ..core.expressions import Expr
from ..learning.autodiff import Parameter
from ..learning.objective import Objective
from ..learning.parameters import ParameterStore
from .runtime import RenderGraph


def _check_graph(graph):
    if not isinstance(graph, RenderGraph):
        raise TypeError("graph must be a RenderGraph")
    graph.check()


def _validate_action_observation_names(actions, observations):
    names = actions + observations
    if not actions or len(set(names)) != len(names):
        raise ValueError("Action and observation bindings must be nonempty and unique")


def _validate_named_bindings(actions, observations):
    if not isinstance(actions, Mapping) or not actions:
        raise ValueError("actions must be a nonempty name-to-binding mapping")
    if observations is not None and not isinstance(observations, Mapping):
        raise TypeError("observations must be a name-to-binding mapping")


def _validate_binding_names(actions, observations):
    if set(actions) & set(observations):
        raise ValueError("Observation and action names must be distinct")
    for name in (*actions, *observations):
        if not isinstance(name, str) or not name:
            raise ValueError("Binding names must be nonempty strings")


def _validate_callbacks(objective, reward, action_transform):
    if objective is not None and not isinstance(objective, Objective):
        raise TypeError("objective must be an Objective")
    if reward is not None and not callable(reward):
        raise TypeError("reward must be callable")
    if action_transform is not None and not callable(action_transform):
        raise TypeError("action_transform must be callable")
    if objective is None and reward is None:
        raise ValueError("Provide an objective or a reward callback")


def _serialize_identity(identity):
    if identity is None:
        return None
    try:
        return json.dumps(
            identity, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Environment identity must be finite JSON-serializable data"
        ) from error


def _input_binding(value, graph):
    if not isinstance(value, Expr) or value.op != "input":
        return None
    if value.args[0] not in graph.inputs:
        raise ValueError(f"Unknown graph input {value.args[0]!r}")
    return "input", value.args[0]


def _named_parameter_binding(value, params, parameters):
    if not isinstance(value, str):
        return None
    parameter = next(
        (item for item in parameters if item.name == value and item.shape == ()), None
    )
    if parameter is None:
        return None
    kind = (
        "scalar"
        if isinstance(params, ParameterStore) and params.is_scalar(value)
        else "parameter"
    )
    return kind, value


class RenderEnvironment:
    """A callable reward function for ``PolicyGradientTrainer.train``.

    Bind each action and optional observation component to a named scalar
    graph parameter. An objective supplies the default reward ``-loss``;
    alternatively ``reward(observation, action, execution)`` can inspect
    the live execution and return a finite scalar. The callback must not
    retain the execution after returning.
    """

    def __init__(
        self,
        graph,
        actions,
        *,
        observations=(),
        objective=None,
        reward=None,
        action_transform=None,
    ):
        _check_graph(graph)
        self.graph = graph
        self.actions = tuple(actions)
        self.observations = tuple(observations)
        _validate_action_observation_names(self.actions, self.observations)
        names = self.actions + self.observations
        self._parameters = {p.name: p for p in graph.parameters}
        for name in names:
            if name not in self._parameters or self._parameters[name].shape != ():
                raise ValueError(f"Binding {name!r} must name a scalar graph parameter")
        _validate_callbacks(objective, reward, action_transform)
        self.objective = objective
        self.reward = reward
        self.action_transform = action_transform

    def __call__(self, observation, action):
        self.graph.require_idle()
        raw_action = action
        action = np.asarray(
            action if self.action_transform is None else self.action_transform(action),
            dtype=float,
        )
        observation = np.asarray(observation, dtype=float)
        if action.shape != (len(self.actions),) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite vector matching its bindings")
        if self.observations and (
            observation.shape != (len(self.observations),)
            or not np.isfinite(observation).all()
        ):
            raise ValueError(
                "Observation must be a finite vector matching its bindings"
            )
        for name, value in zip(self.actions, action):
            self._parameters[name].assign(value)
        for name, value in zip(self.observations, observation):
            self._parameters[name].assign(value)
        with self.graph.run(self.objective) as execution:
            value = (
                -execution.loss
                if self.reward is None
                else self.reward(observation, raw_action, execution)
            )
            value = np.asarray(value, dtype=float)
            if value.shape != () or not np.isfinite(value).item():
                raise ValueError("Reward must be a finite scalar")
            return float(value)


class NamedValues(Mapping):
    """Read-only named scalar values passed to high-level reward callbacks."""

    def __init__(self, values):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def as_dict(self):
        return dict(self._values)


class NamedRenderEnvironment:
    """Bind named observations/actions to scalar parameters and graph inputs."""

    def __init__(
        self,
        graph,
        *,
        actions,
        observations=None,
        objective=None,
        reward=None,
        action_transform=None,
        identity=None,
    ):
        _check_graph(graph)
        _validate_named_bindings(actions, observations)
        self.graph = graph
        self.actions = MappingProxyType(dict(actions))
        self.observations = MappingProxyType(dict(observations or {}))
        _validate_binding_names(self.actions, self.observations)
        self._bindings = {
            key: self._binding(value)
            for key, value in (*self.actions.items(), *self.observations.items())
        }
        targets = list(self._bindings.values())
        if len(set(targets)) != len(targets):
            raise ValueError(
                "Each observation and action must bind a different input or parameter"
            )
        _validate_callbacks(objective, reward, action_transform)
        identity = _serialize_identity(identity)
        self.identity = identity
        self.objective, self.reward, self.action_transform = (
            objective,
            reward,
            action_transform,
        )

    def schema(self):
        """Return ordered bindings and the declared reward/environment identity."""
        params = self.graph.params
        scalar_schema = params.schema() if isinstance(params, ParameterStore) else {}
        return {
            "observations": [
                [name, *self._bindings[name]] for name in self.observations
            ],
            "actions": [[name, *self._bindings[name]] for name in self.actions],
            "parameters": {
                name: scalar_schema[name]
                for kind, name in self._bindings.values()
                if kind in {"scalar", "parameter"} and name in scalar_schema
            },
            "identity": self.identity,
        }

    def _binding(self, value):
        params = self.graph.params
        binding = _input_binding(value, self.graph)
        if binding is not None:
            return binding
        if isinstance(params, ParameterStore):
            name = params.scalar_name(value)
            if name is not None:
                return ("scalar", name)
        if isinstance(value, Expr) and value.op == "param":
            value = value.args[0]
        if isinstance(value, Parameter):
            if not any(value is p for p in self.graph.parameters):
                raise ValueError("Parameter binding must belong to this graph")
            value = value.name
        binding = _named_parameter_binding(value, params, self.graph.parameters)
        if binding is not None:
            return binding
        raise ValueError("Bindings must identify a scalar graph parameter or input")

    @staticmethod
    def _values(values, names, label):
        if not isinstance(values, Mapping) or set(values) != set(names):
            raise ValueError(f"{label} names must match the environment bindings")
        result = {}
        for name in names:
            value = values[name]
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{label} values must be finite scalars")
            result[name] = float(value)
        return result

    def encode_observation(self, observation):
        values = self._values(observation, self.observations, "Observation")
        return [values[name] for name in self.observations]

    def decode_action(self, action):
        if self.action_transform is not None:
            plain = (
                tuple(float(v) for v in action)
                if isinstance(action, (list, tuple, np.ndarray))
                else int(action)
            )
            action = self.action_transform(plain)
            return self._values(action, self.actions, "Action")
        array = np.asarray(action, dtype=float)
        if array.shape != (len(self.actions),) or not np.isfinite(array).all():
            raise ValueError("Policy action dimension must match the action bindings")
        return dict(zip(self.actions, (float(v) for v in array)))

    def reward_vector(self, observation, action):
        obs = np.asarray(observation, dtype=float)
        if obs.shape != (len(self.observations),) or not np.isfinite(obs).all():
            raise ValueError(
                "Policy observation dimension must match the observation bindings"
            )
        return self(
            dict(zip(self.observations, (float(v) for v in obs))),
            self.decode_action(action),
        )

    def act(self, policy, observation):
        """Return a deterministic policy action as named Python scalars."""
        if not callable(getattr(policy, "deterministic_action", None)):
            raise TypeError("policy must provide deterministic_action")
        values = policy.deterministic_action(self.encode_observation(observation))
        return NamedValues(self.decode_action(values))

    def __call__(self, observation, action):
        self.graph.require_idle()
        obs = self._values(observation, self.observations, "Observation")
        acts = self._values(action, self.actions, "Action")
        params = {p.name: p for p in self.graph.parameters}
        inputs, assignments = {}, []
        for key, value in (*obs.items(), *acts.items()):
            kind, name = self._bindings[key]
            if kind == "input":
                inputs[name] = value
            else:
                raw = (
                    self.graph.params.raw_value(name, value)
                    if kind == "scalar"
                    else value
                )
                assignments.append((params[name], raw))
        if inputs:
            self.graph.set_inputs(inputs)
        for parameter, raw in assignments:
            parameter.assign(raw)
        with self.graph.run(self.objective) as execution:
            result = (
                -execution.loss
                if self.reward is None
                else self.reward(NamedValues(obs), NamedValues(acts), execution)
            )
            result = np.asarray(result, dtype=float)
            if result.shape != () or not np.isfinite(result).item():
                raise ValueError("Reward must be a finite scalar")
            return float(result)
