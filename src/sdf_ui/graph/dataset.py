"""Weighted multi-scene training and held-out evaluation on a shared context."""

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from ..learning.objective import Objective
from .runtime import GraphTrainer, RenderGraph


@dataclass(frozen=True)
class GraphCase:
    """One compiled scene and objective; cases share the same parameter leaves."""

    graph: RenderGraph
    objective: Objective
    name: str
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.graph, RenderGraph) or not isinstance(
            self.objective, Objective
        ):
            raise TypeError("A case requires a RenderGraph and Objective")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Case name must be nonempty")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, Real)
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("Case weight must be finite and positive")
        object.__setattr__(self, "weight", float(self.weight))


class GraphDatasetTrainer:
    """Train over scenes with one weighted-mean optimizer update per batch.

    All graphs must share a Canvas and exactly the same Parameter objects.
    Evaluation does not update parameters or optimizer state. Graphs and the
    Canvas remain caller-owned; close this trainer before closing them.
    """

    def __init__(self, cases, optimizer=None, *, lr=0.01):
        self.cases = tuple(cases)
        self._validate(self.cases)
        first = self.cases[0]
        self._trainer = GraphTrainer(first.graph, first.objective, optimizer, lr=lr)
        self.last_metrics = None

    def _validate(self, cases):
        if not cases or any(not isinstance(case, GraphCase) for case in cases):
            raise ValueError("Expected at least one GraphCase")
        if len({case.name for case in cases}) != len(cases):
            raise ValueError("Case names must be unique")
        first = cases[0].graph
        expected = {id(p) for p in first.parameters}
        for case in cases:
            case.graph.check()
            if (
                case.graph.canvas is not first.canvas
                or {id(p) for p in case.graph.parameters} != expected
            ):
                raise ValueError("Cases must share a Canvas and parameter leaves")

    @property
    def optimizer(self):
        return self._trainer.optimizer

    @property
    def steps(self):
        return self._trainer.steps

    def _cases(self, cases):
        cases = self.cases if cases is None else tuple(cases)
        self._validate(cases)
        first = self.cases[0].graph
        if cases[0].graph.canvas is not first.canvas or {
            id(p) for p in cases[0].graph.parameters
        } != {id(p) for p in first.parameters}:
            raise ValueError(
                "Evaluation cases must share the training Canvas and parameters"
            )
        return cases

    def evaluate(self, cases=None):
        from ..learning.training import normalized_case_weights

        self._trainer.check()
        cases = self._cases(cases)
        results = {}
        for case in cases:
            with case.graph.run(case.objective) as execution:
                results[case.name] = execution.loss
        weights = normalized_case_weights(cases)
        return {
            "loss": sum(
                weight * results[case.name] for case, weight in zip(cases, weights)
            ),
            "cases": results,
        }

    def step(self, cases=None):
        self._trainer.check()
        cases = self._cases(cases)
        result = self._trainer.step_batch(cases)
        self.last_metrics = result
        return result

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
        """Run weighted batches and optionally report each completed update."""
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

    def _case_signatures(self):
        result = []
        for case in self.cases:
            graph, objective = case.graph, case.objective
            graph.require_idle()
            result.append(
                {
                    "name": case.name,
                    "weight": case.weight,
                    "graph": graph.fingerprint,
                    "objective": graph.objective_signature(objective, portable=True),
                }
            )
        return result

    def state_dict(self):
        """Save parameters, optimizer state and all training-case identities."""
        return {
            "format_version": 2,
            "cases": self._case_signatures(),
            "case_inputs": {case.name: dict(case.graph.inputs) for case in self.cases},
            "trainer": self._trainer.state_dict(),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("format_version") != 2:
            raise ValueError("Invalid dataset trainer checkpoint")
        if state.get("cases") != self._case_signatures():
            raise ValueError("Checkpoint training cases differ")
        inputs = state.get("case_inputs", {})
        if not isinstance(inputs, dict) or set(inputs) != {
            case.name for case in self.cases
        }:
            raise ValueError("Checkpoint case inputs differ")
        for case in self.cases:
            saved = inputs[case.name]
            if (
                not isinstance(saved, dict)
                or set(saved) != set(case.graph.inputs)
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in saved.values()
                )
            ):
                raise ValueError("Invalid checkpoint case inputs")
        graph_inputs = {}
        for case in self.cases:
            saved = inputs[case.name]
            key = id(case.graph)
            if key in graph_inputs and graph_inputs[key] != saved:
                raise ValueError(
                    "Checkpoint gives conflicting inputs for a shared graph"
                )
            graph_inputs[key] = saved
        trainer_state = state.get("trainer")
        if (
            not isinstance(trainer_state, dict)
            or trainer_state.get("inputs") != inputs[self.cases[0].name]
        ):
            raise ValueError("Checkpoint trainer inputs differ from its graph case")
        self._trainer.load_state_dict(trainer_state)
        for case in self.cases:
            case.graph.set_inputs(inputs[case.name])
        self.last_metrics = None

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    def load(self, path):
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def close(self):
        self._trainer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
