"""Convenient ownership and diagnostics for render-graph training."""

from contextlib import ExitStack

from ..core.texture import MultiOutputResult, TextureNode
from ..learning.objective import Objective
from ..learning.parameters import ParameterStore
from .runtime import GraphTrainer


class TrainingSession:
    """Own a compiled graph, captured targets, and a lazily created trainer."""

    DEFAULT_OUTPUT = "output"

    def __init__(self, canvas, outputs, params=None, *, size=None, inputs=None):
        self.canvas = canvas
        self._stack = ExitStack()
        if isinstance(outputs, (TextureNode, MultiOutputResult)):
            outputs = {self.DEFAULT_OUTPUT: outputs}
        self.graph = self._stack.enter_context(
            canvas.compile(outputs, params, size=size, inputs=inputs)
        )
        self.outputs = tuple(self.graph.plan.outputs)
        self.objective = Objective()
        self.trainer = None
        self._lr = None
        self.closed = False

    def _check(self):
        if self.closed:
            raise RuntimeError("Training session is closed")
        self.graph.check()

    def _select_output(self, output):
        if output is None:
            if len(self.outputs) != 1:
                raise ValueError(
                    "Specify output= when selecting from a session with multiple outputs"
                )
            return self.outputs[0]
        if output not in self.outputs:
            raise ValueError(
                f"Unknown output '{output}'. Available outputs: {', '.join(self.outputs)}"
            )
        return output

    def match(
        self,
        target,
        *,
        output=None,
        kind="mse",
        weight=1.0,
        name=None,
        mask=None,
        normalization="all",
        delta=1.0,
    ):
        """Match one output to a target scene or array.

        A single-output session infers its output name. Multi-output sessions
        require ``output=...`` so each target is associated explicitly.
        Scene targets are captured and owned by the session.
        """
        self._check()
        output = self._select_output(output)
        with ExitStack() as pending:
            if isinstance(target, (TextureNode, MultiOutputResult)):
                target = pending.enter_context(
                    self.canvas.capture(
                        target,
                        self.graph.params,
                        size=self.graph.size,
                        inputs=self.graph.inputs,
                    )
                )
            self.objective.pixel(
                output,
                target,
                kind=kind,
                weight=weight,
                name=name,
                mask=mask,
                normalization=normalization,
                delta=delta,
            )
            self._stack.enter_context(pending.pop_all())
        return self

    def geometry(self, callback, **options):
        self._check()
        self.objective.geometry(callback, **options)
        return self

    def fit(
        self,
        steps,
        *,
        lr=None,
        callback=None,
        validation=None,
        patience=None,
        min_delta=0.0,
        restore_best=True,
        validation_interval=1,
    ):
        self._check()
        if self.trainer is None:
            self._lr = 0.01 if lr is None else lr
            self.trainer = self._stack.enter_context(
                GraphTrainer(self.graph, self.objective, lr=self._lr)
            )
        elif lr is not None and lr != self._lr:
            raise ValueError(
                "Learning rate is fixed for this session; create another session to change it"
            )
        return self.trainer.fit(
            steps,
            callback=callback,
            validation=validation,
            patience=patience,
            min_delta=min_delta,
            restore_best=restore_best,
            validation_interval=validation_interval,
        )

    def evaluate(self):
        self._check()
        with self.graph.run(self.objective) as execution:
            return {"loss": execution.loss, "terms": dict(execution.metrics)}

    def report(self, *, layout=None):
        """Describe the current graph without reading rendered images."""
        self._check()
        params = self.graph.params
        values = {}
        for parameter in self.graph.parameters:
            if isinstance(params, ParameterStore) and params.is_scalar(parameter.name):
                values[parameter.name] = params.value(parameter.name)
            else:
                value = parameter.numpy().tolist()
                values[parameter.name] = value
        return {
            "device": self.graph.backend.device_info,
            "size": self.graph.size,
            "parameters": values,
            "disconnected_parameters": None
            if self.trainer is None or self.trainer.steps == 0
            else [p.name for p in self.graph.parameters if p.grad is None],
            "loss_terms": [
                term.name
                for term in (
                    *self.objective.pixel_losses,
                    *self.objective.geometry_losses,
                )
            ],
            "last_step": None if self.trainer is None else self.trainer.last_metrics,
            "violations": None if layout is None else layout.violations(params),
        }

    def save(self, path, *, output=None):
        """Save a trained output, inferring its name for single-output sessions."""
        self._check()
        output = self._select_output(output)
        return self.graph.save(output, path)

    def save_checkpoint(self, path):
        self._check()
        if self.trainer is None:
            raise RuntimeError("Train before saving a checkpoint")
        self.trainer.save(path)

    def load_checkpoint(self, path, *, lr=None):
        """Restore a checkpoint into a matching session and objective."""
        self._check()
        if self.trainer is None:
            self.fit(0, lr=lr)
        elif lr is not None and lr != self._lr:
            raise ValueError("Learning rate is fixed for this session")
        self.trainer.load(path)
        self._lr = self.trainer.optimizer.lr

    def close(self):
        if not self.closed:
            self.closed = True
            self._stack.close()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()
