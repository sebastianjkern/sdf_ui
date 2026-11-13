"""Shared training loop with optional validation-based early stopping."""

import math
from collections.abc import Mapping
from numbers import Real


def normalized_case_weights(cases):
    """Normalize positive finite weights without overflowing their sum."""
    weights = tuple(case.weight for case in cases)
    if not weights or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
        for value in weights
    ):
        raise ValueError("Case weights must be finite and positive")
    largest = max(weights)
    scaled = tuple(value / largest for value in weights)
    total = sum(scaled)
    return tuple(value / total for value in scaled)


def _validate_fit_options(
    steps, callback, validation, patience, min_delta, restore_best, validation_interval
):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Training steps must be a nonnegative integer")
    if callback is not None and not callable(callback):
        raise TypeError("Training callback must be callable")
    if validation is not None and not callable(validation):
        raise TypeError("validation must be callable")
    _validate_early_stopping(patience, validation, min_delta, restore_best)
    _validate_validation_interval(validation_interval)


def _validate_early_stopping(patience, validation, min_delta, restore_best):
    if patience is not None and (
        isinstance(patience, bool) or not isinstance(patience, int) or patience < 1
    ):
        raise ValueError("patience must be a positive integer")
    if patience is not None and validation is None:
        raise ValueError("patience requires validation")
    if (
        isinstance(min_delta, bool)
        or not isinstance(min_delta, Real)
        or not math.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError("min_delta must be a finite nonnegative number")
    if not isinstance(restore_best, bool):
        raise TypeError("restore_best must be a bool")


def _validate_validation_interval(validation_interval):
    if (
        isinstance(validation_interval, bool)
        or not isinstance(validation_interval, int)
        or validation_interval < 1
    ):
        raise ValueError("validation_interval must be a positive integer")


def _read_validation_loss(validation):
    score = validation()
    if isinstance(score, Mapping):
        score = score.get("loss")
    if (
        isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(score)
    ):
        raise ValueError("validation must return a finite scalar loss")
    return float(score)


def _run_fit_steps(
    trainer,
    steps,
    callback,
    validation,
    patience,
    min_delta,
    restore_best,
    validation_interval,
    step,
):
    history = []
    best = math.inf
    best_state = None
    stale = 0
    if validation is not None and steps:
        best = _read_validation_loss(validation)
        if restore_best:
            best_state = trainer.state_dict()

    for index in range(steps):
        result = trainer.step() if step is None else step()
        should_validate = _should_validate(
            validation, index, steps, validation_interval
        )
        if should_validate:
            best, best_state, stale = _record_validation(
                trainer,
                result,
                validation,
                best,
                best_state,
                stale,
                min_delta,
                restore_best,
            )
        history.append(result)
        if callback is not None:
            callback(result)
        if should_validate and patience is not None and stale >= patience:
            break
    _restore_best(trainer, best_state)
    return history


def _should_validate(validation, index, steps, interval):
    return validation is not None and (
        (index + 1) % interval == 0 or index + 1 == steps
    )


def _restore_best(trainer, best_state):
    if best_state is not None:
        trainer.load_state_dict(best_state)


def _record_validation(
    trainer, result, validation, best, best_state, stale, min_delta, restore_best
):
    score = _read_validation_loss(validation)
    result["validation_loss"] = score
    if score < best - min_delta:
        best = score
        stale = 0
        if restore_best:
            best_state = trainer.state_dict()
    else:
        stale += 1
    return best, best_state, stale


def fit_steps(
    trainer,
    steps,
    callback=None,
    *,
    validation=None,
    patience=None,
    min_delta=0.0,
    restore_best=True,
    validation_interval=1,
    step=None,
):
    """Run updates; compare periodic validation against the initial state.

    Validation must return a finite number or a mapping containing ``loss``.
    Best-state restoration includes parameters and optimizer state, and requires
    a checkpointable objective. Keep final test cases separate from validation.
    """
    _validate_fit_options(
        steps,
        callback,
        validation,
        patience,
        min_delta,
        restore_best,
        validation_interval,
    )
    return _run_fit_steps(
        trainer,
        steps,
        callback,
        validation,
        patience,
        min_delta,
        restore_best,
        validation_interval,
        step,
    )
