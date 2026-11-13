"""Multiple image/field loss roots and scalar geometry terms for native graphs."""

import json
import marshal
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np

from ..core.image import Image


@dataclass(frozen=True)
class PixelLoss:
    output: str
    target: object
    kind: str = "mse"
    weight: float = 1.0
    delta: float = 1.0
    name: str = ""
    mask: object = None
    normalization: str = "all"


@dataclass(frozen=True)
class GeometryLoss:
    callback: object
    weight: float = 1.0
    name: str = "geometry"
    identity: object = None


def _weight(value):
    if (
        isinstance(value, bool)
        or not np.isscalar(value)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("Loss weights must be finite nonnegative scalars")
    with np.errstate(over="ignore"):
        finite_float32 = np.isfinite(np.float32(value))
    if not finite_float32:
        raise ValueError("Loss weights must be finite in float32")
    return float(value)


def _fixed_array(value, label, *, nonnegative=False):
    array = np.array(value, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(array)) or (nonnegative and np.any(array < 0)):
        raise ValueError(
            f"{label} must be finite"
            + (" and nonnegative" if nonnegative else " in float32")
        )
    # A bytes-backed array cannot be made writable again by callers.
    return np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)


class Objective:
    """An additive objective with normalized GPU pixel and CPU scalar losses.

    Targets are copied once and held fixed. Images and fields use the same
    loss API; every element, including alpha, participates in the mean. Scalar
    geometry callbacks build fresh Tensor expressions on each execution.

    An optional nonnegative ``mask`` weights each element. The default
    ``normalization="all"`` averages across the entire output; choose
    ``normalization="mask"`` to divide by the positive sum of the weights.
    An image mask may be scalar, match the output shape, or be a height-by-width
    CPU array shared across RGBA channels. GPU masks must match the output shape.
    """

    def __init__(self):
        self._pixel_losses = []
        self._geometry_losses = []
        self._version = 0

    @property
    def pixel_losses(self):
        return tuple(self._pixel_losses)

    @property
    def geometry_losses(self):
        return tuple(self._geometry_losses)

    @property
    def version(self):
        return self._version

    def _name(self, value, fallback):
        value = fallback if value is None else value
        if not isinstance(value, str) or not value:
            raise ValueError("Loss names must be nonempty strings")
        if value in {term.name for term in (*self.pixel_losses, *self.geometry_losses)}:
            raise ValueError(f"Duplicate loss name '{value}'")
        return value

    def pixel(
        self,
        output,
        target,
        *,
        kind="mse",
        weight=1.0,
        delta=1.0,
        name=None,
        mask=None,
        normalization="all",
    ):
        if kind not in {"mse", "mae", "huber"}:
            raise ValueError("Pixel loss kind must be mse, mae, or huber")
        if normalization not in {"all", "mask"}:
            raise ValueError("normalization must be 'all' or 'mask'")
        if not isinstance(output, str) or not output:
            raise ValueError("Pixel loss output must be a named graph root")
        if not np.isscalar(delta) or not np.isfinite(delta) or delta <= 0:
            raise ValueError("Huber delta must be finite and positive")
        if not isinstance(target, Image):
            target = _fixed_array(target, "Loss target")
        if mask is not None and not isinstance(mask, Image):
            mask = _fixed_array(mask, "Loss mask", nonnegative=True)
        name = self._name(name, f"{kind}:{output}:{len(self.pixel_losses)}")
        self._pixel_losses.append(
            PixelLoss(
                output,
                target,
                kind,
                _weight(weight),
                float(delta),
                name,
                mask,
                normalization,
            )
        )
        self._version += 1
        return self

    def mse(
        self, output, target, *, weight=1.0, name=None, mask=None, normalization="all"
    ):
        return self.pixel(
            output,
            target,
            kind="mse",
            weight=weight,
            name=name,
            mask=mask,
            normalization=normalization,
        )

    def mae(
        self, output, target, *, weight=1.0, name=None, mask=None, normalization="all"
    ):
        return self.pixel(
            output,
            target,
            kind="mae",
            weight=weight,
            name=name,
            mask=mask,
            normalization=normalization,
        )

    def huber(
        self,
        output,
        target,
        *,
        delta=1.0,
        weight=1.0,
        name=None,
        mask=None,
        normalization="all",
    ):
        return self.pixel(
            output,
            target,
            kind="huber",
            delta=delta,
            weight=weight,
            name=name,
            mask=mask,
            normalization=normalization,
        )

    def geometry(self, callback, *, weight=1.0, name=None, identity=None):
        if not callable(callback):
            raise TypeError("Geometry loss must be a zero-argument scalar callback")
        if identity is not None:
            try:
                identity = json.dumps(
                    identity, sort_keys=True, allow_nan=False, separators=(",", ":")
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Geometry identity must be finite JSON-serializable data"
                ) from error
        name = self._name(name, f"geometry:{len(self.geometry_losses)}")
        self._geometry_losses.append(
            GeometryLoss(callback, _weight(weight), name, identity)
        )
        self._version += 1
        return self

    def set_target(self, name, target):
        """Replace a pixel target between executions without recompiling shaders."""
        if not isinstance(target, Image):
            target = _fixed_array(target, "Loss target")
        for index, term in enumerate(self.pixel_losses):
            if term.name == name:
                self._pixel_losses[index] = replace(term, target=target)
                self._version += 1
                return self
        raise KeyError(f"Unknown pixel loss '{name}'")

    def set_mask(self, name, mask):
        """Replace or remove a pixel mask between graph executions."""
        if mask is not None and not isinstance(mask, Image):
            mask = _fixed_array(mask, "Loss mask", nonnegative=True)
        for index, term in enumerate(self.pixel_losses):
            if term.name == name:
                self._pixel_losses[index] = replace(term, mask=mask)
                self._version += 1
                return self
        raise KeyError(f"Unknown pixel loss '{name}'")

    def set_weight(self, name, weight):
        """Change an objective coefficient between executions."""
        weight = _weight(weight)
        for terms in (self._pixel_losses, self._geometry_losses):
            for index, term in enumerate(terms):
                if term.name == name:
                    terms[index] = replace(term, weight=weight)
                    self._version += 1
                    return self
        raise KeyError(f"Unknown loss '{name}'")

    def remove(self, name):
        """Remove a named term and invalidate executions using this objective."""
        for terms in (self._pixel_losses, self._geometry_losses):
            for index, term in enumerate(terms):
                if term.name == name:
                    del terms[index]
                    self._version += 1
                    return self
        raise KeyError(f"Unknown loss '{name}'")

    def signature(self, *, portable=False, target_overrides=None):
        result = self._pixel_signatures(portable, target_overrides)
        result.extend(self._geometry_signatures(portable))
        return sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()

    def _pixel_signatures(self, portable, target_overrides):
        return [
            self._pixel_signature(term, portable, target_overrides or {})
            for term in self.pixel_losses
        ]

    @staticmethod
    def _pixel_signature(term, portable, target_overrides):
        mask = _image_signature(
            term.mask, portable, target_overrides.get(("mask", term.name))
        )
        target = _image_signature(
            term.target, portable, target_overrides.get(term.name)
        )
        return (
            term.name,
            term.output,
            term.kind,
            term.weight,
            term.delta,
            target,
            mask,
            term.normalization,
        )

    def _geometry_signatures(self, portable):
        return [_geometry_signature(term, portable) for term in self.geometry_losses]


def _image_signature(value, portable, override):
    if value is None:
        return None
    if isinstance(value, Image):
        source = override or value
        if portable:
            return source.shape, sha256(source.read().tobytes()).hexdigest()
        return "gpu", value.shape, value.generation
    return value.shape, sha256(value.tobytes()).hexdigest()


def _geometry_signature(term, portable):
    if portable and term.identity is None:
        raise ValueError(
            f"Geometry loss '{term.name}' requires an explicit identity for checkpoints"
        )
    code = getattr(term.callback, "__code__", None)
    code = None if code is None else sha256(marshal.dumps(code)).hexdigest()
    return (
        term.name,
        term.weight,
        None
        if portable
        else getattr(term.callback, "__qualname__", type(term.callback).__name__),
        None if portable else code,
        term.identity,
    )
