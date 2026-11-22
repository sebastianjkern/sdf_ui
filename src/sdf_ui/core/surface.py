"""Scalar height surfaces and their constructors."""

__docformat__ = "google"

from ..gpu.nodes.common import build
from .texture import TextureNode


class HeightTexture(TextureNode):
    """A floating point image containing one height value per canvas pixel."""

    kind = "height"

    def add(self, other: "HeightTexture") -> "HeightTexture":
        """Add another height surface pointwise."""
        return build("height_add", self, other)


class SurfaceNamespace:
    """Factories for canvas-space scalar height surfaces."""

    def ramp(self, origin=(0.0, 0.0), slope=(0.0, 1.0)):
        """Create a plane whose height changes linearly along ``slope``."""
        return build("height_ramp", origin=origin, slope=slope)

    def gaussian(self, center, sigma, amplitude=1.0):
        """Create an anisotropic Gaussian height bump centered at ``center``."""
        return build("height_gaussian", center=center, sigma=sigma, amplitude=amplitude)
