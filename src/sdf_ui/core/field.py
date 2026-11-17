"""Vector-field texture nodes and public field constructors."""

__docformat__ = "google"

from ..gpu.nodes.common import build
from .texture import TextureNode


class FieldTexture(TextureNode):
    """A floating-point image whose red and green channels hold XY vectors."""

    kind = "field"

    def scale(self, amount):
        """Scale every vector in this field by ``amount``."""
        return build("field_scale", self, amount=amount)

    def add(self, other):
        """Add a second vector field to this field."""
        return build("field_add", self, other)

    def visualize(self, scale=1.0, magnitude=1.0):
        """Color field direction by hue and vector magnitude by brightness."""
        return build("field_visualize", self, scale=scale, magnitude=magnitude)

    def contours(self, spacing=8.0, line_width=1.0, scale=1.0):
        """Draw contours of vector magnitude to inspect a field."""
        return build(
            "field_contours", self, spacing=spacing, line_width=line_width, scale=scale
        )


class FieldNamespace:
    """Factories for canvas-space vector fields."""

    def constant(self, vector=(0.0, 0.0)):
        """Create a field with one constant canvas-space displacement."""
        return build("constant_field", vector=vector)

    def radial(self, center, radius, strength=1.0):
        """Create a Gaussian falloff field flowing radially from a center."""
        return build("radial_field", center=center, radius=radius, strength=strength)

    def vortex(self, center, radius, strength=1.0):
        """Create a Gaussian falloff field swirling around a center."""
        return build("vortex_field", center=center, radius=radius, strength=strength)

    def turbulence(
        self,
        frequency=0.012,
        amplitude=24.0,
        octaves=4,
        lacunarity=2.0,
        gain=0.5,
        seed=0,
    ):
        """Create a deterministic fractal-noise vector field in canvas units."""
        return build(
            "turbulence_field",
            frequency=frequency,
            amplitude=amplitude,
            octaves=octaves,
            lacunarity=lacunarity,
            gain=gain,
            seed=seed,
        )
