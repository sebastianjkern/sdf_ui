"""Signed-distance texture helpers and boolean composition aliases."""

__docformat__ = "google"

from ..gpu.nodes.common import build
from .color import _transparent_ink
from .surface import HeightTexture
from .texture import WarpableTextureNode


class SDFTexture(WarpableTextureNode):
    kind = "sdf"
    _warp_operation = "warp_sdf"

    def fill(
        self,
        fg_color,
        bg_color=(0.0, 0.0, 0.0, 1.0),
        inflate=None,
        inner=None,
        outer=None,
    ):
        """Fill this signed distance field with foreground and background colors.

        Text uses a centered one-unit antialiasing ramp by default.
        """
        is_text = self.op == "text"
        if inflate is None:
            inflate = 0.2 if is_text else 0.0
        if outer is None:
            outer = 0.5 if is_text and inner is None else 0.0
        return build(
            "fill",
            self,
            fg_color=fg_color,
            bg_color=bg_color,
            inflate=inflate,
            inner=(-0.5 if is_text else -3 / 2) if inner is None else inner,
            outer=outer,
        )

    def ink(self, fg_color, inflate=0.0, *, inner=-0.5, outer=0.5):
        """Render this SDF as transparent ink with color-preserving antialiasing.

        The transparent background keeps the foreground RGB values, preventing
        straight-alpha compositing from darkening antialiased edges. This is
        useful for text and math glyphs as well as other thin SDF marks.
        """
        return self.fill(
            fg_color,
            _transparent_ink(fg_color),
            inflate=inflate,
            inner=inner,
            outer=outer,
        )

    def stroke(self, width, fg_color, bg_color=(0.0, 0.0, 0.0, 0.0), inflate=0.0):
        """Render a centered stroke as a color layer around this SDF contour."""
        return build(
            "stroke",
            self,
            width=width,
            fg_color=fg_color,
            bg_color=bg_color,
            inflate=inflate,
        )

    def ink_linear(self, rgba):
        """Paint this SDF with linear RGBA, allowing RGB values above one."""
        return build("hdr_ink", self, rgba)

    def light_surface(
        self,
        height_field: HeightTexture,
        fg_color,
        *,
        light_dir=(-9 / 20, -13 / 20, 0.7),
        ambient=0.35,
        diffuse=0.75,
        specular=0.2,
        shininess=24.0,
    ):
        """Light an independent canvas-space height field through this silhouette."""
        return build(
            "light_surface",
            self,
            height_field,
            fg_color=fg_color,
            light_dir=light_dir,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            shininess=shininess,
        )

    def fill_linear(self, rgba, background, *, inner=0.0, outer=1.0):
        """Interpolate two linear RGBA colors over a signed distance interval."""
        return build("hdr_fill", self, rgba, background, inner, outer)

    def domain_warp(self, field, strength=1.0):
        """Displace this shape's domain with a vector field."""
        return self.warp(field, strength=strength)

    def turbulence(
        self,
        amplitude=18.0,
        frequency=0.012,
        octaves=4,
        lacunarity=2.0,
        gain=0.5,
        seed=0,
    ):
        """Apply deterministic fractal-noise displacement to this SDF."""
        from .field import FieldNamespace

        vector_field = FieldNamespace().turbulence(
            frequency=frequency,
            amplitude=amplitude,
            octaves=octaves,
            lacunarity=lacunarity,
            gain=gain,
            seed=seed,
        )
        return self.warp(vector_field)

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __xor__(self, other):
        """Return the symmetric difference of this SDF and another SDF."""
        return self.xor(other)

    def xor(self, other):
        """Combine two SDFs, keeping points inside exactly one of them."""
        return self.union(other).subtract(self.intersection(other))

    def intersection(self, other):
        """Return the shared interior of this SDF and another SDF."""
        return build("intersection", self, other)

    def intersect(self, other):
        """Compatibility alias for :meth:`intersection`."""
        return self.intersection(other)

    def mask(
        self, inflate=0.0, color0=(0.0, 0.0, 0.0, 1.0), color1=(1.0, 1.0, 1.0, 1.0)
    ):
        return build(
            "generate_mask", self, inflate=inflate, color0=color0, color1=color1
        )

    def debug_seams(
        self,
        fg_color=(1.0, 0.25, 0.12, 1.0),
        bg_color=(0.035, 0.045, 0.065, 1.0),
        spacing=8.0,
        line_width=1.0,
    ):
        """Color distance contours to inspect seams and field transitions."""
        return self.isolines(
            fg_color, bg_color, spacing=spacing, line_width=line_width, feather=0.5
        )

    def debug_distance(self, range=32.0, show_contours=True, spacing=8.0):
        """Show signed distance values as a blue-to-orange heatmap."""
        heatmap = build("field_debug", self, range=range)
        if not show_contours:
            return heatmap
        contours = self.isolines(
            (0.08, 0.08, 0.09, 0.9),
            (0.0, 0.0, 0.0, 0.0),
            spacing=spacing,
            line_width=0.65,
            feather=0.25,
        )
        return heatmap.alpha_overlay(contours)

    def debug_boolean(
        self,
        other,
        operation="union",
        fg_color=(1.0, 0.25, 0.12, 1.0),
        spacing=8.0,
        line_width=1.0,
        range=32.0,
    ):
        """Render boolean result contours for inspecting shape transitions."""
        operations = {
            "union": self.union,
            "intersection": self.intersection,
            "subtract": self.subtract,
        }
        try:
            result = operations[operation](other)
        except KeyError as exc:
            choices = ", ".join(operations)
            raise ValueError(
                f"Unknown boolean operation {operation!r}; choose {choices}"
            ) from exc
        heatmap = result.debug_distance(range=range, show_contours=False)
        boundary = result.isolines(
            fg_color,
            (0.0, 0.0, 0.0, 0.0),
            spacing=spacing,
            line_width=line_width,
            feather=0.5,
        )
        return heatmap.alpha_overlay(boundary)
