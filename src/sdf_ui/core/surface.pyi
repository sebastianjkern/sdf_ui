from __future__ import annotations

from typing import Any

from .texture import TextureNode

class HeightTexture(TextureNode):
    kind: str
    def add(self, other: HeightTexture) -> HeightTexture: ...

class SurfaceNamespace:
    def ramp(
        self, origin: Any = (0.0, 0.0), slope: Any = (0.0, 1.0)
    ) -> HeightTexture: ...
    def gaussian(
        self, center: Any, sigma: Any, amplitude: Any = 1.0
    ) -> HeightTexture: ...
