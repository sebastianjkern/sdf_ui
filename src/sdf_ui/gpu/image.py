"""Owned WebGPU images with explicit, row-aligned CPU transfers."""

from __future__ import annotations

import numpy as np
import wgpu

from ..core.image import Image

_FORMATS = {
    "r32float": (1, np.dtype("float32")),
    "rgba32float": (4, np.dtype("float32")),
    "rgba8unorm": (4, np.dtype("uint8")),
}


class GPUImage(Image):
    """A context-owned 2D texture; CPU readback and upload are explicit."""

    def __init__(self, context, texture, *, size, format):
        if format not in _FORMATS:
            raise ValueError(f"Unsupported image format: {format}")
        self.context = context
        self.backend = context
        self.texture = texture
        self.size = tuple(map(int, size))
        self.format = format
        self.channels, self.numpy_dtype = _FORMATS[format]
        self.components = self.channels
        self.dtype = "f4" if self.numpy_dtype == np.dtype("float32") else "f1"
        self.shape = (self.size[1], self.size[0]) + (() if self.channels == 1 else (self.channels,))
        self.closed = False
        context._image_generation += 1
        self.generation = context._image_generation

    def _check_open(self):
        if self.closed:
            raise RuntimeError("GPU image is closed")
        self.context._check_open()

    def write(self, data):
        """Upload a contiguous image array in the texture's native format."""
        self._check_open()
        if isinstance(data, (bytes, bytearray, memoryview)):
            array = np.frombuffer(data, dtype=self.numpy_dtype)
            if array.size != int(np.prod(self.shape)):
                raise ValueError(f"Expected {int(np.prod(self.shape))} image values")
            array = array.reshape(self.shape)
        else:
            array = np.asarray(data, dtype=self.numpy_dtype)
        if array.shape != self.shape:
            raise ValueError(f"Expected image shape {self.shape}, got {array.shape}")
        array = np.ascontiguousarray(array)
        self.context.queue.write_texture(
            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, 0)},
            array,
            {
                "offset": 0,
                "bytes_per_row": self.size[0] * self.channels * self.numpy_dtype.itemsize,
                "rows_per_image": self.size[1],
            },
            (self.size[0], self.size[1], 1),
        )

    def read(self):
        """Copy pixels to CPU memory, respecting WebGPU's 256-byte row pitch."""
        self._check_open()
        stats = getattr(self.context, "_gpu_stats", None)
        if stats is not None:
            stats["image_readbacks"] += 1
        width, height = self.size
        row_bytes = width * self.channels * self.numpy_dtype.itemsize
        padded_row_bytes = (row_bytes + 255) & ~255
        buffer = self.context.device.create_buffer(
            label="image readback",
            size=padded_row_bytes * height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        try:
            encoder = self.context.device.create_command_encoder(label="image readback")
            encoder.copy_texture_to_buffer(
                {"texture": self.texture, "mip_level": 0, "origin": (0, 0, 0)},
                {"buffer": buffer, "offset": 0, "bytes_per_row": padded_row_bytes, "rows_per_image": height},
                (width, height, 1),
            )
            self.context.queue.submit([encoder.finish()])
            data = self.context.queue.read_buffer(buffer)
            rows = np.frombuffer(data, dtype=np.uint8).reshape(height, padded_row_bytes)
            packed = rows[:, :row_bytes].copy()
            return packed.view(self.numpy_dtype).reshape(self.shape).copy()
        finally:
            buffer.destroy()

    def read_bytes(self):
        """Return packed pixel bytes, excluding WebGPU row padding."""
        return self.read().tobytes(order="C")

    numpy = read

    def clone(self):
        self._check_open()
        return self.context.copy_image(self)

    def close(self):
        if not self.closed:
            self.texture.destroy()
            self.closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
