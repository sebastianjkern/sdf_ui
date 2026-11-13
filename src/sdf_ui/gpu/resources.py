"""Owned WebGPU buffers used by graph training and reductions."""

from numbers import Integral


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


class GPUBuffer:
    """Storage buffer with explicit host transfers and lifecycle."""

    def __init__(self, context, buffer):
        self.context = context
        self.buffer = buffer
        self.size = buffer.size
        self.closed = False

    def _check_open(self):
        if self.closed:
            raise RuntimeError("GPU buffer is closed")
        self.context._check_open()

    def write(self, data, offset=0):
        self._check_open()
        offset = _integer(offset, "offset")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("buffer data must be bytes-like")
        payload = bytes(data)
        if offset % 4 or len(payload) % 4:
            raise ValueError("WebGPU buffer writes require four-byte alignment")
        if offset + len(payload) > self.size:
            raise ValueError("WebGPU buffer write exceeds its allocation")
        if payload:
            self.context.queue.write_buffer(self.buffer, offset, payload)

    def read(self, size=None, offset=0):
        self._check_open()
        offset = _integer(offset, "offset")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        if size is not None:
            size = _integer(size, "size")
        size = self.size - offset if size is None else size
        if size < 0 or offset % 4 or size % 4 or offset + size > self.size:
            raise ValueError("WebGPU buffer read has invalid bounds or alignment")
        return bytes(self.context.queue.read_buffer(self.buffer, offset, size))

    def clear(self):
        self._check_open()
        encoder = self.context.device.create_command_encoder(label="clear buffer")
        encoder.clear_buffer(self.buffer, 0, self.size)
        self.context.queue.submit([encoder.finish()])

    def close(self):
        if not self.closed:
            self.buffer.destroy()
            self.closed = True

    release = close

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
