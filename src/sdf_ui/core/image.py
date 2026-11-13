"""Backend-neutral ownership contract for materialized images."""

from abc import ABC, abstractmethod


def shape_size(shape):
    """Number of scalar components in an image port or uniform shape."""
    size = 1
    for extent in shape:
        size *= extent
    return size


class Image(ABC):
    """An owned, explicitly readable image supplied by a rendering backend."""

    @abstractmethod
    def read(self):
        """Copy image data to host memory."""

    @abstractmethod
    def close(self):
        """Release owned resources."""
