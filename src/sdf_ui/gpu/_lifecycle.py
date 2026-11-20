"""Shared GPU owner lifecycle checks."""


def check_context_open(owner_closed, owner_message, context):
    if owner_closed:
        raise RuntimeError(owner_message)
    context._check_open()


def check_optimizer_open(optimizer):
    check_context_open(
        optimizer._closed, "GPU optimizer is closed", optimizer.canvas
    )


def check_image_open(image):
    if image.closed:
        raise RuntimeError("GPU image is closed")
    image.context._check_open()


class ContextResource:
    """Base lifecycle check for an image or buffer owned by a context."""

    _resource_name = "GPU resource"

    def _check_open(self):
        if self.closed:
            raise RuntimeError(f"GPU {self._resource_name} is closed")
        self.context._check_open()
