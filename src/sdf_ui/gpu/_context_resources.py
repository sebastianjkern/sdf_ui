"""Helpers for wrapping and validating context-owned GPU resources."""

def _bind_resource(resource):
    from .image import GPUImage
    from .resources import GPUBuffer

    if isinstance(resource, GPUImage):
        resource._check_open()
        return resource.texture.create_view()
    if isinstance(resource, GPUBuffer):
        resource._check_open()
        return resource.buffer
    return resource


def _wrap_gpu_buffer(context, buffer):
    from .resources import GPUBuffer

    return GPUBuffer(context, buffer)


def _wrap_gpu_image(context, texture, width, height, format, origin, canvas_size):
    from .image import GPUImage

    return GPUImage(
        context, texture, size=(width, height), format=format,
        origin=origin, canvas_size=canvas_size,
    )


def _require_context_image(image, context):
    from .image import GPUImage

    if not isinstance(image, GPUImage) or image.context is not context:
        raise ValueError("Image must belong to this WebGPU context")
    image._check_open()


def _validated_image_region(context, source, destination, rect):
    from .image import GPUImage
    from .regions import normalize_rect

    if not isinstance(source, GPUImage) or not isinstance(destination, GPUImage):
        raise ValueError("Images must belong to this WebGPU context")
    if source.context is not context or destination.context is not context:
        raise ValueError("Images must belong to this WebGPU context")
    source._check_open()
    destination._check_open()
    return normalize_rect(rect, destination.canvas_size)
