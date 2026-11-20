"""Executable analytic VJPs for shader contracts, independent of GL execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ...gpu.shader_operation import ShaderOperation
from ..autodiff import Tensor


@dataclass(frozen=True)
class ShaderGradients:
    """Input pixel adjoints and pixel-reduced uniform adjoints for one pass."""

    inputs: object
    parameters: object

    def __post_init__(self):
        for field in ("inputs", "parameters"):
            values = {}
            for name, value in getattr(self, field).items():
                array = np.array(value, dtype=float, copy=True)
                array.flags.writeable = False
                values[name] = array
            object.__setattr__(self, field, MappingProxyType(values))


def _implementation(contract):
    return ShaderOperation(contract.name, contract.path, contract).reference_rule()


def _validated_arguments(contract, inputs, params):
    inputs = {} if inputs is None else dict(inputs)
    params = {} if params is None else dict(params)
    expected_inputs = {argument.name for argument in contract.inputs}
    if set(inputs) != expected_inputs:
        raise ValueError(
            f"Shader '{contract.name}' needs inputs "
            f"{tuple(argument.name for argument in contract.inputs)}"
        )
    expected_params = {argument.name for argument in contract.parameters}
    unknown = set(params) - expected_params
    if unknown:
        raise ValueError(f"Unknown shader uniforms: {sorted(unknown)}")
    return inputs, params


def _convert_arguments(contract, context, defaults, inputs, params, independent):
    from ..render import _tensor

    textures, uniforms = {}, {}
    groups = (
        (contract.inputs, inputs, textures, True),
        (contract.parameters, params, uniforms, False),
    )
    for arguments, source, target, is_image in groups:
        for argument in arguments:
            spatial_shape = (context.size[1], context.size[0]) if is_image else ()
            expected = spatial_shape + argument.shape
            value = source.get(argument.name, defaults.get(argument.name, np.zeros(expected)))
            tensor = _tensor(value)
            if tensor.shape != expected:
                raise ValueError(
                    f"Shader argument '{argument.name}' has shape {tensor.shape}; expected {expected}"
                )
            target[argument.name] = _independent_tensor(
                tensor, contract, argument, is_image, independent
            )
    return textures, uniforms


def _independent_tensor(tensor, contract, argument, is_image, independent):
    if not independent:
        return tensor
    return Tensor(
        tensor.numpy(),
        requires_grad=is_image or argument.name not in contract.zero_gradient_parameters,
    )


def _prepare(contract, size, inputs, params, independent):
    from ..render import DifferentiableRenderer

    context = DifferentiableRenderer(size)
    rule, defaults = _implementation(contract)
    inputs, params = _validated_arguments(contract, inputs, params)
    textures, uniforms = _convert_arguments(
        contract, context, defaults, inputs, params, independent
    )
    return context, rule, textures, uniforms


def _outputs(contract, context, result):
    if len(contract.outputs) == 1:
        values = (result,)
    elif isinstance(result, (tuple, list)):
        values = tuple(result)
    else:
        raise TypeError(
            f"Shader '{contract.name}' must return {len(contract.outputs)} outputs"
        )
    if len(values) != len(contract.outputs):
        raise ValueError("Shader output count differs from its contract")
    for argument, value in zip(contract.outputs, values):
        expected = (context.size[1], context.size[0]) + argument.shape
        if not isinstance(value, Tensor):
            raise TypeError(f"Shader output '{argument.name}' must be a Tensor")
        if value.shape != expected:
            raise ValueError(
                f"Shader output '{argument.name}' has shape {value.shape}; expected {expected}"
            )
    return values


def evaluate_shader(contract, size, inputs=None, params=None):
    context, rule, textures, uniforms = _prepare(contract, size, inputs, params, False)
    values = _outputs(contract, context, rule(context, textures, uniforms))
    return values[0] if len(values) == 1 else values


def shader_vjp(contract, size, inputs, params, output_gradients):
    context, rule, textures, uniforms = _prepare(contract, size, inputs, params, True)
    values = _outputs(contract, context, rule(context, textures, uniforms))
    seeds = _output_seeds(contract, values, output_gradients)
    _backward_from_seeds(values, seeds)
    return ShaderGradients(_gradient_arrays(textures), _gradient_arrays(uniforms))


def _output_seeds(contract, values, output_gradients):
    if isinstance(output_gradients, Mapping):
        expected_names = {argument.name for argument in contract.outputs}
        if set(output_gradients) != expected_names:
            raise ValueError(
                f"Upstream gradient names must be {sorted(expected_names)}"
            )
        seeds = tuple(output_gradients[argument.name] for argument in contract.outputs)
    elif len(values) == 1:
        seeds = (output_gradients,)
    else:
        seeds = tuple(output_gradients)
    if len(seeds) != len(values):
        raise ValueError("One upstream gradient is required per shader output")
    return seeds


def _backward_from_seeds(values, seeds):
    terms = []
    for value, seed in zip(values, seeds):
        seed = Tensor(seed.numpy() if isinstance(seed, Tensor) else seed)
        if value.shape != seed.shape:
            raise ValueError(
                f"Upstream gradient shape {seed.shape} must match output {value.shape}"
            )
        terms.append((value * seed).sum())
    if terms:
        loss = sum(terms)
        if loss.requires_grad:
            loss.backward()


def _gradient_arrays(values):
    return {
        name: np.zeros(value.shape) if value.grad is None else value.grad
        for name, value in values.items()
    }
