"""Real GL compute updates, CPU parity, transactional state and ownership."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sdf_ui.backends.opengl.optim import GPUSGD, GPUAdam
from sdf_ui.learning import SGD, Adam, Parameter, ParameterStore, Tensor


@pytest.fixture(scope="module")
def gl():
    moderngl = pytest.importorskip("moderngl")
    errors = []
    for options in ({"backend": "egl"}, {}):
        try:
            context = moderngl.create_standalone_context(require=430, **options)
            break
        except Exception as error:
            errors.append(str(error))
    else:
        pytest.skip("OpenGL 4.3 unavailable: " + "; ".join(errors))
    yield context
    context.release()


CASES = [
    (GPUSGD, SGD, {}),
    (GPUSGD, SGD, dict(lr=0.04, momentum=0.8, weight_decay=0.1)),
    (GPUSGD, SGD, dict(lr=0.04, momentum=0.8, weight_decay=0.1, nesterov=True)),
    (GPUSGD, SGD, dict(lr=0, momentum=0.5)),
    (GPUAdam, Adam, {}),
    (GPUAdam, Adam, dict(lr=0.04, betas=(0.7, 0.95), eps=1e-5, weight_decay=0.1)),
    (GPUAdam, Adam, dict(betas=(0, 0))),
    (GPUAdam, Adam, dict(betas=(0.9999, 0.99999))),
]


def assert_state_close(actual, expected):
    assert actual["parameters"] == expected["parameters"]
    assert actual["hyperparameters"] == expected["hyperparameters"]
    assert actual["state"].keys() == expected["state"].keys()
    for name, entry in actual["state"].items():
        assert entry.keys() == expected["state"][name].keys()
        for key, value in entry.items():
            if key == "step":
                assert value == expected["state"][name][key]
            else:
                np.testing.assert_allclose(
                    value, expected["state"][name][key], rtol=4e-6, atol=1e-6
                )


@pytest.mark.parametrize("gpu_type,cpu_type,config", CASES)
def test_several_steps_match_cpu_including_disconnections(
    gl, gpu_type, cpu_type, config
):
    rng = np.random.default_rng(21)
    values = [
        np.array(0.7),
        rng.normal(size=(5, 17)),
        np.array([1.3, -0.7]),
        np.empty(0),
    ]
    gpu_params = [Parameter(str(i), v) for i, v in enumerate(values)]
    cpu_params = [Parameter(str(i), v) for i, v in enumerate(values)]
    cpu = cpu_type(cpu_params, **config)
    with gpu_type(
        SimpleNamespace(_mgl_ctx=gl), ParameterStore(gpu_params), **config
    ) as gpu:
        assert gpu.parameters == tuple(gpu_params)
        for iteration in range(12):
            gpu.zero_grad()
            cpu.zero_grad()
            for i, (p, q) in enumerate(zip(gpu_params, cpu_params)):
                if i and (iteration + i) % 3 == 0:
                    continue
                p.grad = rng.normal(size=p.shape)
                q.grad = p.grad.copy()
            before = [(p.numpy(), p.version) for p in gpu_params]
            if iteration == 5:
                # Assigning a CPU leaf between steps must override stale GPU weights.
                gpu_params[1].assign(np.full((5, 17), 2.0))
                cpu_params[1].assign(np.full((5, 17), 2.0))
                before[1] = (gpu_params[1].numpy(), gpu_params[1].version)
            gpu.step()
            cpu.step()
            for p, q, (value, version) in zip(gpu_params, cpu_params, before):
                np.testing.assert_allclose(p.numpy(), q.numpy(), rtol=3e-6, atol=3e-7)
                if p.grad is None:
                    np.testing.assert_array_equal(p.numpy(), value)
                    assert p.version == version
                else:
                    assert p.version == version + 1
            assert_state_close(gpu.state_dict(), cpu.state_dict())


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
def test_backward_zero_grad_and_no_gradients(gl, gpu_type):
    p, unused = Parameter("p", [1.0, 2.0]), Parameter("unused", 3.0)
    with gpu_type(gl, [p, unused]) as optimizer:
        with pytest.raises(RuntimeError, match="no gradients"):
            optimizer.step()
        (p * p).sum().backward()
        optimizer.step()
        assert not np.array_equal(p.numpy(), [1.0, 2.0])
        assert unused.item() == 3.0
        assert "unused" not in optimizer.state_dict()["state"]
        optimizer.zero_grad()
        assert p.grad is None and unused.grad is None
        with pytest.raises(RuntimeError, match="no gradients"):
            optimizer.step()


@pytest.mark.parametrize(
    "gpu_type,config",
    [
        (GPUSGD, dict(lr=0.03, momentum=0.7, nesterov=True, weight_decay=0.02)),
        (GPUAdam, dict(lr=0.02, betas=(0.7, 0.96), eps=1e-6, weight_decay=0.02)),
    ],
)
def test_json_checkpoint_resume_and_identity(gl, gpu_type, config):
    p, q = Parameter("p", [2.0, 3.0]), Parameter("q", 0.7)
    with gpu_type(gl, [p, q], **config) as original:
        for i in range(5):
            p.grad = np.array([0.2, -0.1])
            q.grad = np.array(0.3) if i % 2 else None
            original.step()
        checkpoint = json.loads(json.dumps(original.state_dict(), allow_nan=False))
        r, s = Parameter("p", p.numpy()), Parameter("q", q.numpy())
        # Checkpoint identities are names/shapes, independent of collection order.
        with gpu_type(gl, {"q": s, "p": r}) as resumed:
            resumed.load_state_dict(checkpoint)
            assert resumed.parameters == (s, r)
            assert resumed.parameters[1] is r
            assert resumed.state_dict() == checkpoint
            for i in range(8):
                original.zero_grad()
                resumed.zero_grad()
                p.grad = r.grad = np.array([0.1 * i, -0.2])
                if i % 2:
                    q.grad = s.grad = np.array(-0.4)
                original.step()
                resumed.step()
                np.testing.assert_array_equal(p.numpy(), r.numpy())
                np.testing.assert_array_equal(q.numpy(), s.numpy())
                assert original.state_dict() == resumed.state_dict()
            saved = resumed.state_dict()
            key = "momentum_buffer" if gpu_type is GPUSGD else "exp_avg"
            saved["state"]["p"][key][0] = 999
            assert resumed.state_dict()["state"]["p"][key][0] != 999
            resumed.load_state_dict(checkpoint)
            checkpoint["state"]["p"][key][0] = 888
            assert resumed.state_dict()["state"]["p"][key][0] != 888


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
def test_loading_empty_state_resets_moments(gl, gpu_type):
    p = Parameter("p", 1.0)
    config = {"momentum": 0.9} if gpu_type is GPUSGD else {}
    with gpu_type(gl, [p], **config) as optimizer:
        initial = optimizer.state_dict()
        p.grad = np.array(0.2)
        optimizer.step()
        assert optimizer.state_dict()["state"]
        optimizer.load_state_dict(initial)
        assert optimizer.state_dict() == initial


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
def test_empty_multidimensional_parameters_resume(gl, gpu_type):
    parameters = [Parameter("a", np.empty((0, 3))), Parameter("b", np.empty((2, 0, 3)))]
    config = {"momentum": 0.8} if gpu_type is GPUSGD else {}
    with gpu_type(gl, parameters, **config) as optimizer:
        for p in parameters:
            p.grad = np.empty(p.shape)
        optimizer.step()
        state = json.loads(json.dumps(optimizer.state_dict()))
        optimizer.step()
        optimizer.load_state_dict(state)
        assert optimizer.state_dict() == state
        optimizer.step()
        assert all(p.numpy().shape == p.shape for p in parameters)
        if gpu_type is GPUAdam:
            assert all(s["step"] == 2 for s in optimizer.state_dict()["state"].values())


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
@pytest.mark.parametrize(
    "bad_gradient",
    [
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        np.array([1e100, 0.0]),
        np.array([1.0]),
        np.array([1j, 2j]),
        ["x", "y"],
    ],
)
def test_bad_gradient_preserves_all_values_versions_and_state(
    gl, gpu_type, bad_gradient
):
    a, b = Parameter("a", [1.0, 2.0]), Parameter("b", [3.0, 4.0])
    config = {"momentum": 0.8} if gpu_type is GPUSGD else {}
    with gpu_type(gl, [a, b], **config) as optimizer:
        a.grad = b.grad = np.array([0.1, -0.1])
        optimizer.step()
        before = optimizer.state_dict()
        values, versions = [p.numpy() for p in (a, b)], [p.version for p in (a, b)]
        b.grad = bad_gradient
        with pytest.raises((ValueError, TypeError)):
            optimizer.step()
        assert optimizer.state_dict() == before
        for p, value, version in zip((a, b), values, versions):
            np.testing.assert_array_equal(p.numpy(), value)
            assert p.version == version


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
def test_gpu_overflow_rejects_candidate_moments_and_recovers(gl, gpu_type):
    a, b = Parameter("a", 1.0), Parameter("b", 2.0)
    config = dict(lr=10, momentum=0.8) if gpu_type is GPUSGD else {}
    with gpu_type(gl, [a, b], **config) as optimizer:
        a.grad = b.grad = np.array(0.1)
        optimizer.step()
        before = optimizer.state_dict()
        values, versions = [p.numpy() for p in (a, b)], [p.version for p in (a, b)]
        b.grad = np.array(1e38 if gpu_type is GPUSGD else 1e30)
        with pytest.raises(FloatingPointError, match="nonfinite"):
            optimizer.step()
        assert optimizer.state_dict() == before
        for p, value, version in zip((a, b), values, versions):
            np.testing.assert_array_equal(p.numpy(), value)
            assert p.version == version
        c, d = Parameter("a", values[0]), Parameter("b", values[1])
        with gpu_type(gl, [c, d]) as control:
            control.load_state_dict(before)
            a.grad = b.grad = c.grad = d.grad = np.array(-0.2)
            optimizer.step()
            control.step()
            np.testing.assert_array_equal(a.numpy(), c.numpy())
            np.testing.assert_array_equal(b.numpy(), d.numpy())
            assert optimizer.state_dict() == control.state_dict()


def corrupt_state(state, kind, gpu_type):
    moment = "momentum_buffer" if gpu_type is GPUSGD else "exp_avg"
    if kind == "format":
        state["format_version"] = 2
    elif kind == "boolean_format":
        state["format_version"] = True
    elif kind == "type":
        state["type"] = "Adam"
    elif kind == "names":
        state["parameters"] = {"other": [2]}
    elif kind == "shapes":
        state["parameters"]["p"] = [1, 2]
    elif kind == "hyper_missing":
        del state["hyperparameters"]["lr"]
    elif kind == "hyper_extra":
        state["hyperparameters"]["unexpected"] = 1
    elif kind == "hyper_bad":
        state["hyperparameters"]["lr"] = -1
    elif kind == "state_type":
        state["state"] = []
    elif kind == "unknown":
        state["state"]["other"] = state["state"]["p"]
    elif kind == "entry_type":
        state["state"]["p"] = []
    elif kind == "entry_extra":
        state["state"]["p"]["unexpected"] = 1
    elif kind == "moment_shape":
        state["state"]["p"][moment] = [1]
    elif kind == "moment_nan":
        state["state"]["p"][moment] = [np.nan, 0]
    elif kind == "moment_overflow":
        state["state"]["p"][moment] = [1e100, 0]
    return state


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
@pytest.mark.parametrize(
    "kind",
    [
        "format",
        "boolean_format",
        "type",
        "names",
        "shapes",
        "hyper_missing",
        "hyper_extra",
        "hyper_bad",
        "state_type",
        "unknown",
        "entry_type",
        "entry_extra",
        "moment_shape",
        "moment_nan",
        "moment_overflow",
    ],
)
def test_bad_checkpoint_is_atomic(gl, gpu_type, kind):
    p = Parameter("p", [1.0, 2.0])
    config = {"momentum": 0.8} if gpu_type is GPUSGD else {}
    with gpu_type(gl, [p], **config) as optimizer:
        p.grad = np.array([0.1, -0.1])
        optimizer.step()
        before = optimizer.state_dict()
        invalid = corrupt_state(copy.deepcopy(before), kind, gpu_type)
        invalid["hyperparameters"]["weight_decay"] = 0.4
        version, value = p.version, p.numpy()
        with pytest.raises(ValueError):
            optimizer.load_state_dict(invalid)
        assert optimizer.state_dict() == before
        assert p.version == version and optimizer.parameters[0] is p
        np.testing.assert_array_equal(p.numpy(), value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("step", 0),
        ("step", -1),
        ("step", True),
        ("step", 1.5),
        ("step", 2**32),
        ("exp_avg_sq", [-1, 0]),
    ],
)
def test_bad_adam_counters_and_variance(gl, key, value):
    p = Parameter("p", [1, 2])
    with GPUAdam(gl, [p]) as optimizer:
        p.grad = np.array([0.1, 0.2])
        optimizer.step()
        before = optimizer.state_dict()
        invalid = copy.deepcopy(before)
        invalid["state"]["p"][key] = value
        with pytest.raises(ValueError):
            optimizer.load_state_dict(invalid)
        assert optimizer.state_dict() == before


def test_exhausted_adam_counter_is_atomic(gl):
    p = Parameter("p", 1.0)
    with GPUAdam(gl, [p]) as optimizer:
        p.grad = np.array(0.2)
        optimizer.step()
        state = optimizer.state_dict()
        state["state"]["p"]["step"] = 2**32 - 1
        optimizer.load_state_dict(state)
        value, version = p.numpy(), p.version
        with pytest.raises(FloatingPointError, match="exhausted"):
            optimizer.step()
        assert optimizer.state_dict() == state
        assert p.version == version
        np.testing.assert_array_equal(p.numpy(), value)


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
@pytest.mark.parametrize(
    "parameters,error",
    [
        ([], ValueError),
        ([Parameter("p", 1), Parameter("p", 2)], ValueError),
        ([Tensor(1)], TypeError),
        ([Parameter("p", 1e100)], ValueError),
    ],
)
def test_invalid_parameters_without_context(gpu_type, parameters, error):
    with pytest.raises(error):
        gpu_type(None, parameters)


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
@pytest.mark.parametrize(
    "config",
    [
        dict(lr=-1),
        dict(lr=np.inf),
        dict(lr=True),
        dict(lr="0.1"),
        dict(lr=[0.1]),
        dict(lr=1e100),
        dict(weight_decay=-1),
        dict(weight_decay=np.nan),
    ],
)
def test_bad_shared_hyperparameters(gpu_type, config):
    with pytest.raises(ValueError):
        gpu_type(None, [Parameter("p", 1)], **config)


@pytest.mark.parametrize(
    "gpu_type,config",
    [
        (GPUSGD, dict(momentum=1)),
        (GPUSGD, dict(momentum=-0.1)),
        (GPUSGD, dict(nesterov=True)),
        (GPUSGD, dict(momentum=0.5, nesterov=1)),
        (GPUAdam, dict(betas=(0.9,))),
        (GPUAdam, dict(betas=(1, 0.9))),
        (GPUAdam, dict(betas=(True, 0.9))),
        (GPUAdam, dict(eps=0)),
        (GPUAdam, dict(eps=1e-100)),
        (GPUAdam, dict(eps=np.nan)),
    ],
)
def test_bad_specific_hyperparameters(gpu_type, config):
    with pytest.raises(ValueError):
        gpu_type(None, [Parameter("p", 1)], **config)


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
def test_requires_compute_context(gpu_type):
    for context in (None, object(), SimpleNamespace(version_code=330)):
        with pytest.raises(ValueError, match="OpenGL 4.3"):
            gpu_type(context, [Parameter("p", 1)])


class TrackedResource:
    def __init__(self, resource):
        self.resource = resource
        self.releases = 0
        self.writes = []
        self.reads = []
        self.runs = 0

    def __getattr__(self, name):
        return getattr(self.resource, name)

    def __getitem__(self, name):
        return self.resource[name]

    def write(self, data):
        self.writes.append(len(data))
        self.resource.write(data)

    def run(self, **kwargs):
        self.runs += 1
        self.resource.run(**kwargs)

    def read(self):
        data = self.resource.read()
        self.reads.append(len(data))
        return data

    def release(self):
        self.releases += 1
        self.resource.release()


class TrackedContext:
    def __init__(self, context, fail_at=None):
        self.context = context
        self.resources = []
        self.fail_at = fail_at
        self.releases = 0

    def __getattr__(self, name):
        return getattr(self.context, name)

    def _create(self, method, *args, **kwargs):
        if len(self.resources) == self.fail_at:
            raise RuntimeError("allocation failed")
        resource = TrackedResource(getattr(self.context, method)(*args, **kwargs))
        self.resources.append(resource)
        return resource

    def compute_shader(self, *args, **kwargs):
        return self._create("compute_shader", *args, **kwargs)

    def buffer(self, *args, **kwargs):
        return self._create("buffer", *args, **kwargs)

    def release(self):
        self.releases += 1


@pytest.mark.parametrize("gpu_type", [GPUSGD, GPUAdam])
def test_native_dispatch_leaf_uploads_close_and_no_cpu_fallback(
    gl, gpu_type, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU optimizer fallback")

    monkeypatch.setattr(SGD, "step", forbidden)
    monkeypatch.setattr(Adam, "step", forbidden)
    tracked = TrackedContext(gl)
    p = Parameter("p", [1, 2, 3])
    with gpu_type(tracked, [p]) as optimizer:
        initial = optimizer.state_dict()
        optimizer.load_state_dict(initial)
        for resource in tracked.resources:
            resource.reads.clear()
        for _ in range(3):
            p.grad = np.array([0.1, 0.2, 0.3])
            optimizer.step()
        assert tracked.resources[0].runs == 3
        assert tracked.resources[1].writes == [3 * 16] * 3
        assert tracked.resources[4].writes == [4] * 3
        assert tracked.resources[4].reads == [4] * 3
        assert tracked.resources[5].reads == [3 * 4] * 3
        assert sum(len(r.reads) for r in tracked.resources) == 6
        assert sum(len(r.writes) for r in tracked.resources) == 6
    optimizer.close()
    assert all(r.releases == 1 for r in tracked.resources)
    assert tracked.releases == 0
    assert optimizer.parameters == (p,)
    for method in (
        optimizer.step,
        optimizer.zero_grad,
        optimizer.state_dict,
        optimizer.__enter__,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            method()
    with pytest.raises(RuntimeError, match="closed"):
        optimizer.load_state_dict(initial)
    buffer = gl.buffer(bytes(4))
    buffer.release()


@pytest.mark.parametrize("fail_at", range(6))
def test_construction_failure_releases_partial_resources(gl, fail_at):
    tracked = TrackedContext(gl, fail_at=fail_at)
    with pytest.raises(RuntimeError, match="allocation failed"):
        GPUAdam(tracked, [Parameter("p", 1)])
    assert all(r.releases == 1 for r in tracked.resources)
    assert tracked.releases == 0


def test_context_manager_releases_on_exception(gl):
    tracked = TrackedContext(gl)
    with pytest.raises(RuntimeError, match="training failure"):
        with GPUAdam(tracked, [Parameter("p", 1)]):
            raise RuntimeError("training failure")
    assert all(r.releases == 1 for r in tracked.resources)
    assert tracked.releases == 0


def test_failed_checkpoint_upload_keeps_live_state(gl):
    tracked = TrackedContext(gl)
    p = Parameter("p", 1)
    with GPUAdam(tracked, [p]) as optimizer:
        p.grad = np.array(0.1)
        optimizer.step()
        state = optimizer.state_dict()
        changed = copy.deepcopy(state)
        changed["hyperparameters"]["lr"] = 0.5
        tracked.fail_at = len(tracked.resources)
        with pytest.raises(RuntimeError, match="allocation failed"):
            optimizer.load_state_dict(changed)
        assert optimizer.state_dict() == state
        optimizer.step()
