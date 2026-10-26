"""MetaX backend lifecycle contracts tested without accelerator packages."""
from contextlib import contextmanager
import importlib
import linecache
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALID_SOURCE = "import torch\nclass ModelNew(torch.nn.Module):\n    pass\n"


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    # MultiKernelBench uses top-level package names. Isolate them from other
    # checkout/runtime tests and restore the previous modules at fixture exit.
    for name in list(sys.modules):
        if name in {"config", "torch", "triton", "backends", "utils"} or name.startswith(("backends.", "utils.", "triton.")):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(ROOT))
    calls = []
    torch = types.ModuleType("torch")
    torch.__version__ = "2.6.0+metax3.0.0.3"
    torch.version = types.SimpleNamespace(cuda="11.6", maca=None)
    torch._utils_internal = types.SimpleNamespace(USE_MACA=False)
    torch.nn = types.SimpleNamespace(Module=type("Module", (), {}))
    torch.device = lambda value: value

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU initialization must not happen here")

    torch.cuda = types.SimpleNamespace(
        is_initialized=lambda: False, is_available=forbidden, init=forbidden,
        get_device_name=lambda **kwargs: calls.append(("hardware", kwargs)) or "MetaX C500",
        synchronize=lambda **kwargs: calls.append(("synchronize", kwargs)),
        Event=type("Event", (), {}), empty_cache=forbidden, device=forbidden,
    )
    triton = types.ModuleType("triton")
    triton.__version__ = "3.0.0"
    triton.__file__ = str(tmp_path / "triton/__init__.py")

    def jit(function):
        function.warm_cache = forbidden
        return function

    triton.jit = jit
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", triton)
    module = importlib.import_module("backends.triton_metax_backend")
    distribution = types.SimpleNamespace(
        version="3.0.0+metax3.0.0.3",
        locate_file=lambda path: tmp_path / path,
    )
    monkeypatch.setattr(module.metadata, "distribution", lambda name: distribution)
    yield types.SimpleNamespace(module=module, backend=module.TritonMetaxBackend(),
                                torch=torch, triton=triton, distribution=distribution, calls=calls)
    # Modules imported by Python itself are not tracked by monkeypatch. Remove
    # these fixture imports before monkeypatch restores any original modules.
    for name in list(sys.modules):
        if name in {"config", "backends", "utils"} or name.startswith(("backends.", "utils.")):
            sys.modules.pop(name, None)


def test_registration_is_lazy_and_does_not_import_old_backends(runtime, monkeypatch):
    module = runtime.module
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.delitem(sys.modules, "triton")
    importlib.reload(module)
    from backends.backend_registry import BACKEND_REGISTRY

    assert isinstance(BACKEND_REGISTRY["triton_metax"], module.TritonMetaxBackend)
    assert "torch" not in sys.modules and "triton" not in sys.modules
    assert set(BACKEND_REGISTRY) == {"triton_metax"}
    assert not any(name in sys.modules for name in (
        "backends.cuda_backend", "backends.triton_backend", "backends.musa_backend",
        "backends.triton_ascend_backend", "torch_npu", "torch_musa",
    ))


def test_compile_keeps_source_and_metadata_without_eager_jit(runtime, monkeypatch):
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "unchanged")
    progress = {}
    runtime.backend.context = {"__name__": "caller_candidate", "__file__": "/candidate.py",
                               "_evaluation_progress": progress, "old_symbol": object()}
    source = VALID_SOURCE + "import triton\n@triton.jit\ndef kernel():\n    pass\n"
    assert runtime.backend.compile(source, "add_bias_broadcast") == (True, None)
    context = runtime.backend.context
    assert context["__name__"] == "caller_candidate" and context["__file__"] == "/candidate.py"
    assert context["_evaluation_progress"] is progress and "old_symbol" not in context
    filename = context["kernel"].__code__.co_filename
    assert "".join(linecache.getlines(filename)) == source
    assert runtime.calls == []
    import os
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "unchanged"


@pytest.mark.parametrize("version,use_maca,accepted", [
    ("2.6.0+cu124", False, False), ("2.6.0", False, False),
    ("2.6.0", "true", False), ("2.6.0", True, True),
    ("2.6.0+metax3.0.0.3", False, True),
])
def test_runtime_identity_uses_loaded_torch_not_cuda_api_or_metadata(runtime, version, use_maca, accepted):
    runtime.torch.__version__ = version
    runtime.torch._utils_internal.USE_MACA = use_maca
    runtime.torch.version.maca = "3.0.0"  # This alone is deliberately insufficient.
    compiled, info = runtime.backend.compile(VALID_SOURCE, "op")
    assert compiled is accepted
    if not accepted:
        assert "requires MetaX PyTorch" in info
        assert "ModelNew" not in runtime.backend.context


@pytest.mark.parametrize("version,metadata_version,same_path,accepted", [
    ("3.0.0", "3.0.0+metax3.0.0.3", True, True),
    ("3.0.0", "3.0.0", True, False),
    ("3.0.0", "3.0.0+metax3.0.0.3", False, False),
    ("3.0.0+metax3.0.0.3", "3.0.0", True, True),
])
def test_triton_metadata_must_belong_to_loaded_package(runtime, version, metadata_version, same_path, accepted):
    runtime.triton.__version__ = version
    runtime.distribution.version = metadata_version
    if not same_path:
        runtime.triton.__file__ = "/other/nvidia/triton/__init__.py"
    compiled, info = runtime.backend.compile(VALID_SOURCE, "op")
    assert compiled is accepted
    if not accepted:
        assert "requires MetaX Triton" in info


@pytest.mark.parametrize("source", ["x = 1\n", "ModelNew = 3\n", "class ModelNew:\n    pass\n",
                                    "partial = 1\nraise ValueError('broken')\n", "def invalid(\n"])
def test_failed_compile_cannot_reuse_previous_model_or_partial_context(runtime, source):
    assert runtime.backend.compile(VALID_SOURCE + "old_symbol = 1\n", "first")[0]
    filenames = set(runtime.backend._source_files)
    compiled, info = runtime.backend.compile(source, "second")
    assert not compiled and info
    assert "ModelNew" not in runtime.backend.context and "partial" not in runtime.backend.context
    assert "old_symbol" not in runtime.backend.context
    assert all(filename not in linecache.cache for filename in filenames)
    assert runtime.backend._source_files == set()


def test_successful_recompile_and_repeated_cleanup_start_fresh(runtime):
    assert runtime.backend.compile(VALID_SOURCE + "old_symbol = 1\n", "first")[0]
    assert runtime.backend.compile(VALID_SOURCE + "new_symbol = 2\n", "second")[0]
    assert "old_symbol" not in runtime.backend.context
    assert runtime.backend.context["new_symbol"] == 2
    filenames = set(runtime.backend._source_files)
    runtime.backend.cleanup()
    runtime.backend.cleanup()
    assert runtime.backend.context == {}
    assert all(filename not in linecache.cache for filename in filenames)
    assert runtime.backend.compile(VALID_SOURCE, "third")[0]


def test_native_correctness_and_event_timing_are_used_unchanged(runtime, monkeypatch):
    correctness = types.ModuleType("utils.correctness")
    performance = types.ModuleType("utils.performance")
    correctness.execute_template = lambda *args: runtime.calls.append(("correctness", args)) or (True, "")
    performance.time_execution_event_template = lambda *args: runtime.calls.append(("timing", args)) or [1.5]
    monkeypatch.setitem(sys.modules, "utils.correctness", correctness)
    monkeypatch.setitem(sys.modules, "utils.performance", performance)
    runtime.backend.device = "cuda:3"
    assert runtime.backend.compile(VALID_SOURCE, "op")[0]
    assert runtime.backend.correctness_execution("Model = object\n") == (True, "")
    assert runtime.calls[-1][1] == (runtime.torch.cuda.synchronize, "cuda:3", runtime.backend.context)
    assert runtime.backend.time_execution("Model") == [1.5]
    assert runtime.calls[-1][1] == (runtime.backend.context, "cuda:3", runtime.torch.cuda.synchronize,
                                     runtime.torch.cuda.Event, "Model")


def test_cleanup_uses_selected_device_only_after_runtime_initialization(runtime):
    @contextmanager
    def selected(device):
        runtime.calls.append(("device", device))
        yield

    runtime.torch.cuda.is_initialized = lambda: True
    runtime.torch.cuda.device = selected
    runtime.torch.cuda.empty_cache = lambda: runtime.calls.append(("empty",))
    runtime.torch.cuda.reset_peak_memory_stats = lambda **kwargs: runtime.calls.append(("reset", kwargs))
    runtime.backend.device = "cuda:2"
    assert runtime.backend.compile(VALID_SOURCE, "op")[0]
    runtime.backend.cleanup()
    assert runtime.calls == [("device", "cuda:2"), ("synchronize", {"device": "cuda:2"}),
                             ("empty",), ("reset", {"device": "cuda:2"})]
    runtime.backend.cleanup()
    assert len(runtime.calls) == 4


def test_torch_import_error_identifies_environment_requirement(runtime, monkeypatch):
    base = importlib.import_module("backends.metax_backend")
    original = base.importlib.import_module

    def broken(name):
        if name == "torch":
            raise ImportError("undefined symbol: mcclAllToAllExt")
        return original(name)

    monkeypatch.setattr(base.importlib, "import_module", broken)
    compiled, info = runtime.backend.compile(VALID_SOURCE, "op")
    assert not compiled
    assert "prepared before importing torch" in info and "mcclAllToAllExt" in info
