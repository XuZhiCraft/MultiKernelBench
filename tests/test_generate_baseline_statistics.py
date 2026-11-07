"""Reference selection and failure handling without accelerator dependencies."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    calls = []
    reference = tmp_path / "reference.py"
    reference.write_text(
        'from pathlib import Path\nsidecar = Path(__file__).with_suffix(".json").read_text()\n'
    )
    reference.with_suffix(".json").write_text("native inputs")

    def time_execution(target):
        assert target == "Model"
        assert backend.context["sidecar"] == "native inputs"
        return [1.0, 3.0]

    backend = SimpleNamespace(context={}, time_execution=time_execution,
                              get_hardware_name=lambda: "Test GPU",
                              cleanup=lambda: calls.append("cleanup"))
    attributes = {
        "config": {"project_root_path": str(tmp_path), "seed_num": 1024},
        "dataset": {"dataset": {"chosen": {}, "other": {}}},
        "backends.backend_registry": {"BACKEND_REGISTRY": {"fixture": backend}},
        "utils.correctness": {"set_seed": lambda seed: calls.append(seed)},
        "utils.utils": {"get_ref_src_path": lambda op: str(reference)},
        "utils.metax_environment": {"prepare_process": lambda language: calls.append(language)},
    }
    for name, values in attributes.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("baseline_statistics_test", ROOT / "generate_baseline_statistics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, backend, calls


def test_registered_backend_times_reference_with_native_sidecar(baseline):
    module, _, calls = baseline
    result = {}
    module.run_op("chosen", "fixture", result)
    assert result["chosen"] == {
        "mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0,
        "num_trials": 2, "device": "Test GPU",
    }
    assert calls == [1024, "cleanup"]


@pytest.mark.parametrize("samples", [[], [float("nan")], [0.0]])
def test_invalid_timing_is_recorded_as_failure_and_cleans_up(baseline, samples):
    module, backend, calls = baseline
    backend.time_execution = lambda target: samples
    result = {}
    module.run_op("chosen", "fixture", result)
    assert result == {"chosen": "not supported"}
    assert calls[-1] == "cleanup"


def test_unknown_operator_is_rejected_before_preparing_device_environment(baseline):
    module, _, calls = baseline
    with pytest.raises(SystemExit) as error:
        module.main(["--op", "missing", "--language", "fixture"])
    assert error.value.code == 2
    assert calls == []


def inline_workers(module, monkeypatch):
    """Keep CLI/file tests local; real worker execution is covered by a GPU smoke run."""
    class Process:
        def __init__(self, target, args):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

        def close(self):
            pass

    monkeypatch.setattr(module.mp, "Manager", lambda: nullcontext(SimpleNamespace(dict=dict)))
    monkeypatch.setattr(module.mp, "Process", Process)


def test_single_operator_does_not_replace_full_baseline(baseline, tmp_path, monkeypatch):
    module, _, _ = baseline
    inline_workers(module, monkeypatch)
    full = tmp_path / "baselines/fixture_Test GPU.json"
    full.parent.mkdir()
    full.write_text("existing full results")
    assert module.main(["--op", "chosen", "--language", "fixture"]) == 0
    selected = full.with_name("fixture_Test GPU_chosen.json")
    assert list(json.loads(selected.read_text())) == ["chosen"]
    assert full.read_text() == "existing full results"


def test_failure_does_not_abort_remaining_operators_or_result_file(baseline, tmp_path, monkeypatch):
    module, backend, _ = baseline
    inline_workers(module, monkeypatch)
    native_timer = backend.time_execution
    attempts = []

    def timer(target):
        attempts.append(target)
        if len(attempts) == 1:
            raise RuntimeError("unsupported reference")
        return native_timer(target)

    backend.time_execution = timer
    output = tmp_path / "reports/baseline.json"
    assert module.main(["--language", "fixture", "--output", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["chosen"] == "not supported"
    assert result["other"]["mean"] == 2.0
    assert attempts == ["Model", "Model"]
