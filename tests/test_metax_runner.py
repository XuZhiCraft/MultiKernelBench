"""MetaX CLI failure reports cannot be mistaken for a previous success."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    dataset = ModuleType("dataset")
    dataset.dataset = {"fixture": {"category": "test"}}
    monkeypatch.setitem(sys.modules, "dataset", dataset)
    helper = ModuleType("utils.metax_environment")
    helper.METAX_LANGUAGES = {"maca", "triton_metax"}
    helper.prepare_process = lambda language: None
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    evaluation = ModuleType("utils.evaluation_utils")
    evaluation.eval_single = lambda *args: {"compiled": True, "correctness": True,
        "cheating": False, "hardware": "MetaX C500", "performance": {"mean": 0.01, "num_trials": 3}}
    monkeypatch.setitem(sys.modules, evaluation.__name__, evaluation)
    source = Path(__file__).resolve().parents[1] / "eval_single_runner.py"
    spec = importlib.util.spec_from_file_location("metax_runner_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("class ModelNew: pass\n")
    result = tmp_path / "result.json"
    result.write_text('{"compiled": true, "old_success": true}')
    arguments = SimpleNamespace(input_path=str(candidate), result_path=str(result),
                                op="fixture", language="maca", indent=2)
    monkeypatch.setattr(module, "parse_args", lambda: arguments)
    return module, evaluation, helper, arguments, result


@pytest.mark.parametrize("language", ["maca", "triton_metax"])
def test_valid_metax_result_has_successful_exit(runner, language):
    module, _, _, args, result = runner
    args.language = language
    assert module.main() == 0
    assert json.loads(result.read_text())["correctness"]
    assert not list(result.parent.glob("*.tmp"))


@pytest.mark.parametrize("failure", ["compile", "correctness", "timing", "exception", "sdk"])
def test_failed_metax_run_replaces_old_result_and_exits_nonzero(runner, failure):
    module, evaluator, helper, _, result = runner
    def evaluate(*args):
        assert not result.exists()
        if failure == "exception":
            raise RuntimeError("device failure")
        return {"compiled": failure != "compile", "correctness": failure == "timing",
                "performance": None}
    evaluator.eval_single = evaluate
    if failure == "sdk":
        def missing(language):
            raise FileNotFoundError("MACA SDK missing")
        helper.prepare_process = missing
    assert module.main() == 1
    report = json.loads(result.read_text())
    assert "old_success" not in report
    assert report["performance"] is None
    if failure in {"exception", "sdk"}:
        assert report["error"]


@pytest.mark.parametrize("language", ["cuda", "triton", "ascendc", "musa"])
def test_existing_backends_keep_native_failure_exit_semantics(runner, language):
    module, evaluator, _, args, result = runner
    args.language = language
    report = {"compiled": False, "correctness": False, "performance": None, "compile_info": "bad source"}
    evaluator.eval_single = lambda *args: report
    assert module.main() is None
    assert json.loads(result.read_text()) == report


@pytest.mark.parametrize("symlink", [False, True])
def test_result_cannot_replace_candidate_source(runner, symlink):
    module, _, _, args, _ = runner
    candidate = Path(args.input_path)
    original = candidate.read_text()
    result = candidate
    if symlink:
        result = candidate.with_name("candidate_link.py")
        result.symlink_to(candidate)
    args.result_path = str(result)
    with pytest.raises(ValueError, match="paths must be different"):
        module.main()
    assert candidate.read_text() == original
    assert result.read_text() == original
