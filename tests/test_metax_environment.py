"""Standalone setup must happen before runtimes are imported."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def environment_module():
    path = Path(__file__).resolve().parents[1] / "utils/metax_environment.py"
    spec = importlib.util.spec_from_file_location("mkb_metax_environment_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_preserves_device_binding_and_selects_sdk(environment_module, tmp_path):
    root = tmp_path / "maca"
    for relative in ("bin", "mxgpu_llvm/bin", "lib"):
        (root / relative).mkdir(parents=True)
    original = {"MACA_PATH": str(root), "LD_LIBRARY_PATH": "/old/lib", "PATH": "/usr/bin",
                "MACA_VISIBLE_DEVICES": "7", "CUDA_VISIBLE_DEVICES": "7", "PRIVATE": "keep"}
    result = environment_module.maca_environment(original)
    assert result["PATH"].split(":")[:2] == [str(root / "bin"), str(root / "mxgpu_llvm/bin")]
    assert result["LD_LIBRARY_PATH"].split(":")[0] == str(root / "lib")
    assert result["MACA_VISIBLE_DEVICES"] == result["CUDA_VISIBLE_DEVICES"] == "7"
    assert result["PRIVATE"] == "keep" and original["LD_LIBRARY_PATH"] == "/old/lib"
    assert environment_module.maca_environment(result) == result


@pytest.mark.parametrize("language", ["cuda", "triton", "ascendc", "triton_ascend", "musa", "pallas", "sycl"])
def test_other_platforms_never_change_or_restart(environment_module, monkeypatch, language):
    monkeypatch.setattr(environment_module, "maca_environment", lambda env: pytest.fail("wrong platform"))
    monkeypatch.setattr(environment_module.os, "execve", lambda *args: pytest.fail("unexpected restart"))
    environment_module.prepare_process(language)


def test_metax_restarts_once_with_same_interpreter_and_arguments(environment_module, tmp_path, monkeypatch):
    (tmp_path / "lib").mkdir()
    monkeypatch.setenv("MACA_PATH", str(tmp_path))
    monkeypatch.delenv(environment_module._ENVIRONMENT_MARKER, raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    calls = []
    monkeypatch.setattr(environment_module.os, "execve", lambda *args: calls.append(args))
    environment_module.prepare_process("maca")
    assert len(calls) == 1
    executable, argv, env = calls[0]
    assert executable == sys.executable and argv == [sys.executable, *sys.argv]
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    environment_module.prepare_process("maca")
    assert len(calls) == 1


def test_missing_selected_sdk_has_clear_error(environment_module, tmp_path):
    with pytest.raises(FileNotFoundError, match="MACA SDK"):
        environment_module.maca_environment({"MACA_PATH": str(tmp_path / "missing")})
