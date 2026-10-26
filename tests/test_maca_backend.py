"""MACA project/build contracts tested without importing torch or using a GPU."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def compiler(monkeypatch):
    spec = importlib.util.spec_from_file_location("maca_compiler_test_module", ROOT / "backends/maca_compiler.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project_spec():
    return {"entry": {"model": "ModelNew.py::ModelNew"},
            "build": {"kernel_sources": ["native/kernel.cpp"], "binding_sources": ["binding.cpp"],
                      "include_dirs": [".", "native"], "module_name": "test_maca", "build_dir": "build"},
            "sources": [{"path": "ModelNew.py", "content": "class ModelNew: pass\n"},
                        {"path": "native/kernel.cpp", "content": "#include <mc_runtime.h>\n"},
                        {"path": "binding.cpp", "content": "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}\n"}]}


def test_json_and_fenced_json_have_same_contract(compiler, project_spec):
    plain = compiler.parse_project(json.dumps(project_spec))
    fenced = compiler.parse_project("```json\n" + json.dumps(project_spec) + "\n```")
    assert plain == fenced
    assert plain.model_path == Path("ModelNew.py") and plain.module_name == "test_maca"


@pytest.mark.parametrize("path", ["../escape.cpp", "/tmp/escape.cpp", "a/../../escape.cpp", "", "."])
def test_source_cannot_escape_project(compiler, project_spec, path):
    project_spec["sources"][0]["path"] = path
    with pytest.raises(ValueError, match="path"):
        compiler.parse_project(json.dumps(project_spec))


def test_duplicate_normalized_path_and_file_directory_conflict_rejected(compiler, project_spec):
    project_spec["sources"].append({"path": "native/./kernel.cpp", "content": "duplicate"})
    with pytest.raises(ValueError, match="Duplicate"):
        compiler.parse_project(json.dumps(project_spec))
    project_spec["sources"][-1] = {"path": "native", "content": "file shadows directory"}
    with pytest.raises(ValueError, match="Conflicting"):
        compiler.parse_project(json.dumps(project_spec))


@pytest.mark.parametrize("change,match", [
    ({"kernel_sources": ["absent.cpp"]}, "Missing compiled"),
    ({"binding_sources": ["native/kernel.cpp"]}, "disjoint"),
    ({"module_name": "has-hyphen"}, "identifier"),
    ({"build_dir": "../outside"}, "Unsafe"),
    ({"build_dir": "native"}, "separate"),
    ({"include_dirs": "native"}, "must be a list"),
])
def test_invalid_build_spec_is_rejected(compiler, project_spec, change, match):
    project_spec["build"].update(change)
    with pytest.raises(ValueError, match=match):
        compiler.parse_project(json.dumps(project_spec))


def test_inference_and_literal_module_name_agree(compiler, project_spec):
    project_spec["build"] = {}
    project_spec["sources"][-1]["content"] = "PYBIND11_MODULE(literal_name, m) {}"
    project = compiler.parse_project(json.dumps(project_spec))
    assert project.module_name == "literal_name"
    assert project.kernels == [Path("native/kernel.cpp")]
    project_spec["build"]["module_name"] = "wrong_name"
    with pytest.raises(ValueError, match="disagrees"):
        compiler.parse_project(json.dumps(project_spec))


def test_staging_is_unique_and_keeps_previous_diagnostics(compiler, project_spec, tmp_path):
    project = compiler.parse_project(json.dumps(project_spec))
    config = SimpleNamespace(maca_build_root=tmp_path)
    first = compiler.stage_project(project, "path/operator", config)
    (first / "build.log").write_text("keep this failure")
    second = compiler.stage_project(project, "path/operator", config)
    assert first != second
    assert (first / "build.log").read_text() == "keep this failure"
    assert (second / "native/kernel.cpp").is_file()
    assert first.parent == second.parent == tmp_path


def test_default_build_root_uses_external_cache(compiler, project_spec, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    config = SimpleNamespace(maca_build_root=None, op_engineer_dir=tmp_path / "legacy_ascend")
    work = compiler.stage_project(compiler.parse_project(json.dumps(project_spec)), "task", config)
    assert work.parent == tmp_path / "cache/multikernelbench/maca"
    assert not config.op_engineer_dir.exists()


@pytest.fixture
def build_environment(tmp_path, monkeypatch, compiler):
    maca = tmp_path / "maca"
    torch_root = tmp_path / "torch"
    host_include = tmp_path / "gcc/23/include/c++"
    host_library = tmp_path / "gcc/23/lib"
    python_include = tmp_path / "python/include"
    for path in (maca / "include/mcr", maca / "include/common", maca / "lib",
                 torch_root / "include/torch/csrc/api/include", torch_root / "lib",
                 host_include, host_library, python_include):
        path.mkdir(parents=True)
    (maca / "include/mcr/mc_runtime.h").touch()
    (host_library / "libstdc++.so").touch()
    fake_torch = SimpleNamespace(__file__=str(torch_root / "__init__.py"),
                                _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False,
                                                   _PYBIND11_COMPILER_TYPE="_gcc",
                                                   _PYBIND11_STDLIB="_libstdcpp",
                                                   _PYBIND11_BUILD_ABI="_cxxabi1011"))
    monkeypatch.setattr(compiler.sysconfig, "get_path", lambda key: str(python_include))
    monkeypatch.setattr(compiler.sysconfig, "get_config_var", lambda key: ".cpython-test.so")
    monkeypatch.setattr(compiler.shutil, "which", lambda name, **kw: name if str(name).startswith("/") else "/host/g++")
    commands = []

    def run(command, **options):
        commands.append((command, options))
        if "-E" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr=f"#include <...> search starts here:\n {host_include}\nEnd of search list.\n")
        if "-print-file-name=libstdc++.so" in command:
            return SimpleNamespace(returncode=0, stdout=str(host_library / "libstdc++.so") + "\n", stderr="")
        target = Path(command[command.index("-o") + 1])
        target.write_bytes(b"fake compiled artifact")
        return SimpleNamespace(returncode=0, stdout="compiler detail", stderr="compiler warning")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    environment = {"PATH": "/host", "MACA_PATH": str(maca), "TORCH_CUDA_ARCH_LIST": "unchanged",
                   "CUDA_VISIBLE_DEVICES": "7", "MACA_VISIBLE_DEVICES": "7"}
    return SimpleNamespace(maca=maca, torch=fake_torch, host_include=host_include,
                           host_library=host_library, environment=environment, commands=commands)


def test_real_compiler_plan_separates_device_and_host_and_preserves_environment(
        compiler, project_spec, build_environment, tmp_path):
    project_spec["sources"].append({"path": "other/kernel.cpp", "content": "// same basename"})
    project_spec["build"]["kernel_sources"].append("other/kernel.cpp")
    project = compiler.parse_project(json.dumps(project_spec))
    config = SimpleNamespace(maca_build_root=tmp_path / "projects", maca_build_jobs=2,
                             maca_build_type="Debug", maca_extra_cflags=["-DHOST_ONLY"],
                             maca_extra_mxcc_flags=["--offload-arch=xcore1000"], maca_extra_ldflags=["-lm"])
    work = compiler.stage_project(project, "task", config)
    before = dict(os.environ)
    result = compiler.build_extension(project, work, build_environment.torch, config,
                                      environment=build_environment.environment)
    assert result.name == "test_maca.cpython-test.so" and result.is_file()
    compilation = [command for command, _ in build_environment.commands if "-c" in command]
    device_commands = [command for command in compilation if command[0].endswith("mxcc")]
    host_commands = [command for command in compilation if command[0] == "/host/g++"]
    assert len(device_commands) == 2 and len(host_commands) == 1
    for command in device_commands:
        assert command[command.index("-x") + 1] == "maca"
        assert str(build_environment.host_include) in command
        assert str(build_environment.host_library) in command
        assert "--offload-arch=xcore1000" in command and "-DHOST_ONLY" not in command
    assert "-DHOST_ONLY" in host_commands[0] and "--offload-arch=xcore1000" not in host_commands[0]
    assert len({command[command.index("-o") + 1] for command in compilation}) == 3
    assert all("-D_GLIBCXX_USE_CXX11_ABI=0" in command for command in compilation)
    assert all('-DPYBIND11_COMPILER_TYPE="_gcc"' in command for command in compilation)
    link = build_environment.commands[-1][0]
    assert "-ltorch_cuda" in link and "-lmcruntime" in link and "-lm" in link
    assert os.environ == before
    assert all(options["env"]["TORCH_CUDA_ARCH_LIST"] == "unchanged" for _, options in build_environment.commands)
    assert "compiler warning" in (work / "build.log").read_text()


def test_compile_failure_keeps_complete_diagnostics(compiler, tmp_path, monkeypatch):
    monkeypatch.setattr(compiler.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=17, stdout="full stdout detail", stderr="full stderr detail"))
    runner = compiler.BuildRunner(tmp_path, {})
    with pytest.raises(RuntimeError) as caught:
        runner.run("MACA kernel", ["/tool with spaces/mxcc", "-c", "kernel.cpp"])
    text = str(caught.value)
    assert "17" in text and "full stdout detail" in text and "full stderr detail" in text
    assert "'/tool with spaces/mxcc'" in text
    assert "full stderr detail" in (tmp_path / "build.log").read_text()


@pytest.fixture
def backend_module(compiler, monkeypatch, tmp_path):
    package = ModuleType("backends")
    package.__path__ = [str(ROOT / "backends")]
    monkeypatch.setitem(sys.modules, "backends", package)
    monkeypatch.setitem(sys.modules, "backends.maca_compiler", compiler)
    registry = ModuleType("backends.backend_registry")
    registry.BACKEND_REGISTRY = {}

    def register(name):
        def decorate(cls):
            registry.BACKEND_REGISTRY[name] = cls()
            return cls
        return decorate

    registry.register_backend = register
    monkeypatch.setitem(sys.modules, "backends.backend_registry", registry)
    base = ModuleType("backends.metax_backend")

    class MetaxBackend:
        def __init__(self):
            self.context = {}

        def reset_context(self):
            self.context = {key: value for key, value in self.context.items() if key == "_evaluation_progress"}
            return self.context

        def validate_runtime(self):
            return "fake torch runtime"

        def validate_model(self, context, torch):
            assert isinstance(context.get("ModelNew"), type)

        def cleanup(self):
            self.reset_context()

    base.MetaxBackend = MetaxBackend
    monkeypatch.setitem(sys.modules, "backends.metax_backend", base)
    config = ModuleType("config")
    config.maca_build_root = tmp_path
    monkeypatch.setitem(sys.modules, "config", config)
    spec = importlib.util.spec_from_file_location("maca_backend_test_module", ROOT / "backends/maca_backend.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert "maca" in registry.BACKEND_REGISTRY
    return module


def test_backend_loads_entry_and_clears_previous_candidate(backend_module, project_spec, monkeypatch):
    built = []

    def build(project, work, torch, config):
        assert torch == "fake torch runtime"
        path = work / "test_maca.so"
        path.touch()
        built.append(path)
        return path

    monkeypatch.setattr(backend_module, "build_extension", build)
    project_spec["entry"]["model"] = "ModelNew.py::CustomModel"
    project_spec["sources"][0]["content"] = "class CustomModel:\n    extension = _maca_extension\n    name = _maca_module_name\n"
    backend = backend_module.MacaBackend()
    backend.context.update(ModelNew="stale", _maca_extension="stale", _evaluation_progress={"total": 1})
    passed, error = backend.compile(json.dumps(project_spec), "task")
    assert passed and error is None
    assert backend.context["ModelNew"].extension == str(built[0])
    assert backend.context["ModelNew"].name == "test_maca"
    assert backend.context["_evaluation_progress"] == {"total": 1}
    previous = backend.work_dir
    project_spec["sources"][0]["content"] = "raise RuntimeError('model load failed')"
    passed, error = backend.compile(json.dumps(project_spec), "task")
    assert not passed and "model load failed" in error
    assert "ModelNew" not in backend.context and "_maca_extension" not in backend.context
    assert previous.is_dir() and backend.work_dir.is_dir()
    retained = backend.work_dir
    passed, error = backend.compile("not JSON", "task")
    assert not passed and backend.work_dir is None
    assert "Project/build diagnostics" not in error
    backend.cleanup()
    assert retained.is_dir() and backend.work_dir is None
