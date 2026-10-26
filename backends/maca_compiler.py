"""Build a self-contained MACA C++ project without global compiler overrides."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sysconfig
import tempfile
import threading


def _relative(value, *, directory=False):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"Expected a relative project path, got {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or (not directory and path == Path(".")):
        raise ValueError(f"Unsafe project path: {value}")
    return path


def _paths(value, label, *, directory=False):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of relative paths")
    paths = [_relative(item, directory=directory) for item in value]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate paths in {label}")
    return paths


@dataclass(frozen=True)
class Project:
    sources: dict[Path, str]
    kernels: list[Path]
    bindings: list[Path]
    includes: list[Path]
    module_name: str
    build_dir: Path
    model_path: Path
    model_name: str


def parse_project(generated_code: str) -> Project:
    text = generated_code.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if not match:
            raise ValueError("Expected one complete JSON project code block")
        text = match[1]
    elif text.startswith("json\n"):
        text = text[5:]
    spec = json.loads(text)
    if not isinstance(spec, dict) or not isinstance(spec.get("sources"), list) or not spec["sources"]:
        raise ValueError("MACA project requires a nonempty sources list")
    sources = {}
    for source in spec["sources"]:
        if not isinstance(source, dict) or not isinstance(source.get("content"), str):
            raise ValueError("Each source requires path and string content")
        path = _relative(source.get("path"))
        if path in sources:
            raise ValueError(f"Duplicate source path: {path}")
        if any(path in other.parents or other in path.parents for other in sources):
            raise ValueError(f"Conflicting file/directory source path: {path}")
        sources[path] = source["content"]
    entry = spec.get("entry", {})
    build = spec.get("build", {})
    if not isinstance(entry, dict) or not isinstance(build, dict):
        raise ValueError("entry and build must be JSON objects")
    model = entry.get("model", "ModelNew.py::ModelNew")
    if not isinstance(model, str) or model.count("::") != 1:
        raise ValueError("entry.model must have the form ModelNew.py::ModelNew")
    model_path, model_name = model.split("::")
    model_path = _relative(model_path)
    if model_path not in sources or not model_name.isidentifier():
        raise ValueError(f"Missing or invalid model entry: {model}")
    cpp = [path for path in sources if path.suffix.lower() in {".cpp", ".cc", ".cxx", ".cu", ".maca"}]
    bindings = _paths(build.get("binding_sources", []), "build.binding_sources")
    if not bindings:
        bindings = [path for path in cpp if "PYBIND11_MODULE" in sources[path]]
    if not bindings:
        raise ValueError("Specify build.binding_sources containing the PyTorch Python binding")
    kernels = _paths(build.get("kernel_sources", []), "build.kernel_sources")
    if not kernels:
        kernels = [path for path in cpp if path not in bindings]
    if not kernels:
        raise ValueError("Specify at least one MACA device source in build.kernel_sources")
    if set(kernels) & set(bindings):
        raise ValueError("Kernel and binding source lists must be disjoint")
    for path in kernels + bindings:
        if path not in sources:
            raise ValueError(f"Missing compiled source: {path}")
    names = {match[1] for path in bindings for match in re.finditer(
        r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", sources[path])
        if match[1] != "TORCH_EXTENSION_NAME"}
    if len(names) > 1:
        raise ValueError("Binding sources declare different Python extension module names")
    module_name = build.get("module_name") or next(iter(names), "maca_ops")
    if not isinstance(module_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module_name):
        raise ValueError("build.module_name must be a C/Python module identifier")
    if names and module_name not in names:
        raise ValueError("build.module_name disagrees with PYBIND11_MODULE")
    includes = _paths(build.get("include_dirs", []), "build.include_dirs", directory=True)
    if not includes:
        includes = list(dict.fromkeys([Path("."), *(path.parent for path in sources)]))
    build_dir = _relative(build.get("build_dir", "build"))
    if any(build_dir == path or build_dir in path.parents or path in build_dir.parents for path in sources):
        raise ValueError("build.build_dir must be separate from submitted source files")
    return Project(sources, kernels, bindings, includes, module_name, build_dir, model_path, model_name)


def stage_project(project: Project, op: str, config) -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    root = (getattr(config, "maca_build_root", None) or getattr(config, "maca_build_dir", None)
            or cache / "multikernelbench" / "maca")
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(op)).strip("_")[:64] or "operator"
    work_dir = Path(tempfile.mkdtemp(prefix=prefix + "_", dir=root))
    for path, content in project.sources.items():
        target = work_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return work_dir


class BuildRunner:
    def __init__(self, work_dir: Path, env: dict):
        self.work_dir, self.env = work_dir, env
        self.log = work_dir / "build.log"
        self._lock = threading.Lock()

    def run(self, stage, command, *, input=None):
        try:
            result = subprocess.run(command, cwd=self.work_dir, env=self.env, input=input,
                                    capture_output=True, text=True, errors="replace", check=False)
        except OSError as exc:
            message = f"{stage} failed\nCommand: {shlex.join(command)}\nWorking directory: {self.work_dir}\n{exc}\n"
            self.record(message)
            raise RuntimeError(message) from exc
        message = (f"{stage}\nCommand: {shlex.join(command)}\nWorking directory: {self.work_dir}\n"
                   f"Exit code: {result.returncode}\n[STDOUT]\n{result.stdout}\n[STDERR]\n{result.stderr}\n")
        self.record(message)
        if result.returncode:
            raise RuntimeError(message + f"Build log: {self.log}")
        return result

    def record(self, message):
        with self._lock, self.log.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def _executable(value, default: str, env: dict, label: str) -> str:
    value = value or default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must name one compiler executable")
    binary = shutil.which(value, path=env.get("PATH", os.defpath))
    if binary is None:
        raise FileNotFoundError(f"Cannot find {label}: {value}; configure an executable path, not a shell function")
    return str(Path(binary).resolve())


def _flags(config, name):
    values = getattr(config, name, None) or []
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{name} must be a list of command-line arguments")
    return list(values)


def host_cpp_paths(cxx: str, runner: BuildRunner) -> tuple[list[Path], Path]:
    """Ask the selected host compiler; mxcc's baked-in GCC version may differ."""
    result = runner.run("Discover host C++ headers", [cxx, "-E", "-x", "c++", "-", "-v"], input="")
    inside = False
    includes = []
    for line in result.stderr.splitlines():
        if "#include <...> search starts here:" in line:
            inside = True
        elif "End of search list." in line:
            inside = False
        elif inside:
            path = Path(line.strip().removesuffix(" (framework directory)"))
            if path.is_dir():
                includes.append(path.resolve())
    if not includes:
        raise RuntimeError("Could not determine host C++ include paths; inspect build.log and maca_cxx")
    result = runner.run("Discover host C++ library", [cxx, "-print-file-name=libstdc++.so"])
    library = Path(result.stdout.strip())
    if not library.is_absolute() or not library.is_file():
        raise RuntimeError("Host compiler did not resolve libstdc++.so; configure a GCC-compatible maca_cxx")
    return list(dict.fromkeys(includes)), library.parent


def build_extension(project: Project, work_dir: Path, torch, config, *, environment=None) -> Path:
    env = dict(os.environ if environment is None else environment)
    maca = Path(env.get("MACA_PATH") or "/opt/maca").expanduser().resolve()
    if not (maca / "include/mcr/mc_runtime.h").is_file():
        raise FileNotFoundError(f"Selected MACA SDK has no include/mcr/mc_runtime.h: {maca}")
    cxx = _executable(getattr(config, "maca_cxx", None), env.get("CXX") or "g++", env, "maca_cxx")
    mxcc = _executable(getattr(config, "maca_mxcc", None), str(maca / "mxgpu_llvm/bin/mxcc"), env, "maca_mxcc")
    jobs = getattr(config, "maca_build_jobs", 1)
    if type(jobs) is not int or jobs < 1:
        raise ValueError("maca_build_jobs must be a positive integer")
    build_type = getattr(config, "maca_build_type", "Release")
    optimizations = {"Release": ["-O3", "-DNDEBUG"], "Debug": ["-O0", "-g"],
                     "RelWithDebInfo": ["-O2", "-g", "-DNDEBUG"], "MinSizeRel": ["-Os", "-DNDEBUG"]}
    if build_type not in optimizations:
        raise ValueError(f"Unsupported maca_build_type: {build_type}")
    build = work_dir / project.build_dir
    objects = build / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    runner = BuildRunner(work_dir, env)
    host_includes, host_library = host_cpp_paths(cxx, runner)
    torch_root = Path(torch.__file__).resolve().parent
    include_dirs = [*(work_dir / path for path in project.includes),
                    torch_root / "include", torch_root / "include/torch/csrc/api/include",
                    *(Path(path) for key in ("include", "platinclude") if (path := sysconfig.get_path(key))),
                    maca / "include", maca / "include/mcr", maca / "include/common"]
    for path in include_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required/project include directory: {path}")
    includes = [argument for path in dict.fromkeys(include_dirs) for argument in ("-I", str(path))]
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    common = ["-std=c++17", "-fPIC", "-DUSE_MACA", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
              f"-DTORCH_EXTENSION_NAME={project.module_name}", "-DTORCH_API_INCLUDE_EXTENSION_H", *includes]
    for name in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI"):
        value = getattr(torch._C, f"_PYBIND11_{name}", None)
        if value is not None:
            common.append(f'-DPYBIND11_{name}="{value}"')
    host_flags = _flags(config, "maca_extra_cflags")
    device_flags = _flags(config, "maca_extra_mxcc_flags")
    link_flags = _flags(config, "maca_extra_ldflags")
    commands, object_paths = [], []
    for kind, sources, compiler in (("MACA kernel", project.kernels, mxcc), ("C++ binding", project.bindings, cxx)):
        for relative in sources:
            token = hashlib.sha256(str(relative).encode()).hexdigest()[:16]
            target = objects / f"{relative.stem}_{token}.o"
            object_paths.append(target)
            flags = [*optimizations[build_type]]
            if kind == "MACA kernel":
                flags += ["-x", "maca", f"--maca-path={maca}", *device_flags,
                          *(arg for path in host_includes for arg in ("-I", str(path))), "-L", str(host_library)]
            else:
                flags += host_flags
            commands.append((kind + " " + str(relative), [compiler, *flags, *common,
                                                         "-c", str(work_dir / relative), "-o", str(target)]))
    # Each source has its own object name, including files with identical stems.
    with ThreadPoolExecutor(max_workers=min(jobs, len(commands))) as workers:
        futures = [workers.submit(runner.run, stage, command) for stage, command in commands]
        for future in futures:
            future.result()
    extension = build / (project.module_name + (sysconfig.get_config_var("EXT_SUFFIX") or ".so"))
    libraries = [torch_root / "lib", maca / "lib", host_library]
    link = [cxx, "-shared", *(str(path) for path in object_paths)]
    for directory in libraries:
        link += ["-L", str(directory), "-Xlinker", "-rpath", "-Xlinker", str(directory)]
    link += ["-ltorch_python", "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10_cuda", "-lc10", "-lmcruntime",
             *link_flags, "-o", str(extension)]
    runner.run("Link MACA Python extension", link)
    if not extension.is_file():
        raise FileNotFoundError(f"Compiler succeeded without creating {extension}; see {runner.log}")
    return extension
