"""Prepare standalone MetaX runners before importing torch or Triton."""
from __future__ import annotations

import os
from pathlib import Path
import sys


METAX_LANGUAGES = frozenset({"triton_metax", "maca"})
_ENVIRONMENT_MARKER = "MULTIKERNELBENCH_MACA_ENVIRONMENT"


def maca_environment(environment):
    """Select one MACA SDK while preserving user settings and device bindings."""
    result = dict(environment)
    root = Path(result.get("MACA_PATH") or "/opt/maca").expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MetaX MACA SDK not found: {root}; set MACA_PATH")
    result["MACA_PATH"] = str(root)
    for key, directories in (
        ("PATH", (root / "bin", root / "mxgpu_llvm/bin")),
        ("LD_LIBRARY_PATH", (root / "lib", root / "lib64",
                             Path("/opt/mxdriver/lib"), Path("/opt/mxdriver/lib64"))),
    ):
        paths = [str(path.resolve()) for path in directories if path.is_dir()]
        paths.extend(part for part in result.get(key, "").split(os.pathsep) if part)
        result[key] = os.pathsep.join(dict.fromkeys(paths))
    return result


def prepare_process(language):
    """Re-exec this CLI once so the dynamic loader sees the selected libraries.

    Other platforms return untouched. A Python caller importing backends
    directly must start its interpreter with the desired MACA environment.
    """
    if language not in METAX_LANGUAGES:
        return
    environment = maca_environment(os.environ)
    marker = environment["MACA_PATH"] + os.pathsep + environment["LD_LIBRARY_PATH"]
    if (os.environ.get(_ENVIRONMENT_MARKER) == marker
            and all(os.environ.get(key) == environment[key]
                    for key in ("MACA_PATH", "PATH", "LD_LIBRARY_PATH"))):
        return
    if "torch" in sys.modules or "triton" in sys.modules:
        raise RuntimeError("MetaX environment must be prepared before importing torch/triton; "
                           "start eval_single_runner.py in a fresh process")
    environment[_ENVIRONMENT_MARKER] = marker
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)
