import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def packager():
    source = Path(__file__).resolve().parents[1] / "scripts/maca_project_to_json.py"
    spec = importlib.util.spec_from_file_location("maca_packaging_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project(root, **build):
    (root / "manifest.json").write_text(json.dumps({"build": build}))
    (root / "ModelNew.py").write_text("class ModelNew: pass\n")
    (root / "kernel.cpp").write_text("// native MACA\n")


def test_project_packaging_is_deterministic_and_excludes_outputs(tmp_path, packager):
    project(tmp_path, kernel_sources=["kernel.cpp"], build_dir="out")
    (tmp_path / "out").mkdir()
    (tmp_path / "out/generated.cpp").write_text("not a candidate")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache/private.py").write_text("not a candidate")
    result = packager.package_project(tmp_path)
    assert result == packager.package_project(tmp_path)
    assert [row["path"] for row in result["sources"]] == ["ModelNew.py", "kernel.cpp"]
    assert result["build"]["build_dir"] == "out"


@pytest.mark.parametrize("path", ["../outside.cpp", "/outside.cpp"])
def test_manifest_cannot_read_outside_project(tmp_path, packager, path):
    project(tmp_path, kernel_sources=[path])
    with pytest.raises(ValueError, match="inside"):
        packager.package_project(tmp_path)


def test_source_symlink_is_rejected(tmp_path, packager):
    project(tmp_path)
    (tmp_path / "linked.cpp").symlink_to(tmp_path / "kernel.cpp")
    with pytest.raises(ValueError, match="symlink"):
        packager.package_project(tmp_path)


def test_maca_source_suffix_is_preserved(tmp_path, packager):
    project(tmp_path, kernel_sources=["kernel.maca"])
    (tmp_path / "kernel.cpp").rename(tmp_path / "kernel.maca")
    assert "kernel.maca" in {row["path"] for row in packager.package_project(tmp_path)["sources"]}
