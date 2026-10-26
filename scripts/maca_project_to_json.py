"""Package a MACA source project for eval_single_runner.py --language maca."""
import argparse
import json
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".cxx", ".cu", ".cuh", ".maca", ".c", ".h", ".hh",
                   ".hpp", ".hxx", ".inc", ".inl", ".json"}


def relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("manifest paths must be nonempty strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must stay inside the project: {value}")
    return path


def package_project(directory):
    root = Path(directory).resolve()
    manifest = root / "manifest.json"
    if manifest.is_symlink():
        raise ValueError("manifest symlinks are not supported")
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    entry = spec.get("entry", {}).get("model", "ModelNew.py::ModelNew")
    filename, separator, model = entry.partition("::")
    if separator != "::" or model != "ModelNew":
        raise ValueError("entry.model must be a relative file followed by ::ModelNew")
    relative_path(filename)
    build = spec.get("build", {})
    for key in ("kernel_sources", "binding_sources", "include_dirs"):
        for value in build.get(key, []):
            relative_path(value)
    build_directory = relative_path(build.get("build_dir", "build"))
    sources = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part in {"build", "dist", "__pycache__", "node_modules"}
               for part in relative.parts):
            continue
        if relative == build_directory or build_directory in relative.parents:
            continue
        if path.is_symlink():
            raise ValueError(f"source symlinks are not supported: {relative}")
        if path == manifest or not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        sources.append({"path": relative.as_posix(), "content": path.read_text(encoding="utf-8")})
    if filename not in {source["path"] for source in sources}:
        raise ValueError(f"missing model source: {filename}")
    return {"entry": {"model": entry}, "build": build, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(package_project(args.project_dir), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
