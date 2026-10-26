"""MetaX Triton candidates; specialization occurs at the first real launch."""
import hashlib
import importlib
from importlib import metadata
import linecache
from pathlib import Path

from backends.backend_registry import register_backend
from backends.metax_backend import MetaxBackend


def _validate_triton():
    try:
        triton = importlib.import_module("triton")
    except (ImportError, OSError) as error:
        raise RuntimeError(f"triton_metax requires the MetaX Triton package: {error}") from error
    version = str(getattr(triton, "__version__", ""))
    if "metax" in version.lower():
        return triton
    # MACA 3.0 exposes plain '3.0.0' in __version__, while its distribution
    # metadata says '3.0.0+metax...'. Bind that evidence to the loaded package;
    # metadata for another installation must not legitimize NVIDIA Triton.
    try:
        distribution = metadata.distribution("triton")
        installed_version = distribution.version
        expected = Path(distribution.locate_file("triton/__init__.py")).resolve()
        actual_file = getattr(triton, "__file__", None)
        actual = Path(actual_file).resolve() if actual_file else None
        matched = actual == expected and "metax" in installed_version.lower()
    except (metadata.PackageNotFoundError, OSError, TypeError, ValueError):
        matched = False
    if not matched:
        raise RuntimeError(
            f"triton_metax requires MetaX Triton; loaded triton {version!r} "
            "does not have matching MetaX package metadata. Install the Triton "
            "wheel supplied with the selected MACA/PyTorch toolchain."
        )
    return triton


@register_backend("triton_metax")
class TritonMetaxBackend(MetaxBackend):
    def compile(self, generated_code, op):
        self.reset_context()
        try:
            torch = self.validate_runtime()
            _validate_triton()
            digest = hashlib.sha256(generated_code.encode("utf-8")).hexdigest()
            filename = f"<triton_metax_{id(self):x}_{digest}>"
            linecache.cache[filename] = (
                len(generated_code), None, generated_code.splitlines(True), filename,
            )
            self._source_files.add(filename)
            context = dict(self.context)
            context.setdefault("__file__", filename)
            exec(compile(generated_code, filename, "exec"), context)
            self.validate_model(context, torch)
            self.context = context
            # Do not call warm_cache or invent tensor signatures. MACA Triton
            # specializes with real inputs during native correctness execution.
            return True, None
        except Exception as error:
            self._clear_sources()
            return False, f"{type(error).__name__}: {error}"
