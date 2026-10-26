"""Shared MetaX execution using MACA's torch.cuda compatibility interface.

Importing this module does not import torch or initialize an accelerator. The
calling process must prepare its MACA environment before importing PyTorch.
"""
import importlib
import linecache

from backends.backend_registry import Backend


class MetaxBackend(Backend):
    _CONTEXT_METADATA = (
        "__name__", "__file__", "__package__", "__spec__", "__loader__",
        "_evaluation_progress",
    )

    def __init__(self):
        self.context = {}
        self._device = None
        self._torch = None
        self._source_files = set()

    def validate_runtime(self):
        """Require the loaded MetaX wheel, without querying or initializing GPUs."""
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "MetaX requires a working MetaX PyTorch wheel and the MACA library "
                f"environment prepared before importing torch: {error}"
            ) from error
        version = str(getattr(torch, "__version__", ""))
        internal = getattr(torch, "_utils_internal", None)
        if "metax" not in version.lower() and getattr(internal, "USE_MACA", False) is not True:
            raise RuntimeError(
                f"MetaX backend requires MetaX PyTorch; loaded torch {version!r} "
                "has neither a metax version marker nor USE_MACA=True. "
                "A CUDA-compatible device API does not identify a MetaX wheel."
            )
        self._torch = torch
        return torch

    @property
    def device(self):
        if self._device is None:
            self._device = self.get_device()
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def get_device(self):
        return self.validate_runtime().device("cuda:0")

    def get_hardware_name(self):
        torch = self.validate_runtime()
        return torch.cuda.get_device_name(device=self.device)

    def _clear_sources(self):
        for filename in self._source_files:
            linecache.cache.pop(filename, None)
        self._source_files.clear()

    def reset_context(self):
        """Drop the previous candidate, retaining only caller-supplied metadata."""
        metadata = {key: self.context[key] for key in self._CONTEXT_METADATA if key in self.context}
        metadata.setdefault("__name__", "__multikernelbench_candidate__")
        self._clear_sources()
        self.context = metadata
        return self.context

    @staticmethod
    def validate_model(context, torch):
        model = context.get("ModelNew")
        if not isinstance(model, type) or not issubclass(model, torch.nn.Module):
            raise ValueError("Candidate must define ModelNew as a torch.nn.Module subclass")

    def correctness_execution(self, ref_src):
        torch = self.validate_runtime()
        from utils.correctness import execute_template

        try:
            exec(compile(ref_src, "<multikernelbench_reference>", "exec"), self.context)
        except Exception as error:
            raise RuntimeError(f"Failed to compile reference model: {error}") from error
        return execute_template(torch.cuda.synchronize, self.device, self.context)

    def time_execution(self, eval_target="ModelNew"):
        torch = self.validate_runtime()
        from utils.performance import time_execution_event_template

        return time_execution_event_template(
            self.context, self.device, torch.cuda.synchronize, torch.cuda.Event, eval_target
        )

    def cleanup(self):
        self.context = {}
        self._clear_sources()
        torch, self._torch = self._torch, None
        # Failed source validation and repeated cleanup must not initialize a
        # device just to clear it. Successful execution uses the selected card.
        if torch is None or not torch.cuda.is_initialized():
            return
        device = self._device if self._device is not None else torch.device("cuda:0")
        with torch.cuda.device(device):
            torch.cuda.synchronize(device=device)
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats(device=device)
