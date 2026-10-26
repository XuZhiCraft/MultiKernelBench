"""MetaX MACA C++ JSON projects, with real device and host compilation."""
from __future__ import annotations

import traceback

import config
from backends.backend_registry import register_backend
from backends.metax_backend import MetaxBackend
from backends.maca_compiler import build_extension, parse_project, stage_project


@register_backend("maca")
class MacaBackend(MetaxBackend):
    def __init__(self):
        super().__init__()
        self.work_dir = None

    def compile(self, generated_code, op):
        self.reset_context()
        self.work_dir = None
        try:
            project = parse_project(generated_code)
            torch = self.validate_runtime()
            self.work_dir = stage_project(project, op, config)
            print(f"[maca] project directory: {self.work_dir}")
            extension = build_extension(project, self.work_dir, torch, config)
            self.context.update(_maca_extension=str(extension), _maca_module_name=project.module_name)
            model_path = self.work_dir / project.model_path
            self.context["__file__"] = str(model_path)
            exec(compile(project.sources[project.model_path], str(model_path), "exec"), self.context)
            model = self.context.get(project.model_name)
            if not callable(model):
                raise ValueError(f"Model entry {project.model_name!r} is missing or not callable")
            self.context["ModelNew"] = model
            self.validate_model(self.context, torch)
            return True, None
        except Exception:
            self.reset_context()
            detail = traceback.format_exc()
            if self.work_dir is not None:
                detail += f"\nProject/build diagnostics: {self.work_dir}\n"
            return False, detail

    def cleanup(self):
        super().cleanup()
        # Keep complete generated sources, objects and build.log for diagnostics.
        self.work_dir = None
