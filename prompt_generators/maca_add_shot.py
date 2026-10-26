"""One-shot native MACA strategy with a complete, packaged source project."""
import json

from prompt_generators.metax_prompt_utils import (
    EXAMPLE_OP, ROOT, code_block, reference_source,
)
from prompt_generators.prompt_registry import BasePromptStrategy, register_prompt
from scripts.maca_project_to_json import package_project


@register_prompt("maca", "add_shot")
class MacaAddShotPromptStrategy(BasePromptStrategy):
    def generate(self, op) -> str:
        example = package_project(ROOT / "examples/metax/add_bias_maca")
        return "\n\n".join([
            "Optimize the supplied PyTorch Model with native MACA C++ kernels for MetaX GPUs. "
            "Return a self-contained JSON source project. The backend compiles device sources "
            "with mxcc -x maca and host PyTorch bindings with the configured C++ compiler.",
            f"One-shot example: {EXAMPLE_OP}. Original architecture:",
            code_block(reference_source(EXAMPLE_OP), "python"),
            "Expected JSON submission:",
            code_block(json.dumps(example, indent=2, ensure_ascii=False), "json"),
            "The example implements contiguous float32 input with scalar bias. Match the target "
            "reference's actual shapes, strides, dtypes, constructor, broadcasting and output "
            "structure; do not inherit restrictions that do not apply to the target.",
            f"Now optimize the following architecture for problem `{op}`:",
            code_block(reference_source(op), "python"),
            "Return exactly one JSON object, without Markdown fences or other text. Required "
            "top-level fields are entry, build and sources. Set entry.model to "
            '"ModelNew.py::ModelNew". Each sources item must contain path and content strings; '
            "embed every Python file, device source, host binding and header the project needs. "
            "All paths must be relative within the project, without absolute paths or '..'.",
            "Set build.module_name to a valid Python/C identifier, build.kernel_sources to "
            "the MACA device source paths, build.binding_sources to separate host binding paths, "
            "build.include_dirs to relative include directories, and build.build_dir to a "
            "directory separate from submitted sources. Device and binding source lists must "
            "be disjoint. Use PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) in the host binding "
            "or a literal module name matching build.module_name. The backend handles compilation "
            "and linking; do not invoke a compiler or torch.utils.cpp_extension from ModelNew.py.",
            "ModelNew.py must define ModelNew(torch.nn.Module) with the same constructor and "
            "forward interface as Model. The backend injects _maca_extension (the absolute "
            "compiled library path) and _maca_module_name (its module name). Load the library "
            "with importlib.util.spec_from_file_location as shown in the example; do not "
            "hardcode a build path or search for a global extension.",
            "Use native MACA headers such as <mc_runtime.h> and MACA runtime APIs. MetaX "
            "PyTorch uses torch.cuda and reports device tensors as is_cuda; this does not "
            "imply an NVIDIA toolkit. Within torch.cuda.device(x.device), pass "
            "torch.cuda.current_stream(x.device).cuda_stream to the native binding and launch "
            "on that mcStream_t. Keep allocations on the input device, validate native pointer "
            "dtype/layout assumptions, and check launch errors. Do not assume default stream "
            "zero, physical device IDs, NVIDIA PTX/architectures or a CUDA bridge installation.",
            "Use actual runtime inputs and preserve reference semantics. Include only the "
            "implementation project, without tests, benchmark code, precomputed answers or "
            "import-time device execution. Do not synchronize inside forward for timing.",
        ]) + "\n"
