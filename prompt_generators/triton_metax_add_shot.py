"""One-shot MetaX Triton strategy using the native add-bias example."""
from prompt_generators.metax_prompt_utils import (
    EXAMPLE_OP, ROOT, code_block, reference_source,
)
from prompt_generators.prompt_registry import BasePromptStrategy, register_prompt


@register_prompt("triton_metax", "add_shot")
class TritonMetaxAddShotPromptStrategy(BasePromptStrategy):
    def generate(self, op) -> str:
        example = (ROOT / "examples/metax/add_bias_triton.py").read_text(encoding="utf-8")
        return "\n\n".join([
            "Optimize the supplied PyTorch Model with custom Triton kernels for MetaX GPUs "
            "using the installed MetaX PyTorch and MetaX Triton distributions. "
            "You may fuse operators or change algorithms while preserving the reference semantics.",
            "MetaX PyTorch exposes tensors and streams through torch.cuda. This API does not "
            "imply NVIDIA hardware. Use portable triton.language operations; do not assume "
            "NVIDIA PTX, CUDA architecture names, a fixed warp width, or NVIDIA-only intrinsics. "
            "Do not set TORCH_CUDA_ARCH_LIST or depend on torch.version.maca being populated.",
            f"One-shot example: {EXAMPLE_OP}. Original architecture:",
            code_block(reference_source(EXAMPLE_OP), "python"),
            "MetaX Triton implementation:",
            code_block(example, "python"),
            "The example handles contiguous float32 input and scalar bias for its reference. "
            "Determine shapes, strides, dtypes, broadcasting and output structure from the target "
            "below; do not copy the example's restrictions onto a different task. "
            "Handle partial blocks with valid masks. Keep allocations on the input device and "
            "use the current PyTorch stream; do not force a physical device or synchronize "
            "inside forward for timing.",
            f"Now optimize the following architecture for problem `{op}`:",
            code_block(reference_source(op), "python"),
            "Return one Python code block defining ModelNew as a torch.nn.Module, with the "
            "same constructor and forward inputs and compatible outputs as Model. Include all "
            "imports and Triton kernels required to run it. Use actual runtime inputs; do not "
            "precompute answers from the example inputs. Output only the implementation, "
            "without explanations, tests, benchmark code or import-time device execution.",
        ]) + "\n"
