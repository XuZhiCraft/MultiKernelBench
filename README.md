# MultiKernelBench

A benchmark for evaluating LLMs' ability to generate kernels for various platform. Now supporting CUDA and triton kernels for GPUs, MUSA kernels for Moore Threads GPUs, Ascendc and TileLang kernels for NPUs, pallas kernels for TPUs and SYCL kernels for Intel GPUs. This compatibility fork also supports MetaX Triton and native MACA C++ kernels.

## Directory Structure

```text
MultiKernelBench/
├── ascend_op_projects/     # Ascend operator projects and extensions
├── backends/               # Backend implementations (cuda, triton, ascendc, pallas, sycl, etc.)
├── prompt_generators/      # Prompt strategy implementations
├── prompts/                # Prompt templates and related resources
├── reference/              # PyTorch reference implementations used for correctness checks
│   ├── activation/
│   ├── attention/
│   ├── convolution/
│   ├── matmul/
│   └── ...
├── utils/                  # Utility modules
├── config.py               # Global runtime and model configuration
├── dataset.py              # Dataset loading and task organization
├── generate_and_write.py   # Generate kernels and write to output directory
├── generate_baseline_statistics.py  # Generate baseline statistics across tasks/categories
├── evaluation.py           # End-to-end evaluation entrypoint
└── eval_single_runner.py   # Single task/category evaluation runner
```

`reference/` is organized by `category`. The `--categories` argument should use these directory names (e.g., `activation`, `attention`, `convolution`, `matmul`, etc.).

## Latest News
- **18/06/2026** – Added **MUSA selected-shot prompt strategy** support for **Moore Threads GPUs**; use `--language musa --strategy selected_shot`.
- **11/06/2026** – Added initial **MUSA backend** support for **Moore Threads GPUs**; use `--language musa --strategy add_shot`.
- **28/05/2026** – Added **AscendC direct-launch backend** support for `<<<>>>` kernel launches; use `--language ascendc_direct_launch --strategy add_shot`.
- **28/05/2026** – Added **anti-hack detection** for generated code that replaces custom kernels with PyTorch/Python compute.
- **27/10/2025** – Introduced a new task category featuring 15 attention tasks, including MQA and GQA. 
- **08/10/2025** – Added **TileLang-Ascend backend** support for **Ascend NPUs**.  
- **12/08/2025** – Added **SYCL backend** support for **Intel GPUs** – thanks to **NinaWie** for the contribution!  
- **18/07/2025** – 🎉 Announced the open-source release of **MultiKernelBench**, a **multi-platform benchmark for kernel generation**, now publicly available!


## Quick start

### Set up
```bash
conda create --name multi-kernel-bench python=3.10
conda activate multi-kernel-bench
pip install -r requirements.txt

# For NPU users:
pip install torch-npu==2.7.1

# For Intel GPU (torch xpu) users:
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/xpu

# For Moore Threads GPU (MUSA) users:
# Install the torch_musa package matching your MUSA runtime.
```
You can rent GPU or NPU resources from online platforms such as [autodl](https://www.autodl.com/home). For TPU resources, you can use services like [Google Colab](https://colab.research.google.com/)

### Config
Set configurations in config.py, including temperature and top_p for LLM. For CUDA-based evaluation, set arch_list. For Ascendc evaluation, set ascendc_device = ai_core-<soc_version>.

### Set API keys for LLM
```
export DEEPSEEK_API_KEY=<your deepseek api key>
export DASHSCOPE_API_KEY=<your aliyun api key>
export OPEN_ROUNTER_KEY=<your openrouter api key>
```

### Generate kernels using LLM and write them to output
```bash
python generate_and_write.py --model deepseek-chat --language ascendc --strategy add_shot --categories activation
```
For MUSA generation, use `--language musa --strategy add_shot` or `--language musa --strategy selected_shot`.
Generated code is saved in ```output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}```.

#### Available Arguments

- `--runs`: Number of runs (default: `1`)
- `--model`: Model name (default: `deepseek-chat`)
- `--language`: Language used (default: `cuda`)
- `--strategy`: Prompt strategy type (default: `add_shot`)
- `--categories`: Space-separated list of categories (default: `activation`)  
  Use `all` to include all categories.

### Evalutation
```bash
python evaluation.py --model deepseek-chat --language ascendc --strategy add_shot --categories activation
```
Evaluation result is saved in ```output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}/result_{category}.json```.

### Evaluate a Single Response
Use `eval_single_runner.py` when you already have one generated response file and want to evaluate one task directly:

```bash
python eval_single_runner.py \
  --input prompts/ascendc_direct_launch_model_relu.json \
  --op relu \
  --language ascendc_direct_launch \
  --result relu_result.json
```

The runner writes one JSON object containing compile status, correctness, performance, hardware, and anti-hack detection fields.

## MetaX: Triton and MACA C++

Use the MetaX PyTorch and Triton wheels supplied for the installed MACA SDK.
Do **not** install the generic pinned `torch==2.7.1` from `requirements.txt`
over this vendor environment. The verified stack is C500, MACA 3.0.0.8,
PyTorch 2.6.0+metax3.0.0.3 and Triton 3.0.0+metax3.0.0.3. MetaX uses
`torch.cuda`; `torch.version.maca` can be `None` on a valid vendor installation.
The new backends validate vendor identity without changing the existing CUDA,
Triton, MUSA or Ascend backend registrations.

Run these commands from the MultiKernelBench directory. The standalone runner
prepares `MACA_PATH` (default `/opt/maca`) and the corresponding library paths,
then restarts the interpreter before importing device packages. Direct Python
API users must start their process with the matching MACA environment.
For these two MetaX backends, the runner exits nonzero on compilation,
correctness or timing failure and replaces any previous result with the current
report. The existing platforms retain their original CLI exit behavior.

```bash
MACA_VISIBLE_DEVICES=0 python3 eval_single_runner.py \
  --language triton_metax --op add_bias_broadcast \
  --input examples/metax/add_bias_triton.py --result /tmp/mkb_triton_result.json

python3 scripts/maca_project_to_json.py examples/metax/add_bias_maca \
  --output /tmp/mkb_maca_input.json
MACA_VISIBLE_DEVICES=0 python3 eval_single_runner.py \
  --language maca --op add_bias_broadcast \
  --input /tmp/mkb_maca_input.json --result /tmp/mkb_maca_result.json
```

`triton_metax` accepts a Python `ModelNew` module. JIT specialization uses real
inputs during correctness/warmup, before timed samples; it does not set NVIDIA
architecture flags. `maca` accepts a JSON project with `entry`, `build` and
`sources: [{"path": "...", "content": "..."}]`. The packaging script reads an
ordinary `manifest.json` and source files; see the complete example project.

MACA device sources compile with `mxcc -x maca`; host bindings compile with the
selected C++ compiler and PyTorch ABI. The compiler discovers host C++ headers
and libraries instead of assuming a GCC version. `ModelNew.py` receives
`_maca_extension` and `_maca_module_name` to load the extension. Pass
`torch.cuda.current_stream(x.device).cuda_stream` into the native launcher;
the example uses MACA headers and never assumes default stream zero or a CUDA
bridge installation.

`config.py` exposes `maca_cxx`, `maca_mxcc`, `maca_build_jobs`, `maca_build_type`
and separate host/device/link flag lists. Standalone builds use unique
directories below `$XDG_CACHE_HOME/multikernelbench/maca` or
`~/.cache/multikernelbench/maca`; set `MULTIKERNELBENCH_MACA_BUILD_ROOT` to choose
another root. Source files, objects and `build.log` remain available there.
Evaluation does not require the optional OpenAI client package.

The native correctness tolerances and event timing procedure are unchanged.
Support for a backend does not make NPU-specific reference operators portable.
The existing anti-hack checks vary by backend and are not proof that a candidate
uses only permitted operations. For these examples, the device kernel sources
are explicit and can be inspected.

Standalone visibility selection does not reserve a card. When this checkout is
used by OmniKernel, configure `device_pool.devices` and run through its normal
MultiKernelBench adapter, or wrap a native command with the hardware pool:

```bash
PYTHONPATH=../src OMNIKERNEL_DEVICE_POOL=1 python3 -m omnikernel.hardware.pool run -- \
  python3 eval_single_runner.py --language triton_metax --op add_bias_broadcast \
  --input examples/metax/add_bias_triton.py --result /tmp/mkb_pooled_result.json
```

### Generate MetaX candidates

The native generator has separate `add_shot` strategies for `triton_metax` and
`maca`. Both read `reference/broadcast/add_bias_broadcast.py` and the matching
tested implementation under `examples/metax/`. The MACA strategy embeds a
complete `entry`/`build`/`sources` JSON project using the same packaging helper
as evaluation. The original platform strategies remain unchanged.

To inspect a prompt offline, without importing torch or contacting an LLM:

```bash
python3 - <<'PY'
import importlib
from prompt_generators.prompt_registry import PROMPT_REGISTRY

language = "maca"  # or "triton_metax"
importlib.import_module(f"prompt_generators.{language}_add_shot")
print(PROMPT_REGISTRY[language]["add_shot"].generate("add_bias_broadcast"))
PY
```

With the provider client and credentials configured, generate candidates with:

```bash
python3 generate_and_write.py --model deepseek-chat --language triton_metax \
  --strategy add_shot --categories broadcast
python3 generate_and_write.py --model deepseek-chat --language maca \
  --strategy add_shot --categories broadcast
```

These generation commands contact the configured LLM and write candidates to
`output/<language>/add_shot/...`. Importing the generation entry point and
constructing MetaX prompts loads no device dependencies and needs no MACA
environment. Evaluating the generated candidates requires the runtime described
above.

## Adding a Prompting Strategy for a New or Existing Language

To add a custom prompting strategy, follow these steps:
1. **Create a Python file:**  
   Add a new file under `prompt_generators/` named as:  
   `prompt_generators/{language}_{strategy_name}.py`  

2. **Create a New Strategy Class**

   - Inherit from `BasePromptStrategy`.
   - Implement the `generate(self, op)` method.

2. **Register the Strategy**

   Use the `@register_prompt(language, strategy_name)` decorator with the desired language and strategy name.
## Adding a New Backend

To integrate a new backend, follow these steps:

1. **Create a New Python File**

   Add a new file under `backends/` named as:  
   `backends/{backend_name}.py`

2. **Create a Backend Class**

   - Inherit from `Backend`.
   - Implement all required methods:
     - `get_device()`
     - `get_hardware_name()`
     - `compile(generated_code, op)`
     - `correctness_execution(ref_src)`
     - `time_execution()`
     - `cleanup()` (optional)

3. **Register the Backend**

   Use the `@register_backend(name)` decorator with your backend's unique name.

## Credits

This project uses code from [KernelBench](https://github.com/ScalingIntelligence/KernelBench), licensed under the MIT License.
