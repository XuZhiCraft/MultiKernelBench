# OmniKernel integration notes

This repository is the OmniKernel compatibility fork at
https://github.com/XuZhiCraft/MultiKernelBench, based on
https://github.com/wzzll123/MultiKernelBench at the commit in `UPSTREAM.json`.
OmniKernel checks it out as the `MultiKernelBench/` submodule and pins the exact
fork commit. `UPSTREAM.json` retains the original snapshot metadata unchanged
so existing workspace provenance stays compatible. Original licenses and source
headers are preserved. Runtime outputs belong to OmniKernel workspaces, not
this directory.

OmniKernel's adapter is in `src/omnikernel/benchmark_new/multikernelbench/`.
It uses native task references, correctness comparisons and event timing;
OmniKernel provides task staging, process/build isolation, result normalization,
profiling, and candidate/version tracking.

The local compatibility changes are limited to:

- `backends/ascendc_direct_launch_backend.py`: use the selected Python interpreter,
  PyTorch C++ ABI, actual SoC, build type and bounded build parallelism. These
  replace upstream's fixed interpreter/SoC/ABI assumptions. The adapter selects
  Release builds on Ascend910_9382 with the dav-2201 instruction architecture.
- `config.py`: defaults for the optional direct-launch build settings. The
  adapter configures them in a fresh child process before importing backends.
- `utils/correctness.py`: optional progress reporting through
  `context['_evaluation_progress']`, so the adapter can distinguish completed
  comparisons from runtime failures and empty input groups. Numerical
  comparison, input generation and tolerances remain native.

Public `ascendc` selects `ascendc_direct_launch`. The legacy `ascendc` backend
with shared operator installation directories remains upstream material and is
not selected by OmniKernel. Public `triton` on Ascend selects `triton_ascend`.

MetaX additions retain the existing platform modules and numerical procedures:

- `backends/metax_backend.py` provides the shared MetaX runtime and native
  correctness/event-timing integration. `triton_metax_backend.py` registers
  MetaX Triton independently of the existing `triton` backend.
- `backends/maca_backend.py` and `maca_compiler.py` implement JSON source projects
  compiled by real `mxcc` and a host C++ compiler, with discovered host headers,
  the installed PyTorch ABI and a private build directory. `config.py` adds
  optional `maca_*` settings without changing the Ascend defaults.
- `utils/metax_environment.py` and the single-runner bootstrap prepare MACA
  before importing torch. `utils/utils.py` loads OpenAI only when creating an
  LLM client, allowing evaluation without that optional dependency.
- `scripts/maca_project_to_json.py` packages ordinary source projects.
  `examples/metax/` contains small Triton and MACA add-bias implementations;
  `tests/` checks these new contracts with CPU fixtures.
- `prompt_generators/triton_metax_add_shot.py` and `maca_add_shot.py` register
  independent native generation strategies. They pair the add-bias reference
  with those same examples, and construct prompts without accelerator or LLM
  imports. MACA examples use the shared source-project packager. See the native
  README's "Generate MetaX candidates" section for offline previews and
  `generate_and_write.py --language triton_metax|maca --strategy add_shot`
  usage. Importing the generation entry point and constructing these prompts
  requires no device dependencies or MACA environment.

The parent adapter maps `metax`/`maca` + public `triton` to `triton_metax`, and
public `maca` to the native `maca` backend. MACA candidates live in
`solution/maca/`; their current PyTorch stream is passed explicitly to native
launchers. Both paths reuse the hardware device pool and normal result
archiving. Standalone builds default to the user's cache, while the parent
adapter always supplies a workspace-local build root.

The upstream default ReLU task allocates a very large tensor. For an initial
native smoke run, `broadcast/add_bias_broadcast` is a small task with unchanged
reference inputs. See the adapter README for commands and measurement scope.
