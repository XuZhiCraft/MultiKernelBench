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
not selected by OmniKernel. Public `triton` selects `triton_ascend`.

The upstream default ReLU task allocates a very large tensor. For an initial
native smoke run, `broadcast/add_bias_broadcast` is a small task with unchanged
reference inputs. See the adapter README for commands and measurement scope.
