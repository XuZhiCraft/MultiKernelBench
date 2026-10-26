"""Scalar add using a native MACA kernel on PyTorch's current stream."""
import importlib.util

import torch


_spec = importlib.util.spec_from_file_location(_maca_module_name, _maca_extension)
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)


class ModelNew(torch.nn.Module):
    def forward(self, x, bias):
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream(x.device).cuda_stream
            return _native.add_bias(x, bias, stream)
