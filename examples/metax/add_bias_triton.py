"""Small MetaX Triton candidate for MultiKernelBench add_bias_broadcast."""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_scalar_bias(X, Bias, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0)
    bias = tl.load(Bias)
    tl.store(Out + offsets, values + bias, offsets < N)


class ModelNew(torch.nn.Module):
    def forward(self, x, bias):
        if not x.is_contiguous() or bias.numel() != 1:
            raise ValueError("This smoke candidate requires contiguous x and scalar bias")
        output = torch.empty_like(x)
        count = x.numel()
        _add_scalar_bias[(triton.cdiv(count, 256),)](x, bias, output, count, BLOCK=256)
        return output
