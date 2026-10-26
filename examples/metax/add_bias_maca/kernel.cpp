#include <mc_runtime.h>
#include <cstdint>

__global__ void add_bias_kernel(const float* x, const float* bias,
                                float* output, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = x[index] + bias[0];
    }
}

extern "C" int launch_add_bias(const float* x, const float* bias, float* output,
                               int64_t count, uint64_t stream) {
    if (count == 0) return 0;
    constexpr int block = 256;
    const auto grid = static_cast<unsigned int>((count + block - 1) / block);
    add_bias_kernel<<<grid, block, 0, reinterpret_cast<mcStream_t>(stream)>>>(
        x, bias, output, count);
    return static_cast<int>(mcGetLastError());
}
