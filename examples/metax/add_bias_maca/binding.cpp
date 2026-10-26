#include <torch/extension.h>
#include <cstdint>

extern "C" int launch_add_bias(const float* x, const float* bias, float* output,
                               int64_t count, uint64_t stream);

torch::Tensor add_bias(torch::Tensor x, torch::Tensor bias, uint64_t stream) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "Inputs must be on a MetaX device");
    TORCH_CHECK(x.device() == bias.device(), "Inputs must be on the same device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
                "Inputs must be float32");
    TORCH_CHECK(x.is_contiguous() && bias.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(bias.numel() == 1, "Expected scalar bias");
    auto output = torch::empty_like(x);
    const int status = launch_add_bias(x.data_ptr<float>(), bias.data_ptr<float>(),
                                      output.data_ptr<float>(), x.numel(), stream);
    TORCH_CHECK(status == 0, "MACA launch failed with error code ", status);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("add_bias", &add_bias);
}
