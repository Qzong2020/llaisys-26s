#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../../device/nvidia/nvidia_resource.cuh"
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: out, in, weight must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
           "Linear: out, in, weight must be 2D tensors.");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    ASSERT(weight->shape()[1] == K,
           "Linear: weight.shape[1] must match in.shape[1].");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N,
           "Linear: output shape mismatch.");
    if (bias) {
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == N,
               "Linear: bias shape mismatch.");
    }

    // CPU 调度
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           M, N, K, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           M, N, K, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        auto &runtime = llaisys::core::context().runtime();
        auto &resource = static_cast<llaisys::device::nvidia::Resource &>(
            runtime.resource());
        return nvidia::linear(out->data(), in->data(), weight->data(),
                              bias ? bias->data() : nullptr,
                              M, N, K, out->dtype(),
                              resource.cublasHandle(), runtime.stream());
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
