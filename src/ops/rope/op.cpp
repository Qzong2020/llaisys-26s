#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // pos_ids 固定是 Int64
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64,
           "RoPE: pos_ids must be Int64.");

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    ASSERT(out->ndim() == 3 && in->ndim() == 3,
           "RoPE: out, in must be 3D tensors.");
    ASSERT(pos_ids->ndim() == 1,
           "RoPE: pos_ids must be 1D tensor.");

    size_t seqlen = in->shape()[0];
    size_t nhead  = in->shape()[1];
    size_t d      = in->shape()[2];

    ASSERT(d % 2 == 0, "RoPE: head dimension d must be even.");
    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nhead && out->shape()[2] == d,
           "RoPE: output shape mismatch.");
    ASSERT(pos_ids->shape()[0] == seqlen,
           "RoPE: pos_ids length must match seqlen.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         theta, seqlen, nhead, d, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         theta, seqlen, nhead, d, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(),
                            theta, seqlen, nhead, d, out->dtype(),
                            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
