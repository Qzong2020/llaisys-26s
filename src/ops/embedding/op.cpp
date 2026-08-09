#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64,
           "Embedding: index must be Int64.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    ASSERT(out->ndim() == 2 && index->ndim() == 1 && weight->ndim() == 2,
           "Embedding: out and weight must be 2D, and index must be 1D.");
    ASSERT(out->shape()[0] == index->shape()[0]
               && out->shape()[1] == weight->shape()[1],
           "Embedding: output shape mismatch.");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              index->numel(), weight->shape()[0], weight->shape()[1],
                              weight->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(),
                                 index->numel(), weight->shape()[1], weight->dtype(),
                                 llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
