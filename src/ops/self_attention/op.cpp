#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(attn_val->isContiguous() && q->isContiguous() &&
           k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 &&
           k->ndim() == 3 && v->ndim() == 3,
           "SelfAttention: all tensors must be 3D.");

    size_t qlen  = q->shape()[0];
    size_t nh    = q->shape()[1];
    size_t hd    = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh  = k->shape()[1];

    ASSERT(nkvh > 0, "SelfAttention: nkvh must be positive.");
    ASSERT(kvlen >= qlen, "SelfAttention: kvlen must be at least qlen.");
    ASSERT(nh % nkvh == 0, "SelfAttention: nh must be multiple of nkvh (GQA).");
    ASSERT(k->shape()[2] == hd && v->shape()[2] == hd,
           "SelfAttention: K, V head_dim must match Q head_dim.");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh,
           "SelfAttention: V shape mismatch with K.");
    ASSERT(attn_val->shape()[0] == qlen &&
           attn_val->shape()[1] == nh &&
           attn_val->shape()[2] == hd,
           "SelfAttention: output shape mismatch.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, qlen, kvlen, nh, nkvh, hd, attn_val->dtype());
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, qlen, kvlen, nh, nkvh, hd, attn_val->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            scale, qlen, kvlen, nh, nkvh, hd, attn_val->dtype(),
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
