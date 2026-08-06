#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void swiglu_(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    const T *G = reinterpret_cast<const T *>(gate);
    const T *U = reinterpret_cast<const T *>(up);
    T *Y = reinterpret_cast<T *>(out);

    for (size_t i = 0; i < numel; i++) {
        float g = llaisys::utils::cast<float>(G[i]);
        float u = llaisys::utils::cast<float>(U[i]);
        // sigmoid(g) = 1 / (1 + e^(-g))
        float sig = 1.0f / (1.0f + std::exp(-g));
        float result = u * sig * g;
        Y[i] = llaisys::utils::cast<T>(result);
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            size_t numel, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu