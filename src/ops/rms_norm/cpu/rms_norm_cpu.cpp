#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_(std::byte *out, const std::byte *in, const std::byte *weight,
               float eps, size_t M, size_t d) {
    const T *X = reinterpret_cast<const T *>(in);
    const T *W = reinterpret_cast<const T *>(weight);
    T *Y = reinterpret_cast<T *>(out);

    for (size_t i = 0; i < M; i++) {
        // 1. sum(x_j^2)  for row i
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                float val = llaisys::utils::cast<float>(X[i * d + j]);
                sum_sq += val * val;
            } else {
                float val = static_cast<float>(X[i * d + j]);
                sum_sq += val * val;
            }
        }

        // 2. rsqrt = 1 / sqrt(mean(x^2) + eps)
        float rsqrt = 1.0f / std::sqrt(sum_sq / static_cast<float>(d) + eps);

        // 3. Y[i][j] = X[i][j] * rsqrt * W[j]
        for (size_t j = 0; j < d; j++) {
            float w_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                w_val = llaisys::utils::cast<float>(W[j]);
            } else {
                w_val = static_cast<float>(W[j]);
            }

            float x_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                x_val = llaisys::utils::cast<float>(X[i * d + j]);
            } else {
                x_val = static_cast<float>(X[i * d + j]);
            }

            float result = x_val * rsqrt * w_val;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                Y[i * d + j] = llaisys::utils::cast<T>(result);
            } else {
                Y[i * d + j] = static_cast<T>(result);
            }
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, size_t M, size_t d, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(out, in, weight, eps, M, d);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(out, in, weight, eps, M, d);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(out, in, weight, eps, M, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu