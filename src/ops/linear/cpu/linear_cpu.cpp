#include "linear_cpu.hpp"
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void linear_(std::byte *out, const std::byte *in, const std::byte *weight,
             const std::byte *bias, size_t M, size_t N, size_t K) {
    const T *X = reinterpret_cast<const T *>(in);
    const T *W = reinterpret_cast<const T *>(weight);
    const T *B = reinterpret_cast<const T *>(bias);
    T *Y = reinterpret_cast<T *>(out);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                              std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(X[i * K + k]) *
                           llaisys::utils::cast<float>(W[j * K + k]);
                } else {
                    sum += static_cast<float>(X[i * K + k]) *
                           static_cast<float>(W[j * K + k]);
                }
            }
            if (B != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                              std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(B[j]);
                } else {
                    sum += static_cast<float>(B[j]);
                }
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                Y[i * N + j] = llaisys::utils::cast<T>(sum);
            } else {
                Y[i * N + j] = static_cast<T>(sum);
            }
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, size_t M, size_t N, size_t K,
            llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(out, in, weight, bias, M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(out, in, weight, bias, M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(out, in, weight, bias, M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu