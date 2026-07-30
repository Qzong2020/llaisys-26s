#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void rope_(std::byte *out, const std::byte *in,
           const std::vector<float> &sin_table, const std::vector<float> &cos_table,
           size_t seqlen, size_t nhead, size_t d) {
    const T *X = reinterpret_cast<const T *>(in);
    T *Y = reinterpret_cast<T *>(out);
    size_t half_d = d / 2;

    for (size_t s = 0; s < seqlen; s++) {
        for (size_t h = 0; h < nhead; h++) {
            size_t row_offset = s * nhead * d + h * d;
            for (size_t j = 0; j < half_d; j++) {
                float sin_val = sin_table[s * half_d + j];
                float cos_val = cos_table[s * half_d + j];

                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                              std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(X[row_offset + j]);
                    b = llaisys::utils::cast<float>(X[row_offset + j + half_d]);
                } else {
                    a = static_cast<float>(X[row_offset + j]);
                    b = static_cast<float>(X[row_offset + j + half_d]);
                }

                float new_a = a * cos_val - b * sin_val;
                float new_b = b * cos_val + a * sin_val;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                              std::is_same_v<T, llaisys::fp16_t>) {
                    Y[row_offset + j] = llaisys::utils::cast<T>(new_a);
                    Y[row_offset + j + half_d] = llaisys::utils::cast<T>(new_b);
                } else {
                    Y[row_offset + j] = static_cast<T>(new_a);
                    Y[row_offset + j + half_d] = static_cast<T>(new_b);
                }
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, size_t seqlen, size_t nhead, size_t d,
          llaisysDataType_t dtype) {
    size_t half_d = d / 2;
    const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids);

    std::vector<float> sin_table(seqlen * half_d);
    std::vector<float> cos_table(seqlen * half_d);

    for (size_t s = 0; s < seqlen; s++) {
        float p = static_cast<float>(pos[s]);
        for (size_t j = 0; j < half_d; j++) {
            float freq = p / std::pow(theta, 2.0f * j / d);
            sin_table[s * half_d + j] = std::sin(freq);
            cos_table[s * half_d + j] = std::cos(freq);
        }
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(out, in, sin_table, cos_table, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(out, in, sin_table, cos_table, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(out, in, sin_table, cos_table, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu