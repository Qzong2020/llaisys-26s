#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_(std::byte *attn_val, const std::byte *q,
                     const std::byte *k, const std::byte *v,
                     float scale, size_t qlen, size_t kvlen,
                     size_t nh, size_t nkvh, size_t hd) {
    const T *Q = reinterpret_cast<const T *>(q);
    const T *K = reinterpret_cast<const T *>(k);
    const T *V = reinterpret_cast<const T *>(v);
    T *Y = reinterpret_cast<T *>(attn_val);

    size_t nrep = nh / nkvh;  // GQA: KV 头重复次数

    for (size_t hq = 0; hq < nh; hq++) {
        size_t hkv = hq / nrep;  // 映射到 KV 头

        for (size_t i = 0; i < qlen; i++) {
            // ----- 1. 计算 attention scores -----
            std::vector<float> scores(kvlen);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j < kvlen; j++) {
                // Causal mask: i 只能看到 j <= i + (kvlen - qlen)
                if (j > i + (kvlen - qlen)) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                // 点积 Q[i] · K[j] * scale
                float dot = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    float qi, kj;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                                  std::is_same_v<T, llaisys::fp16_t>) {
                        qi = llaisys::utils::cast<float>(Q[i * nh * hd + hq * hd + d]);
                        kj = llaisys::utils::cast<float>(K[j * nkvh * hd + hkv * hd + d]);
                    } else {
                        qi = static_cast<float>(Q[i * nh * hd + hq * hd + d]);
                        kj = static_cast<float>(K[j * nkvh * hd + hkv * hd + d]);
                    }
                    dot += qi * kj;
                }
                scores[j] = dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            // ----- 2. Softmax（数值稳定版）-----
            float sum_exp = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    scores[j] = 0.0f;  // -inf -> exp(-inf) = 0
                } else {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
            }

            // ----- 3. 加权求和 V -----
            for (size_t d = 0; d < hd; d++) {
                float val = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    if (scores[j] > 0.0f) {
                        float vj;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                                      std::is_same_v<T, llaisys::fp16_t>) {
                            vj = llaisys::utils::cast<float>(V[j * nkvh * hd + hkv * hd + d]);
                        } else {
                            vj = static_cast<float>(V[j * nkvh * hd + hkv * hd + d]);
                        }
                        val += scores[j] * vj;
                    }
                }
                val /= sum_exp;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                              std::is_same_v<T, llaisys::fp16_t>) {
                    Y[i * nh * hd + hq * hd + d] = llaisys::utils::cast<T>(val);
                } else {
                    Y[i * nh * hd + hq * hd + d] = static_cast<T>(val);
                }
            }
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q,
                    const std::byte *k, const std::byte *v,
                    float scale, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t hd,
                    llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(attn_val, q, k, v, scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(attn_val, q, k, v, scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(attn_val, q, k, v, scale, qlen, kvlen, nh, nkvh, hd);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu