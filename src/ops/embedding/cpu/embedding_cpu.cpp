#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t idx_len, size_t vocab_size, size_t embd_dim,
               llaisysDataType_t dtype) {
    size_t row_bytes = embd_dim * llaisys::utils::dsize(dtype);
    const int64_t *idx = reinterpret_cast<const int64_t *>(index);

    for (size_t i = 0; i < idx_len; i++) {
        int64_t row = idx[i];
        const std::byte *src = weight + row * row_bytes;
        std::byte *dst = out + i * row_bytes;
        std::memcpy(dst, src, row_bytes);
    }
}
} // namespace llaisys::ops::cpu