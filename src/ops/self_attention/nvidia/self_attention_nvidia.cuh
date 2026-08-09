#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q,
                    const std::byte *k, const std::byte *v,
                    float scale, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t hd,
                    llaisysDataType_t dtype, llaisysStream_t stream);
} // namespace llaisys::ops::nvidia
