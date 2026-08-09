#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t idx_len, size_t embd_dim, llaisysDataType_t dtype,
               llaisysStream_t stream);
}
