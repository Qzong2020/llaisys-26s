#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t idx_len, size_t vocab_size, size_t embd_dim,
               llaisysDataType_t dtype);
}