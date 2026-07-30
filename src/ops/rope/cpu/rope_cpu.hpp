#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, size_t seqlen, size_t nhead, size_t d,
          llaisysDataType_t dtype);
}