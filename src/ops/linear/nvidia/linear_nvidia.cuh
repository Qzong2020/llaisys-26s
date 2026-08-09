#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, size_t M, size_t N, size_t K,
            llaisysDataType_t dtype, void *cublas_handle,
            llaisysStream_t stream);
}
