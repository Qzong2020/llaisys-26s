#include "rope_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace llaisys::ops::nvidia {

namespace {

static_assert(sizeof(__half) == sizeof(uint16_t));
static_assert(sizeof(__nv_bfloat16) == sizeof(uint16_t));

template <typename T>
__device__ float toFloat(T value);

template <>
__device__ float toFloat(float value) {
    return value;
}

template <>
__device__ float toFloat(__half value) {
    return __half2float(value);
}

template <>
__device__ float toFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T fromFloat(float value);

template <>
__device__ float fromFloat(float value) {
    return value;
}

template <>
__device__ __half fromFloat(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 fromFloat(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos_ids,
                           float theta, size_t pair_count,
                           size_t nhead, size_t d) {
    const size_t half_d = d / 2;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count; pair += stride) {
        const size_t j = pair % half_d;
        const size_t vector = pair / half_d;
        const size_t sequence = vector / nhead;
        const size_t row_offset = vector * d;

        const T a_value = in[row_offset + j];
        const T b_value = in[row_offset + j + half_d];
        const float a = toFloat(a_value);
        const float b = toFloat(b_value);

        const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(d);
        const float position = static_cast<float>(pos_ids[sequence]);
        const float angle = position / powf(theta, exponent);
        const float sin_value = sinf(angle);
        const float cos_value = cosf(angle);

        out[row_offset + j] = fromFloat<T>(a * cos_value - b * sin_value);
        out[row_offset + j + half_d] = fromFloat<T>(b * cos_value + a * sin_value);
    }
}

void checkCudaLaunch() {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        return;
    }

    const char *message = cudaGetErrorString(error);
    std::cerr << "[ERROR] CUDA RoPE kernel launch failed";
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(message != nullptr ? message : "CUDA RoPE kernel launch failed");
}

template <typename T>
void launchRope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
                float theta, size_t pair_count, size_t nhead, size_t d,
                cudaStream_t stream) {
    constexpr unsigned int block_size = 256;
    constexpr size_t max_grid_size = 65535;
    const size_t required_grid_size = (pair_count + block_size - 1) / block_size;
    const unsigned int grid_size = static_cast<unsigned int>(
        required_grid_size < max_grid_size ? required_grid_size : max_grid_size);

    ropeKernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        theta,
        pair_count,
        nhead,
        d);
    checkCudaLaunch();
}

} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, size_t seqlen, size_t nhead, size_t d,
          llaisysDataType_t dtype, llaisysStream_t stream) {
    const size_t pair_count = seqlen * nhead * (d / 2);
    if (pair_count == 0) {
        return;
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRope<float>(out, in, pos_ids, theta, pair_count, nhead, d, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchRope<__half>(out, in, pos_ids, theta, pair_count, nhead, d, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchRope<__nv_bfloat16>(out, in, pos_ids, theta, pair_count, nhead, d, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
