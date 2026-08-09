#include "argmax_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace llaisys::ops::nvidia {

namespace {

constexpr unsigned int BLOCK_SIZE = 256;
constexpr uint64_t INVALID_INDEX = UINT64_MAX;

static_assert(sizeof(__half) == sizeof(uint16_t));
static_assert(sizeof(__nv_bfloat16) == sizeof(uint16_t));

struct Candidate {
    float value;
    uint64_t index;
};

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

__device__ Candidate selectBetter(Candidate lhs, Candidate rhs) {
    if (lhs.index == INVALID_INDEX) {
        return rhs;
    }
    if (rhs.index == INVALID_INDEX) {
        return lhs;
    }
    if (rhs.value > lhs.value
        || (rhs.value == lhs.value && rhs.index < lhs.index)) {
        return rhs;
    }
    return lhs;
}

template <typename T>
__global__ void argmaxKernel(int64_t *max_idx, T *max_val,
                             const T *vals, size_t numel) {
    const unsigned int tid = threadIdx.x;
    const float first_value = toFloat(vals[0]);
    if (isnan(first_value)) {
        if (tid == 0) {
            *max_idx = 0;
            *max_val = vals[0];
        }
        return;
    }

    Candidate best{-CUDART_INF_F, INVALID_INDEX};
    for (size_t i = tid; i < numel; i += blockDim.x) {
        const float value = toFloat(vals[i]);
        if (isnan(value)) {
            continue;
        }
        best = selectBetter(best, Candidate{value, static_cast<uint64_t>(i)});
    }

    __shared__ Candidate candidates[BLOCK_SIZE];
    candidates[tid] = best;
    __syncthreads();

    for (unsigned int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            candidates[tid] = selectBetter(candidates[tid], candidates[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const uint64_t best_index = candidates[0].index;
        *max_idx = static_cast<int64_t>(best_index);
        *max_val = vals[best_index];
    }
}

void checkCudaLaunch() {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        return;
    }

    const char *message = cudaGetErrorString(error);
    std::cerr << "[ERROR] CUDA Argmax kernel launch failed";
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(message != nullptr ? message : "CUDA Argmax kernel launch failed");
}

template <typename T>
void launchArgmax(std::byte *max_idx, std::byte *max_val,
                  const std::byte *vals, size_t numel, cudaStream_t stream) {
    argmaxKernel<<<1, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);
    checkCudaLaunch();
}

} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel, llaisysStream_t stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(max_idx, max_val, vals, numel, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(max_idx, max_val, vals, numel, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__nv_bfloat16>(max_idx, max_val, vals, numel, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
