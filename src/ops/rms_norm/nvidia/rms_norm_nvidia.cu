#include "rms_norm_nvidia.cuh"

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

// The hardware warp is 32 lanes on every target (compute capability 8.0).
// We hardcode 32 instead of the `warpSize` constant: on the MetaX MACA
// toolchain (cu-bridge/mxcc) `warpSize` compiles to 64 while the real warp is
// 32, so `warpSize / 2` yields an extra invalid __shfl_down_sync step that
// doubles (and stacks) the partial sums. nvcc also uses 32, so this matches
// the NVIDIA build exactly.
constexpr unsigned int kThreadsPerWarp = 32;

__device__ float warpReduceSum(float value) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float blockReduceSum(float value) {
    __shared__ float warp_sums[32];

    const unsigned int lane = threadIdx.x % kThreadsPerWarp;
    const unsigned int warp = threadIdx.x / kThreadsPerWarp;
    value = warpReduceSum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    const unsigned int warp_count =
        (blockDim.x + kThreadsPerWarp - 1) / kThreadsPerWarp;
    value = threadIdx.x < warp_count ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warpReduceSum(value);
    }
    return value;
}

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight,
                              float eps, size_t M, size_t d) {
    __shared__ float row_scale;

    for (size_t row = blockIdx.x; row < M; row += gridDim.x) {
        float sum_sq = 0.0f;
        for (size_t column = threadIdx.x; column < d; column += blockDim.x) {
            const float value = toFloat(in[row * d + column]);
            sum_sq += value * value;
        }

        sum_sq = blockReduceSum(sum_sq);
        if (threadIdx.x == 0) {
            row_scale = rsqrtf(sum_sq / static_cast<float>(d) + eps);
        }
        __syncthreads();

        const float scale = row_scale;
        for (size_t column = threadIdx.x; column < d; column += blockDim.x) {
            const float value = toFloat(in[row * d + column]);
            const float weight_value = toFloat(weight[column]);
            out[row * d + column] = fromFloat<T>(value * scale * weight_value);
        }
        __syncthreads();
    }
}

void checkCudaLaunch() {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        return;
    }

    const char *message = cudaGetErrorString(error);
    std::cerr << "[ERROR] CUDA RMSNorm kernel launch failed";
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(message != nullptr ? message : "CUDA RMSNorm kernel launch failed");
}

template <typename T>
void launchRmsNorm(std::byte *out, const std::byte *in, const std::byte *weight,
                   float eps, size_t M, size_t d, cudaStream_t stream) {
    constexpr unsigned int block_size = 256;
    constexpr size_t max_grid_size = 65535;
    const unsigned int grid_size = static_cast<unsigned int>(
        M < max_grid_size ? M : max_grid_size);

    rmsNormKernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        eps,
        M,
        d);
    checkCudaLaunch();
}

} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, size_t M, size_t d, llaisysDataType_t dtype,
              llaisysStream_t stream) {
    if (M == 0 || d == 0) {
        return;
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(out, in, weight, eps, M, d, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(out, in, weight, eps, M, d, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__nv_bfloat16>(out, in, weight, eps, M, d, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
