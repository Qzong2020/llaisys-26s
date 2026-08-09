#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace llaisys::ops::nvidia {

namespace {

constexpr unsigned int BLOCK_SIZE = 256;
constexpr size_t MAX_GRID_SIZE = 65535;

template <typename T>
__device__ T zeroValue();

template <>
__device__ float zeroValue<float>() {
    return 0.0f;
}

template <>
__device__ __half zeroValue<__half>() {
    return __float2half_rn(0.0f);
}

template <>
__device__ __nv_bfloat16 zeroValue<__nv_bfloat16>() {
    return __float2bfloat16_rn(0.0f);
}

template <typename T>
__global__ void initializeOutputKernel(T *out, const T *bias,
                                       size_t numel, size_t N) {
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel;
         i += step) {
        out[i] = bias != nullptr ? bias[i % N] : zeroValue<T>();
    }
}

void checkCudaLaunch() {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        return;
    }

    const char *message = cudaGetErrorString(error);
    std::cerr << "[ERROR] CUDA Linear initialization kernel launch failed";
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(message != nullptr ? message : "CUDA Linear kernel launch failed");
}

void checkCublas(cublasStatus_t status, const char *expression) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }

    std::cerr << "[ERROR] cuBLAS Linear call failed: " << expression
              << " (status " << static_cast<int>(status) << ")" << std::endl;
    throw std::runtime_error("cuBLAS Linear call failed");
}

template <typename T>
void initializeOutput(std::byte *out, const std::byte *bias,
                      size_t M, size_t N, cudaStream_t stream) {
    const size_t numel = M * N;
    if (numel == 0) {
        return;
    }

    const size_t required_blocks = (numel - 1) / BLOCK_SIZE + 1;
    const unsigned int grid_size = static_cast<unsigned int>(
        std::min(required_blocks, MAX_GRID_SIZE));
    initializeOutputKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        numel,
        N);
    checkCudaLaunch();
}

void initializeOutput(std::byte *out, const std::byte *bias,
                      size_t M, size_t N, llaisysDataType_t dtype,
                      cudaStream_t stream) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return initializeOutput<float>(out, bias, M, N, stream);
    case LLAISYS_DTYPE_F16:
        return initializeOutput<__half>(out, bias, M, N, stream);
    case LLAISYS_DTYPE_BF16:
        return initializeOutput<__nv_bfloat16>(out, bias, M, N, stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

cudaDataType_t cudaDataType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        return CUDA_R_32F;
    }
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, size_t M, size_t N, size_t K,
            llaisysDataType_t dtype, void *cublas_handle,
            llaisysStream_t stream) {
    CHECK_ARGUMENT(cublas_handle != nullptr, "Linear: cuBLAS handle must not be null");
    CHECK_ARGUMENT(M <= static_cast<size_t>(std::numeric_limits<int>::max())
                       && N <= static_cast<size_t>(std::numeric_limits<int>::max())
                       && K <= static_cast<size_t>(std::numeric_limits<int>::max()),
                   "Linear: dimensions exceed the cuBLAS int32 range");
    CHECK_ARGUMENT(N == 0 || M <= std::numeric_limits<size_t>::max() / N,
                   "Linear: output size overflow");

    if (M == 0 || N == 0) {
        return;
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (bias != nullptr || K == 0) {
        initializeOutput(out, bias, M, N, dtype, cuda_stream);
    }
    if (K == 0) {
        return;
    }

    const float alpha = 1.0f;
    const float beta = bias != nullptr ? 1.0f : 0.0f;
    const cudaDataType_t data_type = cudaDataType(dtype);
    const auto handle = reinterpret_cast<cublasHandle_t>(cublas_handle);

    checkCublas(
        cublasGemmEx(handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                     &alpha,
                     weight, data_type, static_cast<int>(K),
                     in, data_type, static_cast<int>(K),
                     &beta,
                     out, data_type, static_cast<int>(N),
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}

} // namespace llaisys::ops::nvidia
