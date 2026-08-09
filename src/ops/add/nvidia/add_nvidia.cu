#include "add_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::ops::nvidia {

namespace {

static_assert(sizeof(__half) == sizeof(uint16_t));
static_assert(sizeof(__nv_bfloat16) == sizeof(uint16_t));

template <typename T>
__device__ T addValue(T a, T b) {
    return a + b;
}

template <>
__device__ __half addValue(__half a, __half b) {
    return __float2half_rn(__half2float(a) + __half2float(b));
}

template <>
__device__ __nv_bfloat16 addValue(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16_rn(__bfloat162float(a) + __bfloat162float(b));
}

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < numel; i += stride) {
        const T lhs = a[i];
        const T rhs = b[i];
        c[i] = addValue(lhs, rhs);
    }
}

void checkCudaLaunch() {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        return;
    }

    const char *message = cudaGetErrorString(error);
    std::cerr << "[ERROR] CUDA Add kernel launch failed";
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(message != nullptr ? message : "CUDA Add kernel launch failed");
}

template <typename T>
void launchAdd(std::byte *c, const std::byte *a, const std::byte *b, size_t numel, cudaStream_t stream) {
    constexpr unsigned int block_size = 256;
    constexpr size_t max_grid_size = 65535;
    const size_t required_grid_size = (numel + block_size - 1) / block_size;
    const unsigned int grid_size = static_cast<unsigned int>(
        required_grid_size < max_grid_size ? required_grid_size : max_grid_size);

    addKernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);
    checkCudaLaunch();
}

} // namespace

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel,
         llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchAdd<float>(c, a, b, numel, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchAdd<__half>(c, a, b, numel, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchAdd<__nv_bfloat16>(c, a, b, numel, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
