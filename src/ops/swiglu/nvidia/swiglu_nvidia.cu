#include "swiglu_nvidia.cuh"

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
__device__ T swigluValue(T gate, T up) {
    const float gate_f32 = toFloat(gate);
    const float up_f32 = toFloat(up);

    const T exp_value = fromFloat<T>(expf(-gate_f32));
    const T denominator = fromFloat<T>(1.0f + toFloat(exp_value));
    const T silu = fromFloat<T>(gate_f32 / toFloat(denominator));
    return fromFloat<T>(up_f32 * toFloat(silu));
}

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += stride) {
        const T gate_value = gate[i];
        const T up_value = up[i];
        out[i] = swigluValue(gate_value, up_value);
    }
}

void checkCudaLaunch() {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        return;
    }

    const char *message = cudaGetErrorString(error);
    std::cerr << "[ERROR] CUDA SwiGLU kernel launch failed";
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(message != nullptr ? message : "CUDA SwiGLU kernel launch failed");
}

template <typename T>
void launchSwiglu(std::byte *out, const std::byte *gate, const std::byte *up,
                   size_t numel, cudaStream_t stream) {
    constexpr unsigned int block_size = 256;
    constexpr size_t max_grid_size = 65535;
    const size_t required_grid_size = (numel + block_size - 1) / block_size;
    const unsigned int grid_size = static_cast<unsigned int>(
        required_grid_size < max_grid_size ? required_grid_size : max_grid_size);

    swigluKernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
    checkCudaLaunch();
}

} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
             size_t numel, llaisysDataType_t dtype, llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchSwiglu<float>(out, gate, up, numel, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchSwiglu<__half>(out, gate, up, numel, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchSwiglu<__nv_bfloat16>(out, gate, up, numel, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
