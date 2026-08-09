#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace llaisys::ops::nvidia {

namespace {

template <typename T>
__global__ void embeddingKernel(T *out, const int64_t *index, const T *weight,
                                size_t numel, size_t embd_dim) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += stride) {
        const size_t token = i / embd_dim;
        const size_t column = i % embd_dim;
        const int64_t row = index[token];
        out[i] = weight[static_cast<size_t>(row) * embd_dim + column];
    }
}

void checkCudaLaunch() {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        return;
    }

    const char *message = cudaGetErrorString(error);
    std::cerr << "[ERROR] CUDA Embedding kernel launch failed";
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(message != nullptr ? message : "CUDA Embedding kernel launch failed");
}

template <typename T>
void launchEmbedding(std::byte *out, const std::byte *index, const std::byte *weight,
                     size_t numel, size_t embd_dim, cudaStream_t stream) {
    constexpr unsigned int block_size = 256;
    constexpr size_t max_grid_size = 65535;
    const size_t required_grid_size = (numel + block_size - 1) / block_size;
    const unsigned int grid_size = static_cast<unsigned int>(
        required_grid_size < max_grid_size ? required_grid_size : max_grid_size);

    embeddingKernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        numel,
        embd_dim);
    checkCudaLaunch();
}

} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t idx_len, size_t embd_dim, llaisysDataType_t dtype,
               llaisysStream_t stream) {
    const size_t numel = idx_len * embd_dim;
    if (numel == 0) {
        return;
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchEmbedding<uint32_t>(out, index, weight, numel, embd_dim, cuda_stream);
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return launchEmbedding<uint16_t>(out, index, weight, numel, embd_dim, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
