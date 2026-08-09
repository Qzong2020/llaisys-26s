#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace llaisys::ops::nvidia {

namespace {

constexpr unsigned int BLOCK_SIZE = 256;
constexpr size_t MAX_GRID_SIZE = 65535;

template <typename T>
__device__ float toFloat(T value);

template <>
__device__ float toFloat<float>(float value) {
    return value;
}

template <>
__device__ float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T fromFloat(float value);

template <>
__device__ float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void selfAttentionKernel(T *attn_val, const T *q,
                                    const T *k, const T *v,
                                    float scale, size_t qlen, size_t kvlen,
                                    size_t nh, size_t nkvh, size_t hd,
                                    size_t output_tiles, size_t task_count) {
    __shared__ float reduction[BLOCK_SIZE];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float previous_weight;
    __shared__ float current_weight;

    const size_t thread_id = threadIdx.x;
    const size_t nrep = nh / nkvh;

    for (size_t task = blockIdx.x; task < task_count; task += gridDim.x) {
        const size_t output_tile = task % output_tiles;
        const size_t query_head = task / output_tiles;
        const size_t head = query_head % nh;
        const size_t query_index = query_head / nh;
        const size_t kv_head = head / nrep;
        const size_t output_dim = output_tile * BLOCK_SIZE + thread_id;

        float output_accumulator = 0.0f;
        if (thread_id == 0) {
            running_max = -CUDART_INF_F;
            running_sum = 0.0f;
            previous_weight = 0.0f;
            current_weight = 0.0f;
        }
        __syncthreads();

        const size_t query_base = (query_index * nh + head) * hd;
        const size_t causal_limit = query_index + (kvlen - qlen);
        for (size_t key_index = 0; key_index <= causal_limit; ++key_index) {
            const size_t key_base = (key_index * nkvh + kv_head) * hd;
            float partial_dot = 0.0f;
            for (size_t dim = thread_id; dim < hd; dim += BLOCK_SIZE) {
                partial_dot += toFloat(q[query_base + dim])
                             * toFloat(k[key_base + dim]);
            }
            reduction[thread_id] = partial_dot;
            __syncthreads();

            for (unsigned int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
                if (thread_id < offset) {
                    reduction[thread_id] += reduction[thread_id + offset];
                }
                __syncthreads();
            }

            if (thread_id == 0) {
                const float score = reduction[0] * scale;
                if (running_sum == 0.0f) {
                    running_max = score;
                    running_sum = 1.0f;
                    previous_weight = 0.0f;
                    current_weight = 1.0f;
                } else {
                    const float next_max = fmaxf(running_max, score);
                    previous_weight = expf(running_max - next_max);
                    current_weight = expf(score - next_max);
                    running_sum = running_sum * previous_weight + current_weight;
                    running_max = next_max;
                }
            }
            __syncthreads();

            if (output_dim < hd) {
                output_accumulator = output_accumulator * previous_weight
                                   + current_weight
                                         * toFloat(v[key_base + output_dim]);
            }
            __syncthreads();
        }

        if (output_dim < hd) {
            attn_val[query_base + output_dim] =
                fromFloat<T>(output_accumulator / running_sum);
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
    std::cerr << "[ERROR] CUDA SelfAttention kernel launch failed";
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(
        message != nullptr ? message : "CUDA SelfAttention kernel launch failed");
}

template <typename T>
void launchSelfAttention(std::byte *attn_val, const std::byte *q,
                         const std::byte *k, const std::byte *v,
                         float scale, size_t qlen, size_t kvlen,
                         size_t nh, size_t nkvh, size_t hd,
                         size_t output_tiles, size_t task_count,
                         cudaStream_t stream) {
    const unsigned int grid_size = static_cast<unsigned int>(
        std::min(task_count, MAX_GRID_SIZE));
    selfAttentionKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        scale, qlen, kvlen, nh, nkvh, hd, output_tiles, task_count);
    checkCudaLaunch();
}

} // namespace

void self_attention(std::byte *attn_val, const std::byte *q,
                    const std::byte *k, const std::byte *v,
                    float scale, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t hd,
                    llaisysDataType_t dtype, llaisysStream_t stream) {
    if (qlen == 0 || nh == 0 || hd == 0) {
        return;
    }

    const size_t output_tiles = (hd - 1) / BLOCK_SIZE + 1;
    CHECK_ARGUMENT(qlen <= std::numeric_limits<size_t>::max() / nh,
                   "SelfAttention: query/head task count overflow");
    const size_t query_heads = qlen * nh;
    CHECK_ARGUMENT(query_heads <= std::numeric_limits<size_t>::max() / output_tiles,
                   "SelfAttention: output tile task count overflow");
    const size_t task_count = query_heads * output_tiles;
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    static_assert(sizeof(__half) == 2, "CUDA half storage must be 2 bytes");
    static_assert(sizeof(__nv_bfloat16) == 2,
                  "CUDA bfloat16 storage must be 2 bytes");

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchSelfAttention<float>(attn_val, q, k, v, scale,
                                          qlen, kvlen, nh, nkvh, hd,
                                          output_tiles, task_count, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchSelfAttention<__half>(attn_val, q, k, v, scale,
                                           qlen, kvlen, nh, nkvh, hd,
                                           output_tiles, task_count, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchSelfAttention<__nv_bfloat16>(
            attn_val, q, k, v, scale, qlen, kvlen, nh, nkvh, hd,
            output_tiles, task_count, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
