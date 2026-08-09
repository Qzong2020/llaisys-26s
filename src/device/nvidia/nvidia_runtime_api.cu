#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::device::nvidia {

namespace {

void checkCuda(cudaError_t error, const char *expression, const char *file, int line) {
    if (error == cudaSuccess) {
        return;
    }

    const char *message = cudaGetErrorString(error);
    std::cerr << "[ERROR] CUDA runtime call failed: " << expression << " at " << file << ":" << line;
    if (message != nullptr) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    throw std::runtime_error(message != nullptr ? message : "CUDA runtime call failed");
}

cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Unsupported CUDA memcpy kind");
    }
}

#define LLAISYS_CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

} // namespace

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    LLAISYS_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    LLAISYS_CUDA_CHECK(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    LLAISYS_CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    LLAISYS_CUDA_CHECK(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    LLAISYS_CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}
void streamSynchronize(llaisysStream_t stream) {
    LLAISYS_CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    LLAISYS_CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    LLAISYS_CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    LLAISYS_CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    LLAISYS_CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    LLAISYS_CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    LLAISYS_CUDA_CHECK(
        cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), reinterpret_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia

#undef LLAISYS_CUDA_CHECK
