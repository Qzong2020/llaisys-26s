#include "nvidia_resource.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::device::nvidia {

namespace {

void checkCublas(cublasStatus_t status, const char *expression) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }

    std::cerr << "[ERROR] cuBLAS call failed: " << expression
              << " (status " << static_cast<int>(status) << ")" << std::endl;
    throw std::runtime_error("cuBLAS resource initialization failed");
}

} // namespace

Resource::Resource(int device_id, llaisysStream_t stream)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id),
      _cublas_handle(nullptr) {
    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    try {
        checkCublas(cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream)),
                    "cublasSetStream");
        checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                    "cublasSetPointerMode");
        const auto math_mode = static_cast<cublasMath_t>(
            CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        checkCublas(cublasSetMathMode(handle, math_mode), "cublasSetMathMode");
    } catch (...) {
        const cublasStatus_t destroy_status = cublasDestroy(handle);
        if (destroy_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "[ERROR] cuBLAS cleanup after initialization failure returned status "
                      << static_cast<int>(destroy_status) << std::endl;
        }
        throw;
    }

    _cublas_handle = reinterpret_cast<void *>(handle);
}

Resource::~Resource() noexcept {
    if (_cublas_handle == nullptr) {
        return;
    }

    const cublasStatus_t status = cublasDestroy(
        reinterpret_cast<cublasHandle_t>(_cublas_handle));
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] cublasDestroy returned status "
                  << static_cast<int>(status) << std::endl;
    }
    _cublas_handle = nullptr;
}

void *Resource::cublasHandle() const {
    return _cublas_handle;
}

} // namespace llaisys::device::nvidia
