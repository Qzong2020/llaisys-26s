#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
private:
    void *_cublas_handle;

public:
    Resource(int device_id, llaisysStream_t stream);
    ~Resource() noexcept override;

    void *cublasHandle() const;
};
} // namespace llaisys::device::nvidia
