#include "device_resource.hpp"

#include "cpu/cpu_resource.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/nvidia_resource.cuh"
#endif

namespace llaisys::device {

DeviceResource *createDeviceResource(llaisysDeviceType_t device_type,
                                     int device_id,
                                     llaisysStream_t stream) {
    static_cast<void>(stream);
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        CHECK_ARGUMENT(device_id == 0, "invalid CPU device id");
        return new cpu::Resource();
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return new nvidia::Resource(device_id, stream);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}

} // namespace llaisys::device
